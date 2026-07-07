#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "src/build_flags.h"
#include "src/card_abstraction.h"
#include "src/card_utils.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/strategy_tables.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include "src/training_range.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kAutoWarmupNoGrowthLimit = 100;

// Atomically add `delta` to `*target` using a CAS loop.
// Safe to call from multiple threads concurrently. Relaxed ordering is
// sufficient: CFR+ convergence is robust to the bounded noise from concurrent
// stale reads (equivalent to async-regret MCCFR).
inline void AtomicFloatAdd(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float new_val;
    std::memcpy(&new_val, &old_bits, sizeof(float));
    new_val += delta;
    std::memcpy(&new_bits, &new_val, sizeof(float));
  } while (!__atomic_compare_exchange_n(
      reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
      /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

// Atomically apply CFR+ regret update: new = max(0, old + delta).
// Uses the same CAS loop as AtomicFloatAdd but clips at zero.
inline int64_t AtomicCFRPlusRegretUpdate(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  bool exchanged = false;
  int64_t retries = 0;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float old_val;
    std::memcpy(&old_val, &old_bits, sizeof(float));
    float new_val = std::max(0.0f, old_val + delta);
    std::memcpy(&new_bits, &new_val, sizeof(float));
    exchanged = __atomic_compare_exchange_n(
        reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
        /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    if constexpr (kCasRetryStatsEnabled) {
      if (!exchanged) {
        ++retries;
      }
    }
  } while (!exchanged);
  if constexpr (kCasRetryStatsEnabled) {
    return retries;
  }
  return 0;
}

// Relaxed atomic load of a float (naturally-aligned 32-bit load is atomic
// on all supported architectures, but we use __atomic_load_n for clarity).
inline float AtomicFloatLoad(const float* src) {
  int32_t bits = __atomic_load_n(reinterpret_cast<const int32_t*>(src),
                                 __ATOMIC_RELAXED);
  float val;
  std::memcpy(&val, &bits, sizeof(float));
  return val;
}

std::optional<double> UtilityBeforeShowdown(const CompactPublicState& state,
                                            uint8_t board_count) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }
  if (board_count + 2 < 5) {
    return 0.0;
  }
  return std::nullopt;
}

double ShowdownUtilityFromComparison(const CompactPublicState& state,
                                     int comparison) {
  const double player_a_contribution = state.player_contribution[0];
  if (comparison > 0) {
    return state.pot - player_a_contribution;
  }
  if (comparison < 0) {
    return -player_a_contribution;
  }
  return (state.pot / 2.0) - player_a_contribution;
}

size_t ScratchDepthReserve(const SolverConfig& config, int max_depth) {
  if (max_depth > 0) {
    return static_cast<size_t>(max_depth) + 2;
  }
  const int stack_size = std::max(0, config.starting_stack_size);
  return std::max<size_t>(32, static_cast<size_t>(stack_size) + 12);
}

void PublicCompatibleRangeInto(const TrainingRangeView& hands,
                               CardMask board_mask,
                               TrainingRangeView& compatible_hands) {
  if (hands.empty()) {
    compatible_hands.clear();
    return;
  }

  compatible_hands.reset_to_filtered();
  for (size_t i = 0; i < hands.size(); ++i) {
    if (hands.weight(i) > 0.0 && (hands.mask(i) & board_mask) == 0) {
      compatible_hands.add(hands.combo(i), hands.weight(i));
    }
  }
}

CompactPublicState StateWithBoardFrom(CompactPublicState state,
                                      const CompactPublicState& board_source) {
  state.board_cards = board_source.board_cards;
  state.board_count = board_source.board_count;
  state.board_mask = board_source.board_mask;
  return state;
}

template <typename Callback>
bool ForEachNextStreetDeal(const CompactPublicState& state,
                           Callback callback) {
  const int remaining_board_slots =
      std::max(0, kMaxBoardCards - static_cast<int>(state.board_count));
  const int count =
      std::min(CardsForNextStreet(state.street), remaining_board_slots);
  if (count <= 0) {
    return callback(absl::Span<const CardId>());
  }

  std::array<CardId, kDeckCardCount> candidates = {};
  int candidate_count = 0;
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    const CardId candidate = static_cast<CardId>(card_id);
    if ((state.board_mask & CardBit(candidate)) == 0) {
      candidates[static_cast<size_t>(candidate_count)] = candidate;
      ++candidate_count;
    }
  }
  if (candidate_count < count) {
    throw std::runtime_error("Not enough cards to enumerate next street");
  }

  absl::InlinedVector<CardId, 5> cards;
  cards.resize(static_cast<size_t>(count));
  auto choose = [&](auto& self, int start, int depth) -> bool {
    if (depth == count) {
      return callback(absl::Span<const CardId>(cards));
    }
    const int remaining = count - depth;
    for (int i = start; i <= candidate_count - remaining; ++i) {
      cards[static_cast<size_t>(depth)] = candidates[static_cast<size_t>(i)];
      if (!self(self, i + 1, depth + 1)) {
        return false;
      }
    }
    return true;
  };
  return choose(choose, 0, 0);
}

StreetKind StreetAfterChance(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return StreetKind::kFlop;
    case StreetKind::kFlop:
      return StreetKind::kTurn;
    case StreetKind::kTurn:
    case StreetKind::kRiver:
      return StreetKind::kRiver;
  }
}

int BoardCardsForStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 0;
    case StreetKind::kFlop:
      return 3;
    case StreetKind::kTurn:
      return 4;
    case StreetKind::kRiver:
      return 5;
  }
}

struct CoarseChanceTransitionTemplate {
  CompactPublicState parent_board_state;
  absl::InlinedVector<CardId, 5> cards;
};

using CoarseChanceTransitionMap =
    absl::flat_hash_map<PublicBucketId,
                        std::vector<CoarseChanceTransitionTemplate>>;

template <typename Callback>
void ForEachCardCombination(int count, CardMask blocked_mask,
                            Callback callback) {
  absl::InlinedVector<CardId, 5> cards;
  cards.resize(static_cast<size_t>(count));
  auto choose = [&](auto& self, int start, int depth) -> void {
    if (depth == count) {
      callback(absl::Span<const CardId>(cards));
      return;
    }
    const int remaining = count - depth;
    for (int card_id = start; card_id <= kDeckCardCount - remaining;
         ++card_id) {
      const CardId card = static_cast<CardId>(card_id);
      if ((blocked_mask & CardBit(card)) != 0) {
        continue;
      }
      cards[static_cast<size_t>(depth)] = card;
      self(self, card_id + 1, depth + 1);
    }
  };
  choose(choose, 0, 0);
}

CoarseChanceTransitionMap BuildCoarseChanceTransitions(StreetKind street) {
  CoarseChanceTransitionMap transitions;
  const int board_count = BoardCardsForStreet(street);
  const int deal_count = CardsForNextStreet(street);
  if (deal_count <= 0) {
    return transitions;
  }

  CardAbstraction abstraction;
  absl::flat_hash_map<uint64_t, bool> seen;
  ForEachCardCombination(board_count, 0, [&](absl::Span<const CardId> board) {
    CompactPublicState parent;
    parent.street = street;
    for (CardId card : board) {
      AddBoardCard(parent, card);
    }
    const PublicBucketId parent_bucket = abstraction.public_bucket(parent);
    ForEachCardCombination(deal_count, parent.board_mask,
                           [&](absl::Span<const CardId> cards) {
      CompactPublicState child = parent;
      child.street = StreetAfterChance(street);
      for (CardId card : cards) {
        AddBoardCard(child, card);
      }
      const PublicBucketId child_bucket = abstraction.public_bucket(child);
      const uint64_t seen_key = (parent_bucket << 32) | child_bucket;
      if (seen.contains(seen_key)) {
        return;
      }
      seen.emplace(seen_key, true);
      CoarseChanceTransitionTemplate transition;
      transition.parent_board_state = parent;
      transition.cards.assign(cards.begin(), cards.end());
      transitions[parent_bucket].push_back(std::move(transition));
    });
  });
  return transitions;
}

const CoarseChanceTransitionMap& CoarseChanceTransitions(StreetKind street) {
  static const CoarseChanceTransitionMap preflop =
      BuildCoarseChanceTransitions(StreetKind::kPreflop);
  static const CoarseChanceTransitionMap flop =
      BuildCoarseChanceTransitions(StreetKind::kFlop);
  static const CoarseChanceTransitionMap turn =
      BuildCoarseChanceTransitions(StreetKind::kTurn);
  static const CoarseChanceTransitionMap river =
      BuildCoarseChanceTransitions(StreetKind::kRiver);
  switch (street) {
    case StreetKind::kPreflop:
      return preflop;
    case StreetKind::kFlop:
      return flop;
    case StreetKind::kTurn:
      return turn;
    case StreetKind::kRiver:
      return river;
  }
}

int ChanceSamples(const SolverConfig& config) {
  return std::max(1, config.chance_samples);
}

int WorkerCountForSamples(int samples) {
  int worker_count = static_cast<int>(std::thread::hardware_concurrency());
  if (worker_count <= 0) {
    worker_count = 1;
  }
  return std::min(worker_count, samples);
}

GameState DefaultInitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const int starting_stack = config.starting_stack_size;

  GameState initial_state;
  initial_state.stack[0] = std::max(0, starting_stack - small_blind);
  initial_state.stack[1] = std::max(0, starting_stack - big_blind);
  initial_state.pot = small_blind + big_blind;
  initial_state.folded_player = -1;
  initial_state.street = StreetKind::kPreflop;
  initial_state.all_in = false;
  initial_state.player_to_act = 0;
  initial_state.player_contribution = {small_blind, big_blind};
  return initial_state;
}

}  // namespace

CompactPublicState CompactPublicStateFromGameState(const GameState& state);

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>()) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const GameState& initial_state)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>(), initial_state) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     std::shared_ptr<TerminalUtilityCache> utility_cache)
    : CFRSolver(config, std::move(utility_cache), DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    GameState initial_state)
  : config_(config),
    initial_state_(CompactPublicStateFromGameState(initial_state)),
    game_tree_(std::make_shared<GameTree>(config)),
    rng_(12345),
    cumulative_root_utility_(0.0),
    utility_cache_(std::move(utility_cache)),
    mutable_tables_(std::make_shared<FrozenStrategyTables>()),
    frozen_tables_(mutable_tables_),
    cumulative_(std::make_shared<MutableCumulativeArrays>()) {
  // Pre-allocate strategy table storage when limits are known upfront.
  // This gives fully deterministic peak memory: no reallocation after init.
  if (config_.max_info_sets > 0) {
    // avg ~4 actions per info set (between 2 and kMaxActionsPerNode).
    constexpr int kAvgActionsPerInfoSet = 4;
    const size_t info_set_cap = static_cast<size_t>(config_.max_info_sets);
    const size_t action_cap = info_set_cap * kAvgActionsPerInfoSet;
    mutable_tables_->action_ids.reserve(action_cap);
    cumulative_->cumulative_regrets.reserve(action_cap);
    cumulative_->cumulative_strategies.reserve(action_cap);
  }
  if (config_.max_public_states > 0) {
    const size_t public_state_cap =
        static_cast<size_t>(config_.max_public_states);
    mutable_tables_->public_state_ids.reserve(public_state_cap);
    mutable_tables_->public_state_rows.reserve(public_state_cap);
    mutable_tables_->public_chance_child_ids.reserve(public_state_cap);
    mutable_tables_->chance_child_entries.reserve(public_state_cap);
    mutable_tables_->private_bucket_rows.reserve(public_state_cap);
    mutable_tables_->frozen_info_set_action_offsets.reserve(public_state_cap);
    mutable_tables_->public_info_set_slabs.reserve(public_state_cap);
    mutable_tables_->betting_history_ids.reserve(public_state_cap);
    mutable_tables_->betting_history_rows.reserve(public_state_cap);
  }
}

FrozenStrategyTables& CFRSolver::mutable_tables() {
  if (frozen_ || mutable_tables_ == nullptr) {
    throw std::logic_error("Strategy tables are frozen");
  }
  return *mutable_tables_;
}

CFRSolver::PrivateCards CFRSolver::PrivateCards::FromCombo(
    ComboId combo_id) {
  PrivateCards private_cards;
  private_cards.combo = combo_id;
  return private_cards;
}

CardMask CFRSolver::PrivateCards::mask() const {
  return ComboMask(combo);
}

CFRSolver::RangeSampler::RangeSampler(const TrainingRange& player_a_range,
                                       const TrainingRange& player_b_range)
    : player_a_range(player_a_range),
      player_b_range(player_b_range),
      compatible_player_b_weight(kComboCount, 0.0f),
      compatible_player_b_offsets(kComboCount, 0),
      compatible_player_b_counts(kComboCount, 0) {
  float total_weight = 0.0f;
  player_a_sample_weights.reserve(player_a_range.active_count);
  const size_t max_compatible_pairs =
      static_cast<size_t>(player_a_range.active_count) *
      static_cast<size_t>(player_b_range.active_count);
  compatible_player_b_combos.reserve(max_compatible_pairs);
  compatible_player_b_cumulative_weights.reserve(max_compatible_pairs);
  for (uint16_t a = 0; a < player_a_range.active_count; ++a) {
    const ComboId player_a_combo = player_a_range.active[a];
    const float player_a_weight = player_a_range.weights[player_a_combo];
    if (player_a_weight <= 0.0f) {
      player_a_sample_weights.push_back(0.0f);
      player_a_cumulative_weights.push_back(total_weight);
      continue;
    }
    const size_t offset = compatible_player_b_combos.size();
    compatible_player_b_offsets[player_a_combo] =
        static_cast<uint32_t>(offset);
    float cumulative_player_b_weight = 0.0f;
    for (uint16_t b = 0; b < player_b_range.active_count; ++b) {
      const ComboId player_b_combo = player_b_range.active[b];
      const float player_b_weight = player_b_range.weights[player_b_combo];
      if (player_b_weight <= 0.0f ||
          (ComboMask(player_a_combo) & ComboMask(player_b_combo)) != 0) {
        continue;
      }
      cumulative_player_b_weight += player_b_weight;
      compatible_player_b_combos.push_back(player_b_combo);
      compatible_player_b_cumulative_weights.push_back(
          cumulative_player_b_weight);
    }
    compatible_player_b_weight[player_a_combo] = cumulative_player_b_weight;
    compatible_player_b_counts[player_a_combo] =
        static_cast<uint16_t>(compatible_player_b_combos.size() - offset);
    const float sample_weight =
        player_a_weight * compatible_player_b_weight[player_a_combo];
    player_a_sample_weights.push_back(sample_weight);
    total_weight += sample_weight;
    player_a_cumulative_weights.push_back(total_weight);
  }

  if (total_weight <= 0.0f) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }
  total_player_a_weight = total_weight;
}

CFRSolver::RangeDeal CFRSolver::RangeSampler::sample(std::mt19937& rng) const {
  std::uniform_real_distribution<float> player_a_distribution(
      0.0f, total_player_a_weight);
  const float player_a_sample = player_a_distribution(rng);
  auto player_a_sampled = std::upper_bound(
      player_a_cumulative_weights.begin(), player_a_cumulative_weights.end(),
      player_a_sample);
  if (player_a_sampled == player_a_cumulative_weights.end()) {
    player_a_sampled = player_a_cumulative_weights.end() - 1;
  }
  const size_t player_a_active_index = static_cast<size_t>(
      player_a_sampled - player_a_cumulative_weights.begin());
  const ComboId player_a_combo = player_a_range.active[player_a_active_index];
  const float total_player_b_weight =
      compatible_player_b_weight[player_a_combo];
  const uint16_t compatible_count =
      compatible_player_b_counts[player_a_combo];
  if (total_player_b_weight <= 0.0f || compatible_count == 0) {
    throw std::logic_error("Range sampler selected an incompatible hand");
  }

  std::uniform_real_distribution<float> distribution(
      0.0f, total_player_b_weight);
  const float sample = distribution(rng);
  const size_t offset = compatible_player_b_offsets[player_a_combo];
  const auto first = compatible_player_b_cumulative_weights.begin() + offset;
  const auto last = first + compatible_count;
  auto sampled = std::upper_bound(first, last, sample);
  if (sampled == last) {
    sampled = last - 1;
  }

  const size_t sampled_index =
      offset + static_cast<size_t>(sampled - first);
  return RangeDeal(player_a_combo,
                   compatible_player_b_combos[sampled_index]);
}

CFRSolver::BettingHistoryKey CFRSolver::make_betting_history_key(
    const CompactPublicState& state) const {
  return betting_abstraction_.make_history_key(state);
}

CFRSolver::BettingHistoryRow CFRSolver::make_betting_history_row(
    const CompactPublicState& state) const {
  return betting_abstraction_.make_history_row(state);
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    uint32_t betting_history_id,
    const CompactPublicState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

CompactPublicState
CompactPublicStateFromGameState(const GameState& state) {
  CompactPublicState compact;
  compact.stack = {state.stack[0], state.stack[1]};
  compact.pot = state.pot;
  if (state.board_cards.size() > compact.board_cards.size()) {
    throw std::logic_error("GameState has too many board cards");
  }
  compact.board_count = static_cast<uint8_t>(state.board_cards.size());
  for (size_t i = 0; i < state.board_cards.size(); ++i) {
    compact.board_cards[i] = state.board_cards[i];
  }
  compact.board_mask = state.board_mask;
  if (state.history.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("GameState history exceeds compact row capacity");
  }
  if (state.history.size() > CompactPublicState::kMaxHistoryActions) {
    throw std::logic_error("GameState history exceeds compact inline capacity");
  }
  for (const GameAction& action : state.history) {
    AppendHistoryAction(compact, action);
  }
  compact.street = state.street;
  compact.all_in = state.all_in;
  compact.folded_player = state.folded_player;
  compact.player_to_act = state.player_to_act;
  compact.player_contribution = state.player_contribution;
  compact.player_contribution_count = state.player_contribution_count;
  return compact;
}

CFRSolver::PublicStateRow CFRSolver::make_public_state_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
  state = betting_abstraction_.public_state_for_row(std::move(state));
  PublicStateRow row;
  row.betting_history_id = betting_history_id;
  row.public_bucket = card_abstraction_.public_bucket(state);
  row.is_terminal = game_tree_->is_terminal(state);
  row.player_to_act = game_tree_->get_player_to_act(state);
  row.state = std::move(state);
  row.is_chance_node = !row.is_terminal && row.player_to_act == -1;

  if (!row.is_terminal && !row.is_chance_node && IsPlayer(row.player_to_act)) {
    row.action_count = game_tree_->get_legal_actions(row.state, row.actions);
    for (int i = 0; i < row.action_count; ++i) {
      row.action_ids[static_cast<size_t>(i)] =
          GameTree::action_key(row.actions[static_cast<size_t>(i)]);
    }
  }

  return row;
}

CFRSolver::InfoSetRow CFRSolver::append_info_set_actions(
    absl::Span<const int> action_ids) {
  FrozenStrategyTables& tables = mutable_tables();
  // Keep each real infoset action block on a cache-line boundary to reduce
  // false sharing between worker CAS updates.
  const size_t padding =
      (kCumulativeActionBlockAlignment -
       tables.action_ids.size() % kCumulativeActionBlockAlignment) %
      kCumulativeActionBlockAlignment;
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(tables.action_ids.size() + padding);
  row.action_count = static_cast<uint16_t>(action_ids.size());
  const size_t required_action_capacity =
      tables.action_ids.size() + padding + action_ids.size();
  if (required_action_capacity > tables.action_ids.capacity()) {
    const size_t new_capacity =
        std::max(required_action_capacity,
                 tables.action_ids.empty() ? size_t{4096}
                                           : tables.action_ids.capacity() * 2);
    tables.action_ids.reserve(new_capacity);
    cumulative_->cumulative_regrets.reserve(new_capacity);
    cumulative_->cumulative_strategies.reserve(new_capacity);
  }
  for (size_t i = 0; i < padding; ++i) {
    tables.action_ids.push_back(0);
    cumulative_->cumulative_regrets.push_back(0.0f);
    cumulative_->cumulative_strategies.push_back(0.0f);
  }
  for (int action_id : action_ids) {
    tables.action_ids.push_back(action_id);
    cumulative_->cumulative_regrets.push_back(0.0f);
    cumulative_->cumulative_strategies.push_back(0.0f);
    record_action_entry_touches();
  }
  return row;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
  if (frozen_) {
    auto existing = frozen_tables_->public_state_ids.find(
        make_public_state_key(betting_history_id, state));
    if (existing == frozen_tables_->public_state_ids.end()) {
      return std::nullopt;
    }
    return existing->second;
  }

  FrozenStrategyTables& tables = mutable_tables();
  if (config_.max_public_states > 0 &&
      static_cast<int>(tables.public_state_rows.size()) >=
          config_.max_public_states) {
    auto existing = tables.public_state_ids.find(
        make_public_state_key(betting_history_id, state));
    if (existing != tables.public_state_ids.end()) {
      return existing->second;
    }
    return std::nullopt;
  }

  PublicStateKey key = make_public_state_key(betting_history_id, state);
  const auto [state_iter, inserted] = tables.public_state_ids.try_emplace(
      std::move(key), static_cast<uint32_t>(tables.public_state_ids.size()));
  if (!inserted) {
    return state_iter->second;
  }
  const uint32_t public_state_id = state_iter->second;
  tables.public_state_rows.push_back(
      make_public_state_row(betting_history_id, std::move(state)));
  cache_betting_history_actions(betting_history_id,
                                tables.public_state_rows.back());
  return public_state_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    const CompactPublicState& state) {
  if (frozen_) {
    BettingHistoryKey key = make_betting_history_key(state);
    const auto betting_history =
        frozen_tables_->betting_history_ids.find(key);
    if (betting_history == frozen_tables_->betting_history_ids.end()) {
      return std::nullopt;
    }
    return get_or_create_public_state_row(betting_history->second, state);
  }

  const uint32_t betting_history_id = get_or_create_betting_history_id(state);
  return get_or_create_public_state_row(betting_history_id, state);
}

uint32_t CFRSolver::get_or_create_betting_history_id(
    const CompactPublicState& state) {
  BettingHistoryKey key = make_betting_history_key(state);
  BettingHistoryRow row = make_betting_history_row(state);
  return get_or_create_betting_history_id(std::move(key), std::move(row));
}

uint32_t CFRSolver::get_or_create_betting_history_id(BettingHistoryKey key,
                                                     BettingHistoryRow row) {
  betting_abstraction_.copy_history_to_row(key, row);
  FrozenStrategyTables& tables = mutable_tables();
  const auto [history_iter, inserted] = tables.betting_history_ids.try_emplace(
      std::move(key), static_cast<uint32_t>(tables.betting_history_ids.size()));
  const uint32_t betting_history_id = history_iter->second;
  if (!inserted) {
    if (tables.betting_history_rows.size() <= betting_history_id) {
      tables.betting_history_rows.resize(
          static_cast<size_t>(betting_history_id) + 1);
    }
    return betting_history_id;
  }
  tables.betting_history_rows.push_back(std::move(row));
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_action_child_betting_history_id(
    uint32_t parent_betting_history_id,
    int action_index,
    const CompactPublicState& child_state) {
  FrozenStrategyTables& tables = mutable_tables();
  if (parent_betting_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_betting_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      const uint32_t child_id =
          parent_row.action_child_ids[static_cast<size_t>(action_index)];
      if (child_id != GameTree::kInvalidBettingHistoryId) {
        return child_id;
      }
    }
  }

  BettingHistoryKey child_key = make_betting_history_key(child_state);
  BettingHistoryRow child_row = make_betting_history_row(child_state);
  if (parent_betting_history_id < tables.betting_history_rows.size()) {
    const BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_betting_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      betting_abstraction_.replace_with_action_index_history(
          parent_row, action_index, child_state, child_key);
    }
  }
  const uint32_t child_id = get_or_create_betting_history_id(
      std::move(child_key), std::move(child_row));
  if (parent_betting_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_betting_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      parent_row.action_child_ids[static_cast<size_t>(action_index)] =
          child_id;
    }
  }
  return child_id;
}

uint32_t CFRSolver::get_or_create_chance_child_betting_history_id(
    uint32_t parent_betting_history_id,
    const CompactPublicState& child_state) {
  FrozenStrategyTables& tables = mutable_tables();
  if (parent_betting_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_betting_history_id];
    if (parent_row.chance_child_id !=
        GameTree::kInvalidBettingHistoryId) {
      return parent_row.chance_child_id;
    }
  }

  const uint32_t child_id = get_or_create_betting_history_id(child_state);
  if (parent_betting_history_id < tables.betting_history_rows.size()) {
    tables.betting_history_rows[parent_betting_history_id].chance_child_id =
        child_id;
  }
  return child_id;
}

void CFRSolver::cache_betting_history_actions(
    uint32_t betting_history_id,
    const PublicStateRow& row) {
  FrozenStrategyTables& tables = mutable_tables();
  if (row.action_count == 0 ||
      betting_history_id >= tables.betting_history_rows.size()) {
    return;
  }

  BettingHistoryRow& betting_history =
      tables.betting_history_rows[static_cast<size_t>(betting_history_id)];
  betting_history.action_count = row.action_count;
  for (int i = 0; i < row.action_count; ++i) {
    betting_history.action_ids[static_cast<size_t>(i)] =
        row.action_ids[static_cast<size_t>(i)];
  }
}

std::optional<uint32_t> CFRSolver::action_child_public_state(
    uint32_t public_state_id,
    int action_index) const {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = rows[public_state_id];
  if (action_index < 0 || action_index >= row.action_count) {
    throw std::logic_error("action child index out of range");
  }
  const uint32_t child_id =
      row.action_child_ids[static_cast<size_t>(action_index)];
  if (child_id == GameTree::kInvalidPublicStateId ||
      child_id == kCappedPublicStateId) {
    return std::nullopt;
  }
  return child_id;
}

std::optional<uint32_t> CFRSolver::chance_child_public_state(
    uint32_t public_state_id,
    const CompactPublicState& child_state) const {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = rows[public_state_id];
  const size_t begin = row.chance_child_offset;
  const size_t end = begin + row.chance_child_count;
  const auto& entries = frozen_tables_->chance_child_entries;
  if (begin > entries.size() || end > entries.size()) {
    return std::nullopt;
  }
  const PublicBucketId outcome_id = chance_outcome_id(child_state);
  size_t low = begin;
  size_t high = end;
  while (low < high) {
    const size_t mid = low + (high - low) / 2;
    if (entries[mid].outcome_id < outcome_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low == end || entries[low].outcome_id != outcome_id) {
    return std::nullopt;
  }
  return entries[low].public_state_id;
}

std::optional<uint32_t> CFRSolver::chance_child_public_state(
    uint32_t public_state_id,
    absl::Span<const CardId> cards) const {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  return chance_child_public_state(
      public_state_id, game_tree_->apply_chance(rows[public_state_id].state,
                                                cards));
}

bool CFRSolver::for_each_required_chance_transition(
    const PublicStateRow& row,
    const std::function<bool(const CompactPublicState&,
                             absl::Span<const CardId>)>& callback) const {
  if constexpr (kCoarsePublicBuckets) {
    const auto& transitions = CoarseChanceTransitions(row.state.street);
    const auto existing = transitions.find(row.public_bucket);
    if (existing == transitions.end()) {
      return false;
    }
    for (const CoarseChanceTransitionTemplate& transition : existing->second) {
      CompactPublicState parent_state =
          StateWithBoardFrom(row.state, transition.parent_board_state);
      const CompactPublicState child_state =
          game_tree_->apply_chance(parent_state, transition.cards);
      if (!callback(child_state, absl::Span<const CardId>(transition.cards))) {
        return false;
      }
    }
    return true;
  } else {
    return ForEachNextStreetDeal(row.state, [&](absl::Span<const CardId> cards) {
      return callback(game_tree_->apply_chance(row.state, cards), cards);
    });
  }
}

PublicBucketId CFRSolver::chance_outcome_id(
    const CompactPublicState& child_state) const {
  return card_abstraction_.public_bucket(child_state);
}

std::optional<uint32_t> CFRSolver::get_or_create_action_child_public_state(
    uint32_t public_state_id,
    int action_index) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& read_row = rows[public_state_id];
  if (action_index < 0 || action_index >= read_row.action_count) {
    throw std::logic_error(
        "get_or_create_action_child_public_state: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const uint32_t existing_child_id = read_row.action_child_ids[action_slot];
  if (existing_child_id != GameTree::kInvalidPublicStateId) {
    record_betting_history_transition_hit();
    if (existing_child_id == kCappedPublicStateId) {
      return std::nullopt;
    }
    return existing_child_id;
  }

  const uint32_t parent_betting_history_id = read_row.betting_history_id;
  const auto& betting_history_rows = frozen_tables_->betting_history_rows;
  if (parent_betting_history_id < betting_history_rows.size()) {
    const BettingHistoryRow& parent_betting_history =
        betting_history_rows[parent_betting_history_id];
    if (action_index >= 0 &&
        action_index < parent_betting_history.action_count) {
      const uint32_t child_betting_history_id =
          parent_betting_history.action_child_ids[action_slot];
      if (child_betting_history_id !=
          GameTree::kInvalidBettingHistoryId) {
        const PublicStateKey child_key{
            child_betting_history_id,
            read_row.public_bucket,
        };
        auto existing_public_child =
            frozen_tables_->public_state_ids.find(child_key);
        if (existing_public_child != frozen_tables_->public_state_ids.end()) {
          if (!frozen_) {
            mutable_tables()
                .public_state_rows[public_state_id]
                .action_child_ids[action_slot] = existing_public_child->second;
          }
          record_betting_history_transition_hit();
          return existing_public_child->second;
        }
      }
    }
  }
  if (frozen_) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(frozen_tables_->public_state_rows.size()) >=
          config_.max_public_states) {
    mutable_tables()
        .public_state_rows[public_state_id]
        .action_child_ids[action_slot] = kCappedPublicStateId;
    record_betting_history_transition_miss();
    return std::nullopt;
  }

  const GameAction action =
      frozen_tables_->public_state_rows[public_state_id].actions[action_slot];
  CompactPublicState child_state = game_tree_->apply_action(
      frozen_tables_->public_state_rows[public_state_id].state, action);
  const uint32_t child_betting_history_id =
      get_or_create_action_child_betting_history_id(
          parent_betting_history_id, action_index, child_state);
  std::optional<uint32_t> child_id =
      get_or_create_public_state_row(child_betting_history_id,
                                     std::move(child_state));
  if (!child_id.has_value()) {
    mutable_tables()
        .public_state_rows[public_state_id]
        .action_child_ids[action_slot] = kCappedPublicStateId;
    record_betting_history_transition_miss();
    return std::nullopt;
  }

  mutable_tables()
      .public_state_rows[public_state_id]
      .action_child_ids[action_slot] = *child_id;
  record_betting_history_transition_miss();
  record_child_node_created();
  return child_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_chance_child_public_state(
    uint32_t public_state_id,
    const CompactPublicState& child_state) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = rows[public_state_id];
  const PublicBucketId outcome_id = chance_outcome_id(child_state);
  const ChanceTransitionKey transition_key{public_state_id, outcome_id};
  const auto& chance_children = frozen_tables_->public_chance_child_ids;
  auto existing = chance_children.find(transition_key);
  if (existing != chance_children.end()) {
    record_betting_history_transition_hit();
    return existing->second;
  }
  if constexpr (kCoarsePublicBuckets) {
    const uint32_t cached_parent_betting_history_id = row.betting_history_id;
    const auto& betting_history_rows = frozen_tables_->betting_history_rows;
    if (cached_parent_betting_history_id < betting_history_rows.size()) {
      const uint32_t child_betting_history_id =
          betting_history_rows[cached_parent_betting_history_id].chance_child_id;
      if (child_betting_history_id != GameTree::kInvalidBettingHistoryId) {
        const PublicStateKey child_public_key{
            child_betting_history_id,
            outcome_id,
        };
        auto existing_public_child =
            frozen_tables_->public_state_ids.find(child_public_key);
        if (existing_public_child != frozen_tables_->public_state_ids.end()) {
          if (!frozen_) {
            mutable_tables().public_chance_child_ids.emplace(
                transition_key, existing_public_child->second);
          }
          record_betting_history_transition_hit();
          return existing_public_child->second;
        }
      }
    }
  }
  if (frozen_) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(frozen_tables_->public_state_rows.size()) >=
          config_.max_public_states) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }

  const uint32_t parent_betting_history_id =
      row.betting_history_id;
  CompactPublicState stored_child_state = child_state;
  const uint32_t child_betting_history_id =
      get_or_create_chance_child_betting_history_id(
          parent_betting_history_id, stored_child_state);
  std::optional<uint32_t> child_id =
      get_or_create_public_state_row(child_betting_history_id,
                                     std::move(stored_child_state));
  if (!child_id.has_value()) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }

  mutable_tables().public_chance_child_ids.emplace(transition_key, *child_id);
  record_betting_history_transition_miss();
  record_child_node_created();
  return child_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_chance_child_public_state(
    uint32_t public_state_id,
    absl::Span<const CardId> cards) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  return get_or_create_chance_child_public_state(
      public_state_id, game_tree_->apply_chance(rows[public_state_id].state,
                                                cards));
}

bool CFRSolver::prebuild_public_state_rows(uint32_t root_public_state_id,
                                           int max_depth) {
  if (frozen_) {
    return true;
  }
  if (root_public_state_id >= frozen_tables_->public_state_rows.size()) {
    return false;
  }

  struct QueueEntry {
    uint32_t public_state_id = 0;
    int depth = 0;
  };

  std::vector<QueueEntry> queue;
  std::vector<char> queued;
  queue.reserve(1024);
  queued.resize(frozen_tables_->public_state_rows.size(), 0);
  queue.push_back({root_public_state_id, 0});
  queued[root_public_state_id] = 1;

  auto enqueue = [&](uint32_t public_state_id, int depth) {
    if (public_state_id >= queued.size()) {
      queued.resize(static_cast<size_t>(public_state_id) + 1, 0);
    }
    if (!queued[public_state_id]) {
      queued[public_state_id] = 1;
      queue.push_back({public_state_id, depth});
    }
  };

  for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const QueueEntry entry = queue[cursor];
    if (entry.public_state_id >= frozen_tables_->public_state_rows.size()) {
      return false;
    }
    const PublicStateRow row =
        frozen_tables_->public_state_rows[entry.public_state_id];
    if (row.is_terminal) {
      continue;
    }

    if (row.is_chance_node) {
      const bool complete = for_each_required_chance_transition(
          row, [&](const CompactPublicState& child_state,
                   absl::Span<const CardId> cards) {
            std::optional<uint32_t> child_public_state_id =
                get_or_create_chance_child_public_state(entry.public_state_id,
                                                        child_state);
            if (!child_public_state_id.has_value()) {
              return false;
            }
            enqueue(*child_public_state_id, entry.depth);
            return true;
          });
      if (!complete) {
        return false;
      }
      continue;
    }

    if (max_depth > 0 && entry.depth >= max_depth) {
      continue;
    }

    for (int action_index = 0; action_index < row.action_count;
         ++action_index) {
      std::optional<uint32_t> child_public_state_id =
          get_or_create_action_child_public_state(entry.public_state_id,
                                                  action_index);
      if (!child_public_state_id.has_value()) {
        return false;
      }
      enqueue(*child_public_state_id, entry.depth + 1);
    }
  }

  rebuild_chance_child_entries();
  return true;
}

void CFRSolver::rebuild_chance_child_entries() {
  FrozenStrategyTables& tables = mutable_tables();
  struct PendingChanceChild {
    uint32_t parent_id = 0;
    PublicBucketId outcome_id = 0;
    uint32_t public_state_id = GameTree::kInvalidPublicStateId;
  };

  std::vector<PendingChanceChild> pending;
  pending.reserve(tables.public_chance_child_ids.size());
  for (const auto& [transition_key, child_id] :
       tables.public_chance_child_ids) {
    const uint32_t parent_id = transition_key.parent_public_state_id;
    if (parent_id >= tables.public_state_rows.size()) {
      continue;
    }
    pending.push_back({
        parent_id,
        transition_key.outcome_id,
        child_id,
    });
  }

  std::sort(pending.begin(), pending.end(),
            [](const PendingChanceChild& left,
               const PendingChanceChild& right) {
              if (left.parent_id != right.parent_id) {
                return left.parent_id < right.parent_id;
              }
              return left.outcome_id < right.outcome_id;
            });

  for (PublicStateRow& row : tables.public_state_rows) {
    row.chance_child_offset = 0;
    row.chance_child_count = 0;
  }
  tables.chance_child_entries.clear();
  tables.chance_child_entries.reserve(pending.size());
  uint32_t current_parent_id = GameTree::kInvalidPublicStateId;
  for (const PendingChanceChild& child : pending) {
    if (child.public_state_id >= tables.public_state_rows.size()) {
      continue;
    }
    if (child.parent_id != current_parent_id) {
      current_parent_id = child.parent_id;
      tables.public_state_rows[current_parent_id].chance_child_offset =
          static_cast<uint32_t>(tables.chance_child_entries.size());
    }
    tables.chance_child_entries.push_back({
        child.outcome_id,
        child.public_state_id,
    });
    ++tables.public_state_rows[current_parent_id].chance_child_count;
  }
}

bool CFRSolver::validate_prebuilt_transitions(
    uint32_t root_public_state_id,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& rows = frozen_tables_->public_state_rows;
  const auto& history_rows = frozen_tables_->betting_history_rows;
  if (root_public_state_id >= rows.size()) {
    return false;
  }
  stats.betting_history_transition_prebuild_complete = true;
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  struct QueueEntry {
    uint32_t public_state_id = 0;
    int depth = 0;
  };

  std::vector<QueueEntry> queue;
  std::vector<char> queued(rows.size(), 0);
  queue.reserve(1024);
  queue.push_back({root_public_state_id, 0});
  queued[root_public_state_id] = 1;

  auto enqueue = [&](uint32_t public_state_id, int depth) {
    if (public_state_id >= rows.size()) {
      return false;
    }
    if (!queued[public_state_id]) {
      queued[public_state_id] = 1;
      queue.push_back({public_state_id, depth});
    }
    return true;
  };

  auto valid_public_child = [&](uint32_t public_state_id) {
    return public_state_id != GameTree::kInvalidPublicStateId &&
           public_state_id != kCappedPublicStateId &&
           public_state_id < rows.size();
  };
  auto mark_missing_betting_history = [&] {
    ++stats.missing_betting_history_transitions;
    stats.betting_history_transition_prebuild_complete = false;
  };
  auto mark_missing_action = [&] {
    ++stats.missing_action_transitions;
    stats.action_transition_prebuild_complete = false;
  };
  auto mark_missing_chance = [&] {
    ++stats.missing_chance_transitions;
    stats.chance_transition_prebuild_complete = false;
  };

  for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const QueueEntry entry = queue[cursor];
    if (entry.public_state_id >= rows.size()) {
      stats.betting_history_transition_prebuild_complete = false;
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const PublicStateRow& row = rows[entry.public_state_id];
    if (row.is_terminal) {
      continue;
    }

    if (row.betting_history_id >= history_rows.size()) {
      mark_missing_betting_history();
      continue;
    }
    const BettingHistoryRow& history_row =
        history_rows[row.betting_history_id];

    if (row.is_chance_node) {
      const bool chance_complete = for_each_required_chance_transition(
          row, [&](const CompactPublicState& child_state,
                   absl::Span<const CardId> cards) {
        ++stats.prebuild_chance_transitions;
        ++stats.prebuild_betting_history_transitions;
        const std::optional<uint32_t> child_public_state_id =
            chance_child_public_state(entry.public_state_id, child_state);
        const bool valid_child = child_public_state_id.has_value() &&
                                 valid_public_child(*child_public_state_id);
        if (!valid_child) {
          mark_missing_chance();
        }
        const bool valid_betting_child =
            valid_child &&
            history_row.chance_child_id !=
                GameTree::kInvalidBettingHistoryId &&
            history_row.chance_child_id < history_rows.size();
        if (!valid_betting_child) {
          mark_missing_betting_history();
        } else if (rows[*child_public_state_id].betting_history_id !=
                   history_row.chance_child_id) {
          mark_missing_betting_history();
        }
        if (valid_child && !enqueue(*child_public_state_id, entry.depth)) {
          mark_missing_chance();
          mark_missing_betting_history();
        }
        return true;
      });
      if (!chance_complete) {
        mark_missing_chance();
        mark_missing_betting_history();
      }
      continue;
    }

    if (max_depth > 0 && entry.depth >= max_depth) {
      continue;
    }

    if (history_row.action_count != row.action_count) {
      mark_missing_betting_history();
    }
    for (int action_index = 0; action_index < row.action_count;
         ++action_index) {
      ++stats.prebuild_action_transitions;
      ++stats.prebuild_betting_history_transitions;
      const size_t action_slot = static_cast<size_t>(action_index);
      const uint32_t child_betting_history_id =
          history_row.action_child_ids[action_slot];
      const uint32_t child_public_state_id = row.action_child_ids[action_slot];
      const bool valid_action_child = valid_public_child(child_public_state_id);
      if (!valid_action_child) {
        mark_missing_action();
      }
      const bool valid_betting_child =
          valid_action_child &&
          history_row.action_ids[action_slot] == row.action_ids[action_slot] &&
          child_betting_history_id !=
              GameTree::kInvalidBettingHistoryId &&
          child_betting_history_id < history_rows.size();
      if (!valid_betting_child) {
        mark_missing_betting_history();
      } else if (rows[child_public_state_id].betting_history_id !=
                 child_betting_history_id) {
        mark_missing_betting_history();
      }
      if (valid_action_child &&
          !enqueue(child_public_state_id, entry.depth + 1)) {
        mark_missing_action();
        mark_missing_betting_history();
      }
    }
  }

  return stats.betting_history_transition_prebuild_complete &&
         stats.action_transition_prebuild_complete &&
         stats.chance_transition_prebuild_complete;
}

bool CFRSolver::prebuild_info_set_rows(
    const TrainingRangeView& player_a_range,
    const TrainingRangeView& player_b_range) {
  if (frozen_) {
    return true;
  }

  std::vector<uint32_t> seen_buckets;
  uint32_t seen_generation = 1;

  for (uint32_t public_state_id = 0;
       public_state_id < frozen_tables_->public_state_rows.size();
       ++public_state_id) {
    const PublicStateRow& row =
        frozen_tables_->public_state_rows[public_state_id];
    const int player = row.player_to_act;
    if (row.is_terminal || row.is_chance_node || row.action_count == 0 ||
        !IsPlayer(player)) {
      continue;
    }

    const TrainingRangeView& range =
        player == 0 ? player_a_range : player_b_range;
    if (range.empty()) {
      continue;
    }

    const uint32_t bucket_count =
        card_abstraction_.private_bucket_count(row.state);
    if (bucket_count == 0 || bucket_count > kComboCount) {
      return false;
    }
    if (seen_buckets.size() < bucket_count) {
      seen_buckets.resize(bucket_count, 0);
    }
    if (seen_generation == 0) {
      std::fill(seen_buckets.begin(), seen_buckets.end(), 0);
      seen_generation = 1;
    }
    const uint32_t generation = seen_generation++;
    const absl::Span<const int> action_ids(row.action_ids.data(),
                                           row.action_count);
    const CardMask board_mask = row.state.board_mask;
    for (size_t i = 0; i < range.size(); ++i) {
      if (range.weight(i) <= 0.0f) {
        continue;
      }
      const ComboId combo_id = range.combo(i);
      if ((ComboMask(combo_id) & board_mask) != 0) {
        continue;
      }
      const PrivateBucketId private_bucket =
          card_abstraction_.private_bucket(combo_id, row.state);
      if (private_bucket >= bucket_count) {
        return false;
      }
      if (seen_buckets[private_bucket] == generation) {
        continue;
      }
      seen_buckets[private_bucket] = generation;
      if (get_or_create_info_set_row(
              {public_state_id, player, private_bucket}, action_ids) ==
          nullptr) {
        return false;
      }
    }
  }

  return true;
}

bool CFRSolver::prebuild_private_bucket_rows() {
  if (frozen_) {
    return true;
  }

  FrozenStrategyTables& tables = mutable_tables();
  tables.private_bucket_rows.resize(tables.public_state_rows.size());
  for (size_t public_state_id = 0;
       public_state_id < tables.public_state_rows.size(); ++public_state_id) {
    const PublicStateRow& row = tables.public_state_rows[public_state_id];
    const uint32_t bucket_count =
        card_abstraction_.private_bucket_count(row.state);
    if (bucket_count == 0 || bucket_count > kComboCount) {
      return false;
    }
    auto& bucket_row = tables.private_bucket_rows[public_state_id];
    for (int combo = 0; combo < kComboCount; ++combo) {
      const PrivateBucketId private_bucket =
          card_abstraction_.private_bucket(static_cast<ComboId>(combo),
                                           row.state);
      if (private_bucket >= bucket_count) {
        return false;
      }
      bucket_row[static_cast<size_t>(combo)] = private_bucket;
    }
  }
  return true;
}

bool CFRSolver::prebuild_frozen_info_set_action_offsets() {
  if (frozen_) {
    return true;
  }

  FrozenStrategyTables& tables = mutable_tables();
  tables.frozen_info_set_action_offsets.resize(tables.public_state_rows.size());
  for (auto& offset_row : tables.frozen_info_set_action_offsets) {
    for (auto& player_offsets : offset_row) {
      player_offsets.fill(FrozenStrategyTables::kInvalidActionOffset);
    }
  }

  for (size_t public_state_id = 0;
       public_state_id < tables.public_state_rows.size(); ++public_state_id) {
    const PublicStateRow& public_row = tables.public_state_rows[public_state_id];
    const PublicInfoSetSlab* slab =
        public_info_set_slab(static_cast<uint32_t>(public_state_id));
    if (slab == nullptr) {
      continue;
    }

    auto& offset_row =
        tables.frozen_info_set_action_offsets[public_state_id];
    for (int player = 0; player < kPlayerCount; ++player) {
      const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
      auto& player_offsets = offset_row[player];
      for (int bucket = 0; bucket < kComboCount; ++bucket) {
        const InfoSetRow* info_set_row = find_info_set_row(
            player_slab, static_cast<PrivateBucketId>(bucket));
        if (info_set_row == nullptr) {
          continue;
        }
        if (info_set_row->action_count != public_row.action_count) {
          return false;
        }
        player_offsets[static_cast<size_t>(bucket)] =
            info_set_row->action_offset;
      }
    }
  }

  return true;
}

FrozenStrategyTables::PrivateBucketId
CFRSolver::private_bucket_for_frozen_row(uint32_t public_state_id,
                                         ComboId combo_id) const {
  return frozen_tables_->private_bucket_rows[public_state_id][combo_id];
}

uint32_t CFRSolver::frozen_info_set_action_offset(
    uint32_t public_state_id,
    int player,
    PrivateBucketId private_bucket) const {
  return frozen_tables_->frozen_info_set_action_offsets[public_state_id][player]
                                                        [private_bucket];
}

CFRSolver::PublicInfoSetSlab& CFRSolver::get_or_create_public_info_set_slab(
    uint32_t public_state_id) {
  FrozenStrategyTables& tables = mutable_tables();
  if (tables.public_info_set_slabs.size() <= public_state_id) {
    tables.public_info_set_slabs.resize(static_cast<size_t>(public_state_id) + 1);
  }
  std::unique_ptr<PublicInfoSetSlab>& slab =
      tables.public_info_set_slabs[public_state_id];
  if (slab == nullptr) {
    slab = std::make_unique<PublicInfoSetSlab>();
  }
  return *slab;
}

const CFRSolver::PublicInfoSetSlab* CFRSolver::public_info_set_slab(
    uint32_t public_state_id) const {
  const auto& slabs = frozen_tables_->public_info_set_slabs;
  if (public_state_id >= slabs.size()) {
    return nullptr;
  }
  return slabs[public_state_id].get();
}

const CFRSolver::InfoSetRow* CFRSolver::find_info_set_row(
    InfoSetAddress address) const {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return nullptr;
  }
  const PublicInfoSetSlab* slab = public_info_set_slab(address.public_state_id);
  if (slab == nullptr) {
    return nullptr;
  }
  const PublicInfoSetSlabPlayer& player_slab = slab->players[address.player];
  return find_info_set_row(player_slab, address.private_bucket);
}

const CFRSolver::InfoSetRow* CFRSolver::find_info_set_row(
    const PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  if (private_bucket >= kComboCount) {
    return nullptr;
  }
  const size_t chunk_index = private_bucket / kPrivateBucketChunkSize;
  const size_t chunk_offset = private_bucket % kPrivateBucketChunkSize;
  const std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    return nullptr;
  }
  const int32_t row_id = chunk->rows[chunk_offset];
  if (row_id < 0 ||
      static_cast<size_t>(row_id) >= player_slab.rows.size()) {
    return nullptr;
  }
  return &player_slab.rows[row_id];
}

int32_t& CFRSolver::get_or_create_private_row_slot(
    PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  const size_t chunk_index = private_bucket / kPrivateBucketChunkSize;
  const size_t chunk_offset = private_bucket % kPrivateBucketChunkSize;
  std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    chunk = std::make_unique<PrivateRowChunk>();
  }
  return chunk->rows[chunk_offset];
}

const CFRSolver::InfoSetRow* CFRSolver::get_or_create_info_set_row(
    InfoSetAddress address,
    absl::Span<const int> action_ids) {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return nullptr;
  }

  if (const InfoSetRow* row = find_info_set_row(address)) {
    return row;
  }

  if (frozen_ || (config_.max_info_sets > 0 &&
                  static_cast<int>(frozen_tables_->info_set_count) >= config_.max_info_sets)) {
    return nullptr;
  }

  InfoSetRow row = append_info_set_actions(action_ids);
  PublicInfoSetSlab& slab =
      get_or_create_public_info_set_slab(address.public_state_id);
  PublicInfoSetSlabPlayer& player_slab = slab.players[address.player];
  int32_t& row_id =
      get_or_create_private_row_slot(player_slab, address.private_bucket);
  row_id = static_cast<int32_t>(player_slab.rows.size());
  player_slab.rows.push_back(row);
  ++mutable_tables().info_set_count;
  return &player_slab.rows.back();
}

void CFRSolver::add_traversal_stats(const TraversalStats& stats) {
  traversal_stats_.cfr_updates += stats.cfr_updates;
  traversal_stats_.preflop_updates += stats.preflop_updates;
  traversal_stats_.flop_updates += stats.flop_updates;
  traversal_stats_.turn_updates += stats.turn_updates;
  traversal_stats_.river_updates += stats.river_updates;
  traversal_stats_.child_nodes_created += stats.child_nodes_created;
  traversal_stats_.chance_samples += stats.chance_samples;
  traversal_stats_.terminal_utility_calls += stats.terminal_utility_calls;
  traversal_stats_.fold_utility_calls += stats.fold_utility_calls;
  traversal_stats_.showdown_utility_calls += stats.showdown_utility_calls;
  traversal_stats_.action_entry_touches += stats.action_entry_touches;
  traversal_stats_.atomic_regret_update_retries +=
      stats.atomic_regret_update_retries;
  traversal_stats_.betting_history_transition_hits +=
      stats.betting_history_transition_hits;
  traversal_stats_.betting_history_transition_misses +=
      stats.betting_history_transition_misses;
}

inline void CFRSolver::record_action_entry_touches(int64_t count) {
  if constexpr (kTraversalStatsEnabled) {
    traversal_stats_.action_entry_touches += count;
  }
}

inline void CFRSolver::record_cfr_update(StreetKind street, int depth) {
  if constexpr (kTraversalStatsEnabled) {
    ++traversal_stats_.cfr_updates;
    traversal_stats_.max_decision_depth =
        std::max(traversal_stats_.max_decision_depth, depth);
    switch (street) {
      case StreetKind::kPreflop:
        ++traversal_stats_.preflop_updates;
        break;
      case StreetKind::kFlop:
        ++traversal_stats_.flop_updates;
        break;
      case StreetKind::kTurn:
        ++traversal_stats_.turn_updates;
        break;
      case StreetKind::kRiver:
        ++traversal_stats_.river_updates;
        break;
    }
  }
}

inline void CFRSolver::record_chance_samples(int64_t count) {
  if constexpr (kTraversalStatsEnabled) {
    traversal_stats_.chance_samples += count;
  }
}

inline void CFRSolver::record_terminal_utility(bool showdown) {
  if constexpr (kTraversalStatsEnabled) {
    ++traversal_stats_.terminal_utility_calls;
    if (showdown) {
      ++traversal_stats_.showdown_utility_calls;
    } else {
      ++traversal_stats_.fold_utility_calls;
    }
  }
}

inline void CFRSolver::record_child_node_created() {
  if constexpr (kTraversalStatsEnabled) {
    ++traversal_stats_.child_nodes_created;
  }
}

inline void CFRSolver::record_betting_history_transition_hit() {
  if constexpr (kTraversalStatsEnabled) {
    ++traversal_stats_.betting_history_transition_hits;
  }
}

inline void CFRSolver::record_betting_history_transition_miss() {
  if constexpr (kTraversalStatsEnabled) {
    ++traversal_stats_.betting_history_transition_misses;
  }
}

inline void CFRSolver::record_atomic_regret_update_retries(int64_t count) {
  if constexpr (kTraversalStatsEnabled) {
    traversal_stats_.atomic_regret_update_retries += count;
  }
}

void CFRSolver::run(int iterations, const HandRange& player_a_range,
                    const HandRange& player_b_range) {
  run_iterations(iterations, player_a_range, player_b_range);
}

void CFRSolver::run_iterations(int iterations,
                               const HandRange& player_a_range,
                               const HandRange& player_b_range) {
  last_training_run_stats_ = {};
  if (iterations <= 0) {
    return;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  TrainingRangeView player_a_hands_view;
  TrainingRangeView player_b_hands_view;
  player_a_hands_view.reset_to_all(player_a_training_range);
  player_b_hands_view.reset_to_all(player_b_training_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);

  VLOG(1) << "Preparing compact public-state rows...";
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return;
  }
  VLOG(1) << "Compact root row has "
          << static_cast<int>(
                 frozen_tables_->public_state_rows[*root_public_state_id].action_count)
          << " legal actions";

  const int num_threads =
      config_.num_training_threads <= 1 ? 1 : config_.num_training_threads;
  const int max_depth = config_.max_depth;
  const bool can_use_frozen_regret_only =
      config_.regret_only_training && max_depth == 0;

  const bool should_run_frozen_phase = prepare_frozen_training(
      *root_public_state_id, num_threads, max_depth, can_use_frozen_regret_only,
      player_a_hands_view, player_b_hands_view);
  const CompactPublicState root_state =
      frozen_tables_->public_state_rows[*root_public_state_id].state;
  const int completed_warmup = run_warmup_phase(
      iterations, *root_public_state_id, root_state, range_sampler,
      player_a_hands_view, player_b_hands_view, max_depth,
      should_run_frozen_phase, can_use_frozen_regret_only);
  maybe_run_frozen_phase(iterations, completed_warmup, num_threads,
                         *root_public_state_id, range_sampler,
                         player_a_training_range, player_b_training_range);

  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_;
  LOG(INFO) << "Information sets: " << get_info_set_count();
  LOG(INFO) << "Public states: " << get_public_state_count();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

bool CFRSolver::prepare_frozen_training(
    uint32_t root_public_state_id,
    int num_threads,
    int max_depth,
    bool can_use_frozen_regret_only,
    const TrainingRangeView& player_a_hands_view,
    const TrainingRangeView& player_b_hands_view) {
  bool public_state_prebuild_complete = false;
  const bool should_prebuild_public_states =
      !frozen_ && (num_threads > 1 || can_use_frozen_regret_only) &&
      (config_.max_public_states > 0 || max_depth > 0);
  if (should_prebuild_public_states) {
    VLOG(1) << "Prebuilding compact public-state rows...";
    const auto prebuild_start = std::chrono::steady_clock::now();
    public_state_prebuild_complete =
        prebuild_public_state_rows(root_public_state_id, max_depth);
    const auto prebuild_end = std::chrono::steady_clock::now();
    last_training_run_stats_.prebuild_seconds =
        std::chrono::duration<double>(prebuild_end - prebuild_start).count();
    if (!public_state_prebuild_complete) {
      LOG(INFO) << "Public-state prebuild stopped before completion; "
                << "continuing without a frozen phase";
    }
  }
  last_training_run_stats_.public_state_prebuild_complete =
      should_prebuild_public_states && public_state_prebuild_complete;
  last_training_run_stats_.prebuild_public_states =
      should_prebuild_public_states
          ? static_cast<int64_t>(get_public_state_count())
          : 0;
  last_training_run_stats_.prebuild_betting_histories =
      should_prebuild_public_states
          ? static_cast<int64_t>(frozen_tables_->betting_history_rows.size())
          : 0;

  bool transition_prebuild_complete = false;
  if (should_prebuild_public_states && public_state_prebuild_complete) {
    transition_prebuild_complete = validate_prebuilt_transitions(
        root_public_state_id, max_depth, last_training_run_stats_);
    if (!last_training_run_stats_
             .betting_history_transition_prebuild_complete) {
      LOG(INFO) << "Betting-history transition prebuild validation failed; "
                << "continuing without a frozen phase";
    }
    if (last_training_run_stats_
            .betting_history_transition_prebuild_complete) {
      if (!last_training_run_stats_.chance_transition_prebuild_complete) {
        LOG(INFO) << "Chance-transition prebuild validation failed; "
                  << "continuing without a frozen phase";
      }
    }
    if (last_training_run_stats_
            .betting_history_transition_prebuild_complete &&
        last_training_run_stats_.chance_transition_prebuild_complete) {
      if (!last_training_run_stats_.action_transition_prebuild_complete) {
        LOG(INFO) << "Action-transition prebuild validation failed; "
                  << "continuing without a frozen phase";
      }
    }
  }

  bool info_set_prebuild_complete = false;
  if (should_prebuild_public_states && public_state_prebuild_complete &&
      transition_prebuild_complete) {
    VLOG(1) << "Prebuilding infoset rows...";
    const auto info_set_prebuild_start = std::chrono::steady_clock::now();
    info_set_prebuild_complete =
        prebuild_info_set_rows(player_a_hands_view, player_b_hands_view);
    const auto info_set_prebuild_end = std::chrono::steady_clock::now();
    last_training_run_stats_.info_set_prebuild_seconds =
        std::chrono::duration<double>(info_set_prebuild_end -
                                      info_set_prebuild_start)
            .count();
    if (!info_set_prebuild_complete) {
      LOG(INFO) << "Infoset prebuild stopped before completion; "
                << "continuing without a frozen phase";
    }
  }
  last_training_run_stats_.info_set_prebuild_complete =
      should_prebuild_public_states && public_state_prebuild_complete &&
      transition_prebuild_complete && info_set_prebuild_complete;

  bool private_bucket_prebuild_complete = false;
  if (last_training_run_stats_.info_set_prebuild_complete) {
    private_bucket_prebuild_complete = prebuild_private_bucket_rows();
    if (!private_bucket_prebuild_complete) {
      LOG(INFO) << "Private-bucket prebuild failed; "
                << "continuing without a frozen phase";
    }
  }
  last_training_run_stats_.private_bucket_prebuild_complete =
      last_training_run_stats_.info_set_prebuild_complete &&
      private_bucket_prebuild_complete;

  bool frozen_info_set_lookup_prebuild_complete = false;
  if (last_training_run_stats_.private_bucket_prebuild_complete) {
    frozen_info_set_lookup_prebuild_complete =
        prebuild_frozen_info_set_action_offsets();
    if (!frozen_info_set_lookup_prebuild_complete) {
      LOG(INFO) << "Frozen infoset lookup prebuild failed; "
                << "continuing without a frozen phase";
    }
  }
  last_training_run_stats_.frozen_info_set_lookup_prebuild_complete =
      last_training_run_stats_.private_bucket_prebuild_complete &&
      frozen_info_set_lookup_prebuild_complete;

  const bool prebuild_tables_are_action_complete =
      should_prebuild_public_states && public_state_prebuild_complete &&
      transition_prebuild_complete;
  last_training_run_stats_.prebuild_info_sets =
      prebuild_tables_are_action_complete
          ? static_cast<int64_t>(get_info_set_count())
          : 0;
  last_training_run_stats_.prebuild_action_entries =
      prebuild_tables_are_action_complete
          ? static_cast<int64_t>(cumulative_->cumulative_regrets.size())
          : 0;
  last_training_run_stats_.prebuild_private_bucket_rows =
      last_training_run_stats_.private_bucket_prebuild_complete
          ? static_cast<int64_t>(frozen_tables_->private_bucket_rows.size())
          : 0;
  last_training_run_stats_.prebuild_frozen_info_set_lookup_rows =
      last_training_run_stats_.frozen_info_set_lookup_prebuild_complete
          ? static_cast<int64_t>(
                frozen_tables_->frozen_info_set_action_offsets.size())
          : 0;

  const bool can_run_frozen_phase =
      last_training_run_stats_.public_state_prebuild_complete &&
      last_training_run_stats_.betting_history_transition_prebuild_complete &&
      last_training_run_stats_.chance_transition_prebuild_complete &&
      last_training_run_stats_.action_transition_prebuild_complete &&
      last_training_run_stats_.info_set_prebuild_complete &&
      last_training_run_stats_.private_bucket_prebuild_complete &&
      last_training_run_stats_.frozen_info_set_lookup_prebuild_complete;
  return can_run_frozen_phase &&
         (num_threads > 1 || can_use_frozen_regret_only);
}

int CFRSolver::run_warmup_phase(
    int iterations,
    uint32_t root_public_state_id,
    const CompactPublicState& root_state,
    RangeSampler& range_sampler,
    const TrainingRangeView& player_a_hands_view,
    const TrainingRangeView& player_b_hands_view,
    int max_depth,
    bool should_run_frozen_phase,
    bool can_use_frozen_regret_only) {
  const bool auto_warmup =
      should_run_frozen_phase && !frozen_ && config_.warmup_iterations <= 0;
  int warmup_count = iterations;
  if (should_run_frozen_phase && !frozen_ &&
      config_.warmup_iterations > 0) {
    const int requested_warmup =
        can_use_frozen_regret_only
            ? config_.warmup_iterations
            : std::max(config_.warmup_iterations, kPlayerCount);
    warmup_count = std::min(requested_warmup, iterations);
  }

  LOG(INFO) << "Starting CFR iterations...";
  VLOG(1) << "Warmup phase: "
          << (auto_warmup ? "adaptive" : std::to_string(warmup_count))
          << " single-threaded iterations";
  TraversalScratch scratch;
  scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
  const int64_t warmup_start_updates = cfr_update_count_;
  const auto warmup_start = std::chrono::steady_clock::now();
  int completed_warmup = 0;
  int no_growth_iterations = 0;
  size_t previous_info_sets = get_info_set_count();
  size_t previous_public_states = get_public_state_count();

  for (int i = 0; i < warmup_count; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    PrivateCards player_a_cards = PrivateCards::FromCombo(deal.player_a_combo);
    PrivateCards player_b_cards = PrivateCards::FromCombo(deal.player_b_combo);

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_;
    const int update_player = cfr_iteration % kPlayerCount;
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    double dealt_value = cfr_with_ranges(
        root_public_state_id, root_state, player_a_cards, player_b_cards,
        reach_probabilities, update_player, cfr_iteration, 0, max_depth,
        scratch, player_a_context_range, player_b_context_range);

    cumulative_root_utility_ += dealt_value;
    ++iterations_run_;
    ++completed_warmup;

    if (auto_warmup) {
      const size_t current_info_sets = get_info_set_count();
      const size_t current_public_states = get_public_state_count();
      if (current_info_sets == previous_info_sets &&
          current_public_states == previous_public_states) {
        ++no_growth_iterations;
      } else {
        no_growth_iterations = 0;
        previous_info_sets = current_info_sets;
        previous_public_states = current_public_states;
      }
      const bool info_set_cap_hit =
          config_.max_info_sets > 0 &&
          current_info_sets >= static_cast<size_t>(config_.max_info_sets);
      const bool public_state_cap_hit =
          config_.max_public_states > 0 &&
          current_public_states >=
              static_cast<size_t>(config_.max_public_states);
      // Cap hits mean the tree is incomplete; do not freeze and turn the
      // frozen phase into mostly skipped missing-child branches.
      if (!info_set_cap_hit && !public_state_cap_hit &&
          no_growth_iterations >= kAutoWarmupNoGrowthLimit) {
        break;
      }
    }
  }

  const auto warmup_end = std::chrono::steady_clock::now();
  last_training_run_stats_.warmup_iterations = completed_warmup;
  last_training_run_stats_.warmup_seconds =
      std::chrono::duration<double>(warmup_end - warmup_start).count();
  last_training_run_stats_.warmup_cfr_updates =
      cfr_update_count_ - warmup_start_updates;
  return completed_warmup;
}

void CFRSolver::maybe_run_frozen_phase(
    int iterations,
    int completed_warmup,
    int num_threads,
    uint32_t root_public_state_id,
    const RangeSampler& range_sampler,
    const TrainingRange& player_a_training_range,
    const TrainingRange& player_b_training_range) {
  const int remaining = iterations - completed_warmup;
  if (remaining <= 0) {
    return;
  }

  frozen_tables_ = mutable_tables_;
  mutable_tables_.reset();
  frozen_ = true;
  require_frozen_children_ = true;
  LOG(INFO) << "Frozen after warmup: " << get_info_set_count()
            << " info sets, " << iterations_run_
            << " warmup iterations. Starting frozen phase ("
            << remaining << " iterations, " << num_threads << " workers)...";
  const int64_t frozen_start_updates = cfr_update_count_;
  const auto frozen_start = std::chrono::steady_clock::now();
  run_frozen_iterations(remaining, num_threads, root_public_state_id,
                        range_sampler, player_a_training_range,
                        player_b_training_range);
  const auto frozen_end = std::chrono::steady_clock::now();
  last_training_run_stats_.frozen_iterations = remaining;
  last_training_run_stats_.frozen_seconds =
      std::chrono::duration<double>(frozen_end - frozen_start).count();
  last_training_run_stats_.frozen_cfr_updates =
      cfr_update_count_ - frozen_start_updates;
}

void CFRSolver::run_frozen_iterations(
    int iterations,
    int num_threads,
    uint32_t root_public_state_id,
    const RangeSampler& range_sampler,
    const TrainingRange& player_a_training_range,
    const TrainingRange& player_b_training_range) {
  // Each worker shares the frozen strategy tables and writes to the same
  // regret/strategy arrays.
  // Workers use their own RNG and TraversalScratch; no locks needed.
  std::shared_ptr<const FrozenStrategyTables> frozen_tables = frozen_tables_;
  std::shared_ptr<MutableCumulativeArrays> cumulative = cumulative_;
  const bool use_frozen_regret_only =
      frozen_ && require_frozen_children_ && config_.regret_only_training &&
      config_.max_depth == 0;
  const bool use_atomic_updates = num_threads > 1;

  ThreadPoolExecutor executor(num_threads);
  std::uniform_int_distribution<unsigned int> seed_dist;
  struct WorkerResult {
    double utility = 0.0;
    TraversalStats traversal_stats;
    int64_t cfr_updates = 0;
    int iterations = 0;
  };
  std::vector<std::future<WorkerResult>> futures;
  futures.reserve(num_threads);

  int iterations_remaining = iterations;
  int next_iteration = iterations_run_;
  for (int t = 0; t < num_threads; ++t) {
    const int shard = iterations_remaining / (num_threads - t);
    iterations_remaining -= shard;
    if (shard <= 0) {
      continue;
    }
    const int iteration_begin = next_iteration;
    next_iteration += shard;

    const unsigned int seed = seed_dist(rng_);
    futures.push_back(executor.submit(
        [this, shard, iteration_begin, seed, root_public_state_id,
         &range_sampler, frozen_tables, cumulative, use_frozen_regret_only,
         use_atomic_updates, &player_a_training_range,
         &player_b_training_range]() mutable {
          // Build a lightweight worker that shares frozen tables.
          CFRSolver worker(config_, utility_cache_);
          worker.mutable_tables_.reset();
          worker.frozen_tables_ = frozen_tables;
          worker.cumulative_ = cumulative;
          worker.frozen_ = true;
          worker.require_frozen_children_ = true;
          worker.rng_.seed(seed);

          TrainingRangeView player_a_hands_view;
          TrainingRangeView player_b_hands_view;
          const int max_depth = config_.max_depth;
          TraversalScratch scratch;
          if (!use_frozen_regret_only) {
            // Per-worker range views (read-only, built from shared training data).
            player_a_hands_view.reset_to_all(player_a_training_range);
            player_b_hands_view.reset_to_all(player_b_training_range);
            scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
          }

          double local_utility = 0.0;
          const CompactPublicState root_state =
              worker.frozen_tables_->public_state_rows[root_public_state_id]
                  .state;
          const ExactBoardState root_board{
              root_state.board_cards,
              root_state.board_count,
              root_state.board_mask,
          };
          for (int i = 0; i < shard; ++i) {
            const RangeDeal deal = range_sampler.sample(worker.rng_);
            PrivateCards player_a_cards =
                PrivateCards::FromCombo(deal.player_a_combo);
            PrivateCards player_b_cards =
                PrivateCards::FromCombo(deal.player_b_combo);

            const int cfr_iteration = iteration_begin + i;
            const int update_player = cfr_iteration % kPlayerCount;
            std::array<double, 2> reach_probabilities = {1.0, 1.0};
            if (use_frozen_regret_only) {
              local_utility += worker.cfr_frozen_regret_only(
                  root_public_state_id, root_board, player_a_cards,
                  player_b_cards, reach_probabilities, update_player, 0,
                  use_atomic_updates);
              continue;
            }

            OptionalTrainingRange player_a_context_range;
            OptionalTrainingRange player_b_context_range;
            if (max_depth > 0) {
              player_a_context_range = std::cref(player_a_hands_view);
              player_b_context_range = std::cref(player_b_hands_view);
            }
            local_utility += worker.cfr_with_ranges(
                root_public_state_id, root_state,
                player_a_cards, player_b_cards,
                reach_probabilities, update_player, cfr_iteration, 0,
                max_depth, scratch, player_a_context_range,
                player_b_context_range);
          }
          return WorkerResult{local_utility, worker.get_traversal_stats(),
                              worker.get_cfr_update_count(), shard};
        }));
  }

  // Accumulate per-worker root utilities and traversal stats into the main solver.
  int completed_iterations = 0;
  for (std::future<WorkerResult>& f : futures) {
    const WorkerResult result = f.get();
    cumulative_root_utility_ += result.utility;
    add_traversal_stats(result.traversal_stats);
    cfr_update_count_ += result.cfr_updates;
    completed_iterations += result.iterations;
  }
  iterations_run_ += completed_iterations;
}

double CFRSolver::cfr_with_ranges(
    uint32_t public_state_id,
    const CompactPublicState& state,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int update_player,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  const auto& public_state_rows = frozen_tables_->public_state_rows;
  if (public_state_id >= public_state_rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = public_state_rows[public_state_id];

  if (row.is_terminal) {
    record_terminal_utility(state.folded_player < 0);
    if (max_depth > 0) {
      return uncached_utility(state, player_a_cards, player_b_cards);
    }
    return utility(state, player_a_cards, player_b_cards);
  }

  if (row.is_chance_node) {
    return chance_sampling_cfr(public_state_id, state, player_a_cards,
                               player_b_cards, reach_probabilities,
                               update_player, iteration, depth,
                               max_depth, scratch, player_a_range,
                               player_b_range);
  }

  if (max_depth > 0 && depth >= max_depth) {
    (void)player_a_range;
    (void)player_b_range;
    return game_tree_->is_betting_round_over(state)
               ? uncached_utility(state, player_a_cards, player_b_cards)
               : 0.0;
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player) || row.action_count == 0) {
    return 0.0;
  }
  const StreetKind street = row.state.street;
  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;

  const InfoSetAddress info_set_address{
      public_state_id, player,
      card_abstraction_.private_bucket(player_cards.combo, row.state)};
  const bool is_update_player = player == update_player;
  const InfoSetRow* info_set_row =
      is_update_player
          ? get_or_create_info_set_row(
                info_set_address,
                absl::Span<const int>(row.action_ids.data(),
                                      row.action_count))
          : find_info_set_row(info_set_address);

  const size_t action_count = row.action_count;
  const size_t info_set_action_count =
      info_set_row == nullptr
          ? 0
          : static_cast<size_t>(info_set_row->action_count);
  const bool has_info_set_row =
      info_set_row != nullptr && info_set_action_count == action_count;
  const size_t info_set_action_offset =
      has_info_set_row ? info_set_row->action_offset : 0;
  double action_probabilities[GameTree::kMaxActionsPerNode];
  double action_values[GameTree::kMaxActionsPerNode];
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    action_values[action_index] = 0.0;
  }
  if (has_info_set_row) {
    auto& regrets = cumulative_->cumulative_regrets;
    double sum_positive_regrets = 0.0;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      record_action_entry_touches();
      const double positive_regret = std::max(
          0.0,
          static_cast<double>(
              AtomicFloatLoad(
                  &regrets[info_set_action_offset + action_index])));
      action_probabilities[action_index] = positive_regret;
      sum_positive_regrets += positive_regret;
    }

    if (sum_positive_regrets > 0.0) {
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] /= sum_positive_regrets;
      }
    } else {
      const double uniform_prob = 1.0 / action_count;
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] = uniform_prob;
      }
    }
  } else {
    const double uniform_prob = 1.0 / action_count;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      action_probabilities[action_index] = uniform_prob;
    }
  }

  double node_value = 0.0;
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  ConditionedRanges& conditioned_player_ranges =
      scratch_frame.conditioned_ranges;
  const bool condition_player_a_range =
      player == 0 && player_a_range.has_value();
  const bool condition_player_b_range =
      player == 1 && player_b_range.has_value();
  if (condition_player_a_range) {
    condition_ranges_for_actions(player_a_range->get(), state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  } else if (condition_player_b_range) {
    condition_ranges_for_actions(player_b_range->get(), state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    uint32_t child_public_state_id = GameTree::kInvalidPublicStateId;
    if (frozen_ && require_frozen_children_) {
      child_public_state_id = row.action_child_ids[action_index];
    } else if (frozen_) {
      std::optional<uint32_t> frozen_child_public_state_id =
          action_child_public_state(public_state_id,
                                    static_cast<int>(action_index));
      if (!frozen_child_public_state_id.has_value()) {
        continue;
      }
      child_public_state_id = *frozen_child_public_state_id;
    } else {
      std::optional<uint32_t> mutable_child_public_state_id =
          get_or_create_action_child_public_state(
              public_state_id, static_cast<int>(action_index));
      if (!mutable_child_public_state_id.has_value()) {
        continue;
      }
      child_public_state_id = *mutable_child_public_state_id;
    }
    const PublicStateRow& child_row =
        frozen_tables_->public_state_rows[child_public_state_id];
    const CompactPublicState child_state =
        StateWithBoardFrom(child_row.state, state);

    const double previous_reach_probability = reach_probabilities[player];
    reach_probabilities[player] =
        previous_reach_probability * action_probabilities[action_index];

    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (condition_player_a_range) {
      child_player_a_range = std::cref(conditioned_player_ranges[action_index]);
    } else if (condition_player_b_range) {
      child_player_b_range = std::cref(conditioned_player_ranges[action_index]);
    }

    const double action_value = cfr_with_ranges(
        child_public_state_id, child_state, player_a_cards, player_b_cards,
        reach_probabilities, update_player, iteration, depth + 1, max_depth,
        scratch, child_player_a_range, child_player_b_range);
    action_values[action_index] = action_value;
    reach_probabilities[player] = previous_reach_probability;
    node_value += action_probabilities[action_index] * action_value;
  }

  ++cfr_update_count_;
  record_cfr_update(street, depth);

  if (has_info_set_row && is_update_player) {
    const double opponent_reach_prob = reach_probabilities[1 - player];
    auto& regrets = cumulative_->cumulative_regrets;
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      record_action_entry_touches(2);
      AtomicCFRPlusRegretUpdate(
          &regrets[info_set_action_offset + action_index],
          static_cast<float>(regret));
    }

    if (!config_.regret_only_training) {
      update_strategy(info_set_action_offset, action_probabilities,
                      action_count,
                      reach_probabilities[player] * (iteration + 1));
    }
  }

  return node_value;
}

std::optional<CFRSolver::SampledChanceTransition>
CFRSolver::sample_chance_transition(uint32_t public_state_id,
                                    const CompactPublicState& state,
                                    CardMask known_private_cards) {
  const auto cards = SampleStreetCards(state, known_private_cards, rng_);
  CompactPublicState sampled_child_state =
      game_tree_->apply_chance(state, cards);
  uint32_t child_public_state_id = GameTree::kInvalidPublicStateId;
  if (frozen_ && require_frozen_children_) {
    std::optional<uint32_t> frozen_child_public_state_id =
        chance_child_public_state(public_state_id, sampled_child_state);
    if (!frozen_child_public_state_id.has_value() ||
        *frozen_child_public_state_id >=
            frozen_tables_->public_state_rows.size()) {
      throw std::logic_error("frozen chance child public state is missing");
    }
    child_public_state_id = *frozen_child_public_state_id;
  } else if (frozen_) {
    std::optional<uint32_t> frozen_child_public_state_id =
        chance_child_public_state(public_state_id, sampled_child_state);
    if (!frozen_child_public_state_id.has_value() ||
        *frozen_child_public_state_id >=
            frozen_tables_->public_state_rows.size()) {
      return std::nullopt;
    }
    child_public_state_id = *frozen_child_public_state_id;
  } else {
    std::optional<uint32_t> mutable_child_public_state_id =
        get_or_create_chance_child_public_state(public_state_id,
                                                sampled_child_state);
    if (!mutable_child_public_state_id.has_value() ||
        *mutable_child_public_state_id >=
            frozen_tables_->public_state_rows.size()) {
      return std::nullopt;
    }
    child_public_state_id = *mutable_child_public_state_id;
  }
  return SampledChanceTransition{
      child_public_state_id,
      std::move(sampled_child_state),
  };
}

CFRSolver::SampledFrozenChanceTransition
CFRSolver::sample_frozen_chance_transition(
    uint32_t public_state_id,
    const PublicStateRow& row,
    const ExactBoardState& exact_board,
    CardMask known_private_cards) {
  const auto cards = SampleStreetCards(
      row.state.street, exact_board.count, exact_board.mask,
      known_private_cards, rng_);
  ExactBoardState child_board = exact_board;
  for (CardId card : cards) {
    child_board.cards[child_board.count] = card;
    ++child_board.count;
    child_board.mask |= CardBit(card);
  }

  CompactPublicState sampled_child_state = row.state;
  switch (sampled_child_state.street) {
    case StreetKind::kPreflop:
      sampled_child_state.street = StreetKind::kFlop;
      break;
    case StreetKind::kFlop:
      sampled_child_state.street = StreetKind::kTurn;
      break;
    case StreetKind::kTurn:
      sampled_child_state.street = StreetKind::kRiver;
      break;
    case StreetKind::kRiver:
      break;
  }
  sampled_child_state.board_cards = child_board.cards;
  sampled_child_state.board_count = child_board.count;
  sampled_child_state.board_mask = child_board.mask;
  ResetHistory(sampled_child_state);
  sampled_child_state.player_to_act =
      sampled_child_state.street == StreetKind::kPreflop ? 0 : 1;

  std::optional<uint32_t> child_public_state_id =
      chance_child_public_state(public_state_id, sampled_child_state);
  if (!child_public_state_id.has_value() ||
      *child_public_state_id >= frozen_tables_->public_state_rows.size()) {
    throw std::logic_error("frozen chance child public state is missing");
  }

  return SampledFrozenChanceTransition{
      *child_public_state_id,
      child_board,
  };
}

double CFRSolver::chance_sampling_cfr(
    uint32_t public_state_id,
    const CompactPublicState& state,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int update_player,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return 0.0;
  }

  const int samples = ChanceSamples(config_);
  record_chance_samples(samples);
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  TrainingRangeView& public_player_a_range =
      scratch_frame.public_player_a_range;
  TrainingRangeView& public_player_b_range =
      scratch_frame.public_player_b_range;

  double value = 0.0;
  int evaluated = 0;
  const CardMask known_private_cards =
      player_a_cards.mask() | player_b_cards.mask();
  for (int i = 0; i < samples; ++i) {
    std::optional<SampledChanceTransition> sampled =
        sample_chance_transition(public_state_id, state, known_private_cards);
    if (!sampled.has_value()) {
      continue;
    }

    const PublicStateRow& child_row =
        frozen_tables_->public_state_rows[sampled->child_public_state_id];
    const CompactPublicState child_state =
        StateWithBoardFrom(child_row.state, sampled->exact_child_state);
    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (player_a_range.has_value()) {
      PublicCompatibleRangeInto(
          player_a_range->get(), child_state.board_mask, public_player_a_range);
      child_player_a_range = std::cref(public_player_a_range);
    }
    if (player_b_range.has_value()) {
      PublicCompatibleRangeInto(
          player_b_range->get(), child_state.board_mask, public_player_b_range);
      child_player_b_range = std::cref(public_player_b_range);
    }

    value += cfr_with_ranges(sampled->child_public_state_id, child_state,
                             player_a_cards, player_b_cards,
                             reach_probabilities, update_player, iteration,
                             depth, max_depth, scratch, child_player_a_range,
                             child_player_b_range);
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

double CFRSolver::cfr_frozen_regret_only(
    uint32_t public_state_id,
    const ExactBoardState& exact_board,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int update_player,
    int depth,
    bool use_atomic_updates) {
  const auto& public_state_rows = frozen_tables_->public_state_rows;
  if (public_state_id >= public_state_rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = public_state_rows[public_state_id];

  if (row.is_terminal) {
    record_terminal_utility(row.state.folded_player < 0);
    return frozen_utility(row, exact_board, player_a_cards, player_b_cards);
  }

  if (row.is_chance_node) {
    return chance_sampling_frozen_regret_only(
        public_state_id, exact_board, player_a_cards, player_b_cards,
        reach_probabilities, update_player, depth, use_atomic_updates);
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player) || row.action_count == 0) {
    return 0.0;
  }
  const StreetKind street = row.state.street;
  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;
  const size_t action_count = row.action_count;
  const PrivateBucketId private_bucket =
      private_bucket_for_frozen_row(public_state_id, player_cards.combo);
  const uint32_t info_set_action_offset =
      frozen_info_set_action_offset(public_state_id, player, private_bucket);
  const bool has_info_set_row =
      info_set_action_offset != FrozenStrategyTables::kInvalidActionOffset;

  double action_probabilities[GameTree::kMaxActionsPerNode];
  double action_values[GameTree::kMaxActionsPerNode];
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    action_values[action_index] = 0.0;
  }
  if (has_info_set_row) {
    auto& regrets = cumulative_->cumulative_regrets;
    double sum_positive_regrets = 0.0;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      record_action_entry_touches();
      const float raw_regret =
          use_atomic_updates
              ? AtomicFloatLoad(&regrets[info_set_action_offset +
                                         action_index])
              : regrets[info_set_action_offset + action_index];
      const double positive_regret =
          std::max(0.0, static_cast<double>(raw_regret));
      action_probabilities[action_index] = positive_regret;
      sum_positive_regrets += positive_regret;
    }

    if (sum_positive_regrets > 0.0) {
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] /= sum_positive_regrets;
      }
    } else {
      const double uniform_prob = 1.0 / action_count;
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] = uniform_prob;
      }
    }
  } else {
    const double uniform_prob = 1.0 / action_count;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      action_probabilities[action_index] = uniform_prob;
    }
  }

  double node_value = 0.0;
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    const uint32_t child_public_state_id =
        row.action_child_ids[action_index];

    const double previous_reach_probability = reach_probabilities[player];
    reach_probabilities[player] =
        previous_reach_probability * action_probabilities[action_index];

    const double action_value = cfr_frozen_regret_only(
        child_public_state_id, exact_board, player_a_cards, player_b_cards,
        reach_probabilities, update_player, depth + 1, use_atomic_updates);
    action_values[action_index] = action_value;
    reach_probabilities[player] = previous_reach_probability;
    node_value += action_probabilities[action_index] * action_value;
  }

  ++cfr_update_count_;
  record_cfr_update(street, depth);

  if (has_info_set_row && player == update_player) {
    const double opponent_reach_prob = reach_probabilities[1 - player];
    auto& regrets = cumulative_->cumulative_regrets;
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      record_action_entry_touches(2);
      float* regret_entry = &regrets[info_set_action_offset + action_index];
      if (use_atomic_updates) {
        const int64_t retries = AtomicCFRPlusRegretUpdate(
            regret_entry, static_cast<float>(regret));
        record_atomic_regret_update_retries(retries);
      } else {
        *regret_entry = std::max(0.0f,
                                 *regret_entry + static_cast<float>(regret));
      }
    }
  }

  return node_value;
}

double CFRSolver::chance_sampling_frozen_regret_only(
    uint32_t public_state_id,
    const ExactBoardState& exact_board,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int update_player,
    int depth,
    bool use_atomic_updates) {
  const PublicStateRow& row = frozen_tables_->public_state_rows[public_state_id];
  const int samples = ChanceSamples(config_);
  record_chance_samples(samples);

  double value = 0.0;
  const CardMask known_private_cards =
      player_a_cards.mask() | player_b_cards.mask();
  for (int i = 0; i < samples; ++i) {
    const SampledFrozenChanceTransition sampled =
        sample_frozen_chance_transition(public_state_id, row, exact_board,
                                        known_private_cards);
    value += cfr_frozen_regret_only(
        sampled.child_public_state_id, sampled.child_board, player_a_cards,
        player_b_cards, reach_probabilities, update_player, depth,
        use_atomic_updates);
  }

  return samples > 0 ? value / samples : 0.0;
}

void CFRSolver::average_strategy_probabilities(
    uint32_t public_state_id,
    const PublicStateRow& row,
    int player,
    const PrivateCards& private_cards,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(row.action_count, 0.0);
  if (row.action_count == 0) {
    return;
  }

  const double uniform_probability = 1.0 / row.action_count;
  const PrivateBucketId private_bucket =
      card_abstraction_.private_bucket(private_cards.combo, row.state);
  const InfoSetRow* info_set_row =
      find_info_set_row({public_state_id, player, private_bucket});
  if (info_set_row == nullptr) {
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }

  average_strategy_probabilities(
      *info_set_row, row, uniform_probability, probabilities);
}

void CFRSolver::average_strategy_probabilities(
    const InfoSetRow& info_set_row,
    const PublicStateRow& public_state_row,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  const int num_actions = public_state_row.action_count;
  probabilities.clear();
  probabilities.resize(num_actions, 0.0);
  double probability_sum = 0.0;
  const size_t action_offset = info_set_row.action_offset;
  const std::vector<int>& action_ids = frozen_tables_->action_ids;
  const auto& cumulative_regrets = cumulative_->cumulative_regrets;
  const auto& cumulative_strategies = cumulative_->cumulative_strategies;

  const bool aligned_action_ids =
      num_actions == info_set_row.action_count &&
      [&]() {
        for (int i = 0; i < num_actions; ++i) {
          if (public_state_row.action_ids[static_cast<size_t>(i)] !=
              action_ids[action_offset + i]) {
            return false;
          }
        }
        return true;
      }();
  if (aligned_action_ids) {
    for (int i = 0; i < num_actions; ++i) {
      record_action_entry_touches();
      const size_t table_index = action_offset + i;
      probabilities[i] =
          config_.regret_only_training
              ? std::max(0.0,
                         static_cast<double>(
                             AtomicFloatLoad(&cumulative_regrets[table_index])))
              : static_cast<double>(
                    AtomicFloatLoad(&cumulative_strategies[table_index]));
      probability_sum += probabilities[i];
    }
  } else {
    for (int legal_action_index = 0; legal_action_index < num_actions;
         ++legal_action_index) {
      const int legal_action_id =
          public_state_row.action_ids[static_cast<size_t>(legal_action_index)];
      for (uint16_t action_index = 0;
           action_index < info_set_row.action_count; ++action_index) {
        record_action_entry_touches();
        const size_t table_index = action_offset + action_index;
        if (action_ids[table_index] == legal_action_id) {
          probabilities[legal_action_index] =
              config_.regret_only_training
                  ? std::max(
                        0.0,
                        static_cast<double>(
                            AtomicFloatLoad(&cumulative_regrets[table_index])))
                  : static_cast<double>(
                        AtomicFloatLoad(&cumulative_strategies[table_index]));
          probability_sum += probabilities[legal_action_index];
          break;
        }
      }
    }
  }

  if (probability_sum <= 0.0) {
    std::fill(probabilities.begin(), probabilities.end(), fallback_probability);
    return;
  }

  for (double& probability : probabilities) {
    probability /= probability_sum;
  }
}

void CFRSolver::condition_ranges_for_actions(
    const TrainingRangeView& range,
    const CompactPublicState& state,
    uint32_t public_state_id,
    int player,
    const int* conditioned_action_ids,
    size_t action_count,
    ConditionedRanges& conditioned_ranges) {
  if (action_count == 0) {
    return;
  }

  if (conditioned_ranges.size() < action_count) {
    conditioned_ranges.resize(action_count);
  }
  for (size_t i = 0; i < action_count; ++i) {
    conditioned_ranges[i].reset_to_filtered();
  }
  const size_t range_size = range.size();
  if (range_size == 0) {
    return;
  }

  const double fallback_probability = 1.0 / action_count;
  const CardMask board_mask = state.board_mask;
  const PublicInfoSetSlab* public_slab =
      public_info_set_slab(public_state_id);
  const PublicInfoSetSlabPlayer* player_slab =
      public_slab != nullptr ? &public_slab->players[player] : nullptr;
  const auto& action_ids = frozen_tables_->action_ids;
  const auto& regrets = cumulative_->cumulative_regrets;
  double positive_regrets[GameTree::kMaxActionsPerNode] = {};
  for (size_t i = 0; i < range_size; ++i) {
    const float range_weight = range.weight(i);
    const ComboId combo_id = range.combo(i);
    if (range_weight <= 0.0 || (ComboMask(combo_id) & board_mask) != 0) {
      continue;
    }

    double positive_regret_sum = 0.0;
    const PrivateBucketId private_bucket =
        card_abstraction_.private_bucket(combo_id, state);
    const InfoSetRow* row = nullptr;
    if (player_slab != nullptr) {
      row = find_info_set_row(*player_slab, private_bucket);
    }

    if (row != nullptr) {
      const size_t table_offset = row->action_offset;
      std::fill(positive_regrets, positive_regrets + action_count, 0.0);
      const size_t info_set_action_count =
          std::min(action_count,
                   static_cast<size_t>(row->action_count));
      for (size_t action_index = 0; action_index < info_set_action_count;
           ++action_index) {
        const size_t table_index = table_offset + action_index;
        if (action_ids[table_index] !=
            conditioned_action_ids[action_index]) {
          continue;
        }

        record_action_entry_touches();
        const double positive_regret =
            std::max(
                0.0,
                static_cast<double>(
                    AtomicFloatLoad(&regrets[table_index])));
        positive_regrets[action_index] = positive_regret;
        positive_regret_sum += positive_regret;
      }
    }

    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      const double probability =
          positive_regret_sum > 0.0
              ? positive_regrets[action_index] / positive_regret_sum
              : fallback_probability;
      const double conditioned_weight = range_weight * probability;
      if (conditioned_weight > 0.0) {
        conditioned_ranges[action_index].add(
            combo_id, static_cast<float>(conditioned_weight));
      }
    }
  }
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }
  const CompactPublicState root_state =
      frozen_tables_->public_state_rows[*root_public_state_id].state;
  return evaluate_strategy_node(
      *root_public_state_id, root_state,
      PrivateCards::FromCombo(player_a_hand),
      PrivateCards::FromCombo(player_b_hand));
}

double CFRSolver::evaluate_strategy(int samples, const HandRange& player_a_range,
                                    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }

  if (samples < kParallelEvaluationSampleThreshold) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  ThreadPoolExecutor executor(worker_count);
  std::uniform_int_distribution<unsigned int> seed_distribution;
  std::vector<unsigned int> worker_seeds;
  worker_seeds.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    worker_seeds.push_back(seed_distribution(rng_));
  }

  SolverConfig config = config_;
  std::shared_ptr<const FrozenStrategyTables> frozen_tables = frozen_tables_;
  std::shared_ptr<MutableCumulativeArrays> cumulative = cumulative_;
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, range_sampler,
                                       root_public_state_id =
                                           *root_public_state_id,
                                       frozen_tables, cumulative,
                                       shard_samples, seed]() mutable {
      CFRSolver worker(config, std::make_shared<TerminalUtilityCache>());
      worker.mutable_tables_.reset();
      worker.frozen_tables_ = frozen_tables;
      worker.cumulative_ = cumulative;
      worker.frozen_ = true;
      worker.rng_.seed(seed);
      const double value = worker.evaluate_strategy_samples(
          shard_samples, root_public_state_id, range_sampler);
      return std::make_pair(
          value * shard_samples,
          worker.get_traversal_stats().action_entry_touches);
    }));
  }

  double total = 0.0;
  for (std::future<std::pair<double, int64_t>>& future : futures) {
    const std::pair<double, int64_t> result = future.get();
    total += result.first;
    record_action_entry_touches(result.second);
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_samples(
    int samples,
    uint32_t root_public_state_id,
    RangeSampler range_sampler) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  const CompactPublicState root_state =
      frozen_tables_->public_state_rows[root_public_state_id].state;
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    total += evaluate_strategy_node(
        root_public_state_id, root_state,
        PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo));
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    uint32_t public_state_id,
    const CompactPublicState& state,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = rows[public_state_id];

  if (row.is_terminal) {
    return utility(state, player_a_cards, player_b_cards);
  }
  if (row.is_chance_node) {
    const int samples = ChanceSamples(config_);
    double value = 0.0;
    int evaluated = 0;
    const CardMask known_private_cards =
        player_a_cards.mask() | player_b_cards.mask();
    for (int i = 0; i < samples; ++i) {
      std::optional<SampledChanceTransition> sampled =
          sample_chance_transition(public_state_id, state, known_private_cards);
      if (!sampled.has_value()) {
        continue;
      }
      const PublicStateRow& child_row =
          frozen_tables_->public_state_rows[sampled->child_public_state_id];
      const CompactPublicState child_state =
          StateWithBoardFrom(child_row.state, sampled->exact_child_state);
      value += evaluate_strategy_node(
          sampled->child_public_state_id, child_state, player_a_cards,
          player_b_cards);
      ++evaluated;
    }
    return evaluated > 0 ? value / evaluated : 0.0;
  }
  if (row.action_count == 0) {
    return 0.0;
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player)) {
    return 0.0;
  }

  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;
  StrategyProbabilities probabilities;
  average_strategy_probabilities(
      public_state_id, row, player, player_cards, probabilities);

  double value = 0.0;
  const int action_count = row.action_count;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    uint32_t child_public_state_id = GameTree::kInvalidPublicStateId;
    if (frozen_ && require_frozen_children_) {
      std::optional<uint32_t> frozen_child_public_state_id =
          action_child_public_state(public_state_id, action_index);
      if (!frozen_child_public_state_id.has_value() ||
          *frozen_child_public_state_id >=
              frozen_tables_->public_state_rows.size()) {
        throw std::logic_error("frozen action child public state is missing");
      }
      child_public_state_id = *frozen_child_public_state_id;
    } else if (frozen_) {
      std::optional<uint32_t> frozen_child_public_state_id =
          action_child_public_state(public_state_id, action_index);
      if (!frozen_child_public_state_id.has_value()) {
        continue;
      }
      child_public_state_id = *frozen_child_public_state_id;
    } else {
      std::optional<uint32_t> mutable_child_public_state_id =
          get_or_create_action_child_public_state(public_state_id,
                                                  action_index);
      if (!mutable_child_public_state_id.has_value()) {
        continue;
      }
      child_public_state_id = *mutable_child_public_state_id;
    }
    const PublicStateRow& child_row =
        frozen_tables_->public_state_rows[child_public_state_id];
    const CompactPublicState child_state =
        StateWithBoardFrom(child_row.state, state);
    value += probabilities[action_index] *
             evaluate_strategy_node(child_public_state_id, child_state,
                                    player_a_cards, player_b_cards);
  }
  return value;
}

double CFRSolver::get_expected_value(int player_id) const {
  const int iters = iterations_run_;
  if (iters == 0) {
    return 0.0;
  }
  double player_a_ev = cumulative_root_utility_ / iters;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

CFRSolver::UtilityCacheStats CFRSolver::get_utility_cache_stats() const {
  TerminalUtilityCache::Stats stats = utility_cache_->stats();
  return {stats.hits, stats.misses, stats.entries};
}

bool CFRSolver::traversal_stats_enabled() {
  return kTraversalStatsEnabled;
}

double CFRSolver::utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, state.board_count)) {
    return *utility;
  }

  return utility_cache_->get_or_compute(
      state, player_a_cards.combo, player_b_cards.combo, [&]() {
        return game_tree_->get_utility(
            state, player_a_cards.combo, player_b_cards.combo);
      });
}

double CFRSolver::frozen_utility(const PublicStateRow& row,
                                 const ExactBoardState& exact_board,
                                 const PrivateCards& player_a_cards,
                                 const PrivateCards& player_b_cards) {
  const CompactPublicState& state = row.state;
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, exact_board.count)) {
    return *utility;
  }

  // Frozen sampled training sees mostly one-off showdowns; direct evaluation
  // is faster than paying shared cache lookup/mutation overhead.
  HandEvaluator evaluator;
  const int comparison =
      evaluator.compare_hands(player_a_cards.combo, player_b_cards.combo,
                              exact_board.cards, exact_board.count);
  return ShowdownUtilityFromComparison(state, comparison);
}

double CFRSolver::uncached_utility(const CompactPublicState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, state.board_count)) {
    return *utility;
  }

  return game_tree_->get_utility(
      state, player_a_cards.combo, player_b_cards.combo);
}

void CFRSolver::update_strategy(size_t action_offset,
                                const double* action_probabilities,
                                size_t action_count,
                                double reach_prob) {
  auto& strategies = cumulative_->cumulative_strategies;
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    record_action_entry_touches(2);
    const float delta = static_cast<float>(
        reach_prob * action_probabilities[action_index]);
    AtomicFloatAdd(&strategies[action_offset + action_index], delta);
  }
}

} // namespace poker
