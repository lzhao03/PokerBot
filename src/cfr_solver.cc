#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "src/best_response.h"
#include "src/card_utils.h"
#include "src/continuation_value.h"
#include "src/hand_range.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;

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
inline void AtomicCFRPlusRegretUpdate(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float old_val;
    std::memcpy(&old_val, &old_bits, sizeof(float));
    float new_val = std::max(0.0f, old_val + delta);
    std::memcpy(&new_bits, &new_val, sizeof(float));
  } while (!__atomic_compare_exchange_n(
      reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
      /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
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

#ifndef POKER_ENABLE_TRAVERSAL_STATS
#define POKER_ENABLE_TRAVERSAL_STATS 1
#endif

#if POKER_ENABLE_TRAVERSAL_STATS
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
    statement;                                 \
  } while (false)
#define POKER_TRAVERSAL_STAT_PTR(member) (&(member))
#else
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
  } while (false)
#define POKER_TRAVERSAL_STAT_PTR(member) nullptr
#endif

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

void PublicCompatibleRangeInto(const TrainingRangeView& hands,
                               const GameState& state,
                               TrainingRangeView& compatible_hands) {
  PublicCompatibleRangeInto(hands, state.board_mask, compatible_hands);
}

int ChanceCardsKey(absl::Span<const CardId> cards) {
  int encoded[5];
  const int n = static_cast<int>(cards.size());
  for (int i = 0; i < n; ++i) {
    encoded[i] = EncodedCard(cards[i]);
  }
  // Chance samples are small (max 5), so a simple sort network is cheaper than
  // calling std::sort on a dynamically-sized vector.
  if (n > 1) {
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (encoded[j] < encoded[i]) {
          std::swap(encoded[i], encoded[j]);
        }
      }
    }
  }

  int key = n;
  for (int i = 0; i < n; ++i) {
    key = key * 128 + encoded[i];
  }
  return -1 - key;
}

uint64_t PublicChanceChildKey(uint32_t parent_id, int child_key) {
  return (static_cast<uint64_t>(parent_id) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(child_key));
}

int RoundedContribution(const GameState& state, int player) {
  return state.player_contribution[player];
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


CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>()) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const GameState& initial_state)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>(),
                std::make_shared<BettingRoundTerminalValueProvider>(),
                initial_state) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     std::shared_ptr<TerminalUtilityCache> utility_cache)
    : CFRSolver(config, std::move(utility_cache),
                std::make_shared<BettingRoundTerminalValueProvider>()) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider)
    : CFRSolver(config, std::move(utility_cache),
                std::move(continuation_value_provider),
                DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider,
    GameState initial_state)
  : config_(config),
    initial_state_(std::move(initial_state)),
    game_tree_(std::make_shared<GameTree>(config)),
    rng_(12345),
    cumulative_root_utility_(0.0),
    utility_cache_(std::move(utility_cache)),
    continuation_value_provider_(std::move(continuation_value_provider)),
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
    mutable_tables_->public_state_rows.reserve(
        static_cast<size_t>(config_.max_public_states));
  }
}

FrozenStrategyTables& CFRSolver::mutable_tables() {
  if (frozen_ || mutable_tables_ == nullptr) {
    throw std::logic_error("Strategy tables are frozen");
  }
  return *mutable_tables_;
}

void CFRSolver::set_continuation_value_provider(
    std::shared_ptr<ContinuationValueProvider> provider) {
  if (provider == nullptr) {
    throw std::invalid_argument("Continuation value provider cannot be null");
  }
  continuation_value_provider_ = std::move(provider);
}

GameTree::Node& CFRSolver::get_or_build_root() {
  if (!game_tree_->has_root()) {
    return game_tree_->build_tree(initial_state_);
  }
  return game_tree_->root();
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
    const GameState& state) const {
  BettingHistoryKey key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  for (int i = 0; i < key.player_contribution_size; ++i) {
    key.player_contributions[i] = RoundedContribution(state, i);
  }

  const int history_value_count = state.history.size() * 3;
  if (history_value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(history_value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  auto add_history_value = [&key](int value) {
    if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
      key.history_values[key.history_size] = value;
    } else {
      key.history_overflow.push_back(value);
    }
    ++key.history_size;
  };
  for (const GameAction& action : state.history) {
    add_history_value(action.player);
    add_history_value(static_cast<int>(action.kind));
    add_history_value(action.amount);
  }

  return key;
}

CFRSolver::BettingHistoryKey CFRSolver::make_betting_history_key(
    const CompactPublicState& state) const {
  BettingHistoryKey key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  key.player_contributions = state.player_contribution;

  const int history_value_count = state.history_size * 3;
  if (history_value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(history_value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  auto add_history_value = [&key](int value) {
    if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
      key.history_values[key.history_size] = value;
    } else {
      key.history_overflow.push_back(value);
    }
    ++key.history_size;
  };
  for (uint16_t i = 0; i < state.history_size; ++i) {
    const CompactAction action = CompactHistoryAction(state, i);
    add_history_value(action.player);
    add_history_value(static_cast<int>(action.kind));
    add_history_value(action.amount);
  }

  return key;
}

CFRSolver::BettingHistoryRow CFRSolver::make_betting_history_row(
    const GameState& state) const {
  BettingHistoryRow row;
  row.street = static_cast<int>(state.street);
  row.pot = state.pot;
  row.stack = {state.stack[0], state.stack[1]};
  row.all_in = state.all_in ? 1 : 0;
  row.folded_player = state.folded_player;
  row.player_to_act = state.player_to_act;
  row.player_contributions = {
      RoundedContribution(state, 0),
      RoundedContribution(state, 1),
  };
  return row;
}

CFRSolver::BettingHistoryRow CFRSolver::make_betting_history_row(
    const CompactPublicState& state) const {
  BettingHistoryRow row;
  row.street = static_cast<int>(state.street);
  row.pot = state.pot;
  row.stack = state.stack;
  row.all_in = state.all_in ? 1 : 0;
  row.folded_player = state.folded_player;
  row.player_to_act = state.player_to_act;
  row.player_contributions = state.player_contribution;
  return row;
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    uint32_t betting_history_id,
    const GameState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    uint32_t betting_history_id,
    const CompactPublicState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

CompactPublicState
CFRSolver::compact_public_state_from_game_state(
    const GameState& state) {
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

GameState CFRSolver::materialize_game_state(
    const CompactPublicState& compact) const {
  GameState state;
  state.stack[0] = compact.stack[0];
  state.stack[1] = compact.stack[1];
  state.pot = compact.pot;
  for (int i = 0; i < compact.board_count; ++i) {
    AddBoardCard(state, compact.board_cards[static_cast<size_t>(i)]);
  }
  if (state.board_mask != compact.board_mask) {
    throw std::logic_error("Compact public state board mask mismatch");
  }

  state.history.reserve(compact.history_size);
  const uint16_t inline_history_size = std::min<uint16_t>(
      compact.history_size, CompactPublicState::kMaxHistoryActions);
  for (uint16_t i = 0; i < inline_history_size; ++i) {
    state.history.push_back(MakeGameAction(CompactHistoryAction(compact, i)));
  }

  state.street = compact.street;
  state.all_in = compact.all_in;
  state.folded_player = compact.folded_player;
  state.player_to_act = compact.player_to_act;
  state.player_contribution = compact.player_contribution;
  state.player_contribution_count = compact.player_contribution_count;
  return state;
}

CFRSolver::PublicStateRow CFRSolver::make_public_state_row(
    uint32_t betting_history_id,
    const GameState& state) {
  return make_public_state_row(
      betting_history_id, compact_public_state_from_game_state(state));
}

CFRSolver::PublicStateRow CFRSolver::make_public_state_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
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
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(tables.action_ids.size());
  row.action_count = static_cast<uint16_t>(action_ids.size());
  const size_t required_action_capacity =
      tables.action_ids.size() + action_ids.size();
  if (required_action_capacity > tables.action_ids.capacity()) {
    const size_t new_capacity =
        std::max(required_action_capacity,
                 tables.action_ids.empty() ? size_t{4096}
                                           : tables.action_ids.capacity() * 2);
    tables.action_ids.reserve(new_capacity);
    cumulative_->cumulative_regrets.reserve(new_capacity);
    cumulative_->cumulative_strategies.reserve(new_capacity);
  }
  for (int action_id : action_ids) {
    tables.action_ids.push_back(action_id);
    cumulative_->cumulative_regrets.push_back(0.0f);
    cumulative_->cumulative_strategies.push_back(0.0f);
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
  }
  return row;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    uint32_t betting_history_id,
    const GameState& state) {
  if (frozen_) {
    auto existing = frozen_tables_->public_state_ids.find(
        make_public_state_key(betting_history_id, state));
    if (existing == frozen_tables_->public_state_ids.end()) {
      return std::nullopt;
    }
    return existing->second;
  }

  PublicStateKey key = make_public_state_key(betting_history_id, state);
  auto existing = frozen_tables_->public_state_ids.find(key);
  if (existing != frozen_tables_->public_state_ids.end()) {
    return existing->second;
  }

  FrozenStrategyTables& tables = mutable_tables();
  if (config_.max_public_states > 0 &&
      static_cast<int>(tables.public_state_rows.size()) >=
          config_.max_public_states) {
    return std::nullopt;
  }

  const uint32_t public_state_id =
      static_cast<uint32_t>(tables.public_state_ids.size());
  tables.public_state_ids.emplace(std::move(key), public_state_id);
  tables.public_state_rows.push_back(
      make_public_state_row(betting_history_id, state));
  cache_betting_history_actions(betting_history_id,
                                tables.public_state_rows.back());
  return public_state_id;
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

  PublicStateKey key = make_public_state_key(betting_history_id, state);
  auto existing = frozen_tables_->public_state_ids.find(key);
  if (existing != frozen_tables_->public_state_ids.end()) {
    return existing->second;
  }

  FrozenStrategyTables& tables = mutable_tables();
  if (config_.max_public_states > 0 &&
      static_cast<int>(tables.public_state_rows.size()) >=
          config_.max_public_states) {
    return std::nullopt;
  }

  const uint32_t public_state_id =
      static_cast<uint32_t>(tables.public_state_ids.size());
  tables.public_state_ids.emplace(std::move(key), public_state_id);
  tables.public_state_rows.push_back(
      make_public_state_row(betting_history_id, std::move(state)));
  cache_betting_history_actions(betting_history_id,
                                tables.public_state_rows.back());
  return public_state_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    const GameState& state) {
  if (frozen_) {
    const std::optional<uint32_t> betting_history_id =
        strategy_betting_history_id(state);
    if (!betting_history_id.has_value()) {
      return std::nullopt;
    }
    return get_or_create_public_state_row(*betting_history_id, state);
  }

  const uint32_t betting_history_id = get_or_create_betting_history_id(state);
  return get_or_create_public_state_row(betting_history_id, state);
}

uint32_t CFRSolver::get_or_create_betting_history_id(const GameState& state) {
  BettingHistoryKey key = make_betting_history_key(state);
  auto existing = frozen_tables_->betting_history_ids.find(key);
  if (existing != frozen_tables_->betting_history_ids.end()) {
    FrozenStrategyTables& tables = mutable_tables();
    if (tables.betting_history_rows.size() <= existing->second) {
      tables.betting_history_rows.resize(static_cast<size_t>(existing->second) + 1);
    }
    return existing->second;
  }
  FrozenStrategyTables& tables = mutable_tables();
  const uint32_t betting_history_id =
      static_cast<uint32_t>(tables.betting_history_ids.size());
  tables.betting_history_ids.emplace(std::move(key), betting_history_id);
  tables.betting_history_rows.push_back(make_betting_history_row(state));
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_betting_history_id(
    const CompactPublicState& state) {
  BettingHistoryKey key = make_betting_history_key(state);
  auto existing = frozen_tables_->betting_history_ids.find(key);
  if (existing != frozen_tables_->betting_history_ids.end()) {
    FrozenStrategyTables& tables = mutable_tables();
    if (tables.betting_history_rows.size() <= existing->second) {
      tables.betting_history_rows.resize(static_cast<size_t>(existing->second) + 1);
    }
    return existing->second;
  }
  FrozenStrategyTables& tables = mutable_tables();
  const uint32_t betting_history_id =
      static_cast<uint32_t>(tables.betting_history_ids.size());
  tables.betting_history_ids.emplace(std::move(key), betting_history_id);
  tables.betting_history_rows.push_back(make_betting_history_row(state));
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_betting_history_id(GameTree::Node& node) {
  uint32_t betting_history_id = node.betting_history_id;
  if (betting_history_id == GameTree::Node::kInvalidBettingHistoryId) {
    betting_history_id = get_or_create_betting_history_id(node.state);
    node.betting_history_id = betting_history_id;
  }
  cache_betting_history_actions(betting_history_id, node);
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
      if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
        return child_id;
      }
    }
  }

  const uint32_t child_id = get_or_create_betting_history_id(child_state);
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
        GameTree::Node::kInvalidBettingHistoryId) {
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
    const GameTree::Node& node) {
  FrozenStrategyTables& tables = mutable_tables();
  if (node.action_count == 0 ||
      betting_history_id >= tables.betting_history_rows.size()) {
    return;
  }

  BettingHistoryRow& row =
      tables.betting_history_rows[static_cast<size_t>(betting_history_id)];
  row.action_count = node.action_count;
  for (int i = 0; i < node.action_count; ++i) {
    row.action_ids[static_cast<size_t>(i)] = node.actions[i].key;
  }
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

void CFRSolver::validate_public_state_row_actions(
    uint32_t public_state_id) const {
  const auto& public_rows = frozen_tables_->public_state_rows;
  if (public_state_id >= public_rows.size()) {
    throw std::logic_error("Public state row is missing");
  }
  const PublicStateRow& row = public_rows[public_state_id];
  const auto& betting_history_rows = frozen_tables_->betting_history_rows;
  if (row.betting_history_id >= betting_history_rows.size()) {
    throw std::logic_error("Betting history row is missing");
  }
  const BettingHistoryRow& betting_history =
      betting_history_rows[row.betting_history_id];
  if (betting_history.action_count != row.action_count) {
    throw std::logic_error("Betting history actions are not aligned");
  }
  for (int i = 0; i < row.action_count; ++i) {
    if (betting_history.action_ids[static_cast<size_t>(i)] !=
        row.action_ids[static_cast<size_t>(i)]) {
      throw std::logic_error("Betting history action key mismatch");
    }
  }
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
  if (existing_child_id != GameTree::Node::kInvalidPublicStateId) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_hits);
    if (existing_child_id == kCappedPublicStateId) {
      return std::nullopt;
    }
    return existing_child_id;
  }
  if (frozen_) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(frozen_tables_->public_state_rows.size()) >=
          config_.max_public_states) {
    mutable_tables()
        .public_state_rows[public_state_id]
        .action_child_ids[action_slot] = kCappedPublicStateId;
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  const GameAction action =
      frozen_tables_->public_state_rows[public_state_id].actions[action_slot];
  const uint32_t parent_betting_history_id =
      frozen_tables_->public_state_rows[public_state_id].betting_history_id;
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
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  mutable_tables()
      .public_state_rows[public_state_id]
      .action_child_ids[action_slot] = *child_id;
  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.child_nodes_created);
  return child_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_chance_child_public_state(
    uint32_t public_state_id,
    absl::Span<const CardId> cards) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const int child_key = ChanceCardsKey(cards);
  const uint64_t map_key = PublicChanceChildKey(public_state_id, child_key);
  const auto& chance_children = frozen_tables_->public_chance_child_ids;
  auto existing = chance_children.find(map_key);
  if (existing != chance_children.end()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_hits);
    return existing->second;
  }
  if (frozen_) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(frozen_tables_->public_state_rows.size()) >=
          config_.max_public_states) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  const uint32_t parent_betting_history_id =
      frozen_tables_->public_state_rows[public_state_id].betting_history_id;
  CompactPublicState child_state = game_tree_->apply_chance(
      frozen_tables_->public_state_rows[public_state_id].state, cards);
  const uint32_t child_betting_history_id =
      get_or_create_chance_child_betting_history_id(
          parent_betting_history_id, child_state);
  std::optional<uint32_t> child_id =
      get_or_create_public_state_row(child_betting_history_id,
                                     std::move(child_state));
  if (!child_id.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  mutable_tables().public_chance_child_ids.emplace(map_key, *child_id);
  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.child_nodes_created);
  return child_id;
}

void CFRSolver::cache_action_betting_history_transition(
    GameTree::Node& node,
    int action_index,
    GameTree::Node& child_node) {
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error(
        "cache_action_betting_history_transition: action index out of range");
  }

  if (!frozen_) {
    const uint32_t parent_id = get_or_create_betting_history_id(node);
    FrozenStrategyTables& tables = mutable_tables();
    if (tables.betting_history_rows.size() <= parent_id) {
      tables.betting_history_rows.resize(static_cast<size_t>(parent_id) + 1);
    }
    const size_t parent_index = static_cast<size_t>(parent_id);
    const size_t action_slot = static_cast<size_t>(action_index);
    uint32_t child_id =
        tables.betting_history_rows[parent_index].action_child_ids[action_slot];
    if (child_id == GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_misses);
      child_id = get_or_create_betting_history_id(child_node);
      tables.betting_history_rows[parent_index].action_child_ids[action_slot] =
          child_id;
    } else {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
    }
    return;
  }

  const std::optional<uint32_t> parent_lookup =
      strategy_betting_history_id(node);
  if (!parent_lookup.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return;
  }

  const uint32_t parent_id = *parent_lookup;
  node.betting_history_id = parent_id;
  const auto& rows = frozen_tables_->betting_history_rows;
  if (parent_id < rows.size()) {
    const uint32_t child_id =
        rows[parent_id].action_child_ids[static_cast<size_t>(action_index)];
    if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
      return;
    }
  }

  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  const std::optional<uint32_t> child_lookup =
      strategy_betting_history_id(child_node);
  if (child_lookup.has_value()) {
    child_node.betting_history_id = *child_lookup;
  }
}

void CFRSolver::cache_chance_betting_history_transition(
    GameTree::Node& node,
    GameTree::Node& child_node) {
  if (!frozen_) {
    const uint32_t parent_id = get_or_create_betting_history_id(node);
    FrozenStrategyTables& tables = mutable_tables();
    if (tables.betting_history_rows.size() <= parent_id) {
      tables.betting_history_rows.resize(static_cast<size_t>(parent_id) + 1);
    }
    const size_t parent_index = static_cast<size_t>(parent_id);
    uint32_t child_id = tables.betting_history_rows[parent_index].chance_child_id;
    if (child_id == GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_misses);
      child_id = get_or_create_betting_history_id(child_node);
      tables.betting_history_rows[parent_index].chance_child_id = child_id;
    } else {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
    }
    return;
  }

  const std::optional<uint32_t> parent_lookup =
      strategy_betting_history_id(node);
  if (!parent_lookup.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return;
  }

  const uint32_t parent_id = *parent_lookup;
  node.betting_history_id = parent_id;
  const auto& rows = frozen_tables_->betting_history_rows;
  if (parent_id < rows.size()) {
    const uint32_t child_id = rows[parent_id].chance_child_id;
    if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
      return;
    }
  }

  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  const std::optional<uint32_t> child_lookup =
      strategy_betting_history_id(child_node);
  if (child_lookup.has_value()) {
    child_node.betting_history_id = *child_lookup;
  }
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

std::optional<uint32_t> CFRSolver::strategy_public_state_id(
    GameTree::Node& node) {
  if (node.public_state_id != GameTree::Node::kInvalidPublicStateId) {
    return node.public_state_id;
  }

  const std::optional<uint32_t> betting_history_id =
      strategy_betting_history_id(node);
  if (!betting_history_id.has_value()) {
    return std::nullopt;
  }
  auto public_state =
      frozen_tables_->public_state_ids.find(
          make_public_state_key(*betting_history_id, node.state));
  if (public_state == frozen_tables_->public_state_ids.end()) {
    return std::nullopt;
  }

  node.public_state_id = public_state->second;
  return node.public_state_id;
}

std::optional<uint32_t> CFRSolver::strategy_betting_history_id(
    const GameState& state) const {
  const auto& betting_history_ids = frozen_tables_->betting_history_ids;
  auto betting_history =
      betting_history_ids.find(make_betting_history_key(state));
  if (betting_history == betting_history_ids.end()) {
    return std::nullopt;
  }
  return betting_history->second;
}

std::optional<uint32_t> CFRSolver::strategy_betting_history_id(
    GameTree::Node& node) {
  if (node.betting_history_id !=
      GameTree::Node::kInvalidBettingHistoryId) {
    return node.betting_history_id;
  }

  std::optional<uint32_t> betting_history_id =
      strategy_betting_history_id(node.state);
  if (betting_history_id.has_value()) {
    node.betting_history_id = *betting_history_id;
  }
  return betting_history_id;
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
  traversal_stats_.betting_history_transition_hits +=
      stats.betting_history_transition_hits;
  traversal_stats_.betting_history_transition_misses +=
      stats.betting_history_transition_misses;
}

void CFRSolver::run(int iterations, ComboId player_a_hand,
                    ComboId player_b_hand) {
  if (iterations <= 0) {
    return;
  }

  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return;
  }
  const int max_depth = config_.max_depth;
  TraversalScratch scratch;
  scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  for (int i = 0; i < iterations; ++i) {
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    cumulative_root_utility_ += cfr_with_ranges(
        *root_public_state_id, player_a_cards, player_b_cards, reach_probabilities,
        iterations_run_.load(std::memory_order_relaxed), 0, max_depth, scratch,
        std::nullopt, std::nullopt);
    iterations_run_.fetch_add(1, std::memory_order_relaxed);
  }
}

void CFRSolver::run(int iterations, const HandRange& player_a_range,
                    const HandRange& player_b_range) {
  run_iterations(iterations, player_a_range, player_b_range);
}

void CFRSolver::run_iterations(int iterations,
                               const HandRange& player_a_range,
                               const HandRange& player_b_range) {
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

  // Determine single-threaded warmup count.
  // Warmup populates the game tree and info set table before we freeze.
  // - If num_threads == 1 there is no parallel phase; warmup = all iterations.
  // - If warmup_iterations is set explicitly, honour it (capped to iterations).
  // - Otherwise: run until max_info_sets is reached OR info set growth stalls,
  //   using at most half the requested iterations as a ceiling.
  int warmup_count = iterations;
  if (num_threads > 1 && !frozen_) {
    if (config_.warmup_iterations > 0) {
      warmup_count = std::min(config_.warmup_iterations, iterations);
    } else if (config_.max_info_sets > 0) {
      // Run until the cap is hit; cap the warmup at half the total budget.
      warmup_count = std::min(iterations / 2 + 1, iterations);
    } else {
      // No cap: default warmup = 10% of iterations, minimum 100.
      warmup_count = std::max(100, iterations / 10);
      warmup_count = std::min(warmup_count, iterations);
    }
  }

  // Phase 1: single-threaded warmup (allocates public states + info sets).
  LOG(INFO) << "Starting CFR iterations...";
  VLOG(1) << "Warmup phase: " << warmup_count << " single-threaded iterations";
  const int max_depth = config_.max_depth;
  TraversalScratch scratch;
  scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
  for (int i = 0; i < warmup_count; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    PrivateCards player_a_cards = PrivateCards::FromCombo(deal.player_a_combo);
    PrivateCards player_b_cards = PrivateCards::FromCombo(deal.player_b_combo);

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_.load(std::memory_order_relaxed);
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    double dealt_value = cfr_with_ranges(
        *root_public_state_id, player_a_cards, player_b_cards,
        reach_probabilities,
        cfr_iteration, 0, max_depth, scratch, player_a_context_range,
        player_b_context_range);

    cumulative_root_utility_ += dealt_value;
    iterations_run_.fetch_add(1, std::memory_order_relaxed);
  }

  const int remaining = iterations - warmup_count;
  if (remaining > 0) {
    // Phase 2: freeze the tables and run remaining iterations in parallel.
    frozen_tables_ = mutable_tables_;
    mutable_tables_.reset();
    frozen_ = true;
    LOG(INFO) << "Frozen after warmup: " << get_info_set_count()
              << " info sets, " << iterations_run_.load(std::memory_order_relaxed)
              << " warmup iterations. Starting parallel phase ("
              << remaining << " iterations, " << num_threads << " threads)...";
    run_iterations_parallel(remaining, num_threads, *root_public_state_id,
                            range_sampler,
                            player_a_training_range, player_b_training_range);
  }

  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_.load(std::memory_order_relaxed);
  LOG(INFO) << "Information sets: " << get_info_set_count();
  LOG(INFO) << "Public states: " << get_public_state_count();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

void CFRSolver::run_iterations_parallel(
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

  ThreadPoolExecutor executor(num_threads);
  std::uniform_int_distribution<unsigned int> seed_dist;
  std::vector<std::future<std::pair<double, TraversalStats>>> futures;
  futures.reserve(num_threads);

  int iterations_remaining = iterations;
  for (int t = 0; t < num_threads; ++t) {
    const int shard = iterations_remaining / (num_threads - t);
    iterations_remaining -= shard;
    if (shard <= 0) {
      continue;
    }

    const unsigned int seed = seed_dist(rng_);
    futures.push_back(executor.submit(
        [this, shard, seed, root_public_state_id, &range_sampler,
         frozen_tables, cumulative,
         &player_a_training_range, &player_b_training_range]() mutable {
          // Build a lightweight worker that shares frozen tables.
          CFRSolver worker(config_, utility_cache_,
                           continuation_value_provider_);
          worker.mutable_tables_.reset();
          worker.frozen_tables_ = frozen_tables;
          worker.cumulative_ = cumulative;
          worker.frozen_ = true;
          worker.rng_.seed(seed);
          // Per-worker range views (read-only, built from shared training data).
          TrainingRangeView player_a_hands_view;
          TrainingRangeView player_b_hands_view;
          player_a_hands_view.reset_to_all(player_a_training_range);
          player_b_hands_view.reset_to_all(player_b_training_range);

          const int max_depth = config_.max_depth;
          TraversalScratch scratch;
          scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));

          double local_utility = 0.0;
          for (int i = 0; i < shard; ++i) {
            const RangeDeal deal = range_sampler.sample(worker.rng_);
            PrivateCards player_a_cards =
                PrivateCards::FromCombo(deal.player_a_combo);
            PrivateCards player_b_cards =
                PrivateCards::FromCombo(deal.player_b_combo);

            // Use the global iteration counter for CFR+ linear weighting.
            const int cfr_iteration =
                iterations_run_.fetch_add(1, std::memory_order_relaxed);
            std::array<double, 2> reach_probabilities = {1.0, 1.0};
            OptionalTrainingRange player_a_context_range;
            OptionalTrainingRange player_b_context_range;
            if (max_depth > 0) {
              player_a_context_range = std::cref(player_a_hands_view);
              player_b_context_range = std::cref(player_b_hands_view);
            }
            local_utility += worker.cfr_with_ranges(
                root_public_state_id, player_a_cards, player_b_cards,
                reach_probabilities, cfr_iteration, 0, max_depth, scratch,
                player_a_context_range, player_b_context_range);
          }
          return std::make_pair(local_utility, worker.get_traversal_stats());
        }));
  }

  // Accumulate per-worker root utilities and traversal stats into the main solver.
  for (std::future<std::pair<double, TraversalStats>>& f : futures) {
    const auto result = f.get();
    cumulative_root_utility_ += result.first;
    add_traversal_stats(result.second);
  }
}

double CFRSolver::cfr_with_ranges(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
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
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.terminal_utility_calls);
    if (row.state.folded_player >= 0) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.fold_utility_calls);
    } else {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.showdown_utility_calls);
    }
    if (max_depth > 0) {
      return uncached_utility(row.state, player_a_cards, player_b_cards);
    }
    return utility(row.state, player_a_cards, player_b_cards);
  }

  if (row.is_chance_node) {
    return chance_sampling_cfr(public_state_id, player_a_cards, player_b_cards,
                               reach_probabilities, iteration, depth,
                               max_depth, scratch, player_a_range,
                               player_b_range);
  }

  if (max_depth > 0 && depth >= max_depth) {
    // Continuation providers still consume GameState. This cutoff path is
    // intentionally left materialized until continuation contexts move compact.
    const GameState state = materialize_game_state(row.state);
    ContinuationContext context = build_continuation_context(
        state, player_a_cards.combo, player_b_cards.combo, player_a_range,
        player_b_range);
    return continuation_value_provider_->value(*game_tree_, context);
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
  const InfoSetRow* info_set_row =
      get_or_create_info_set_row(info_set_address,
                                 absl::Span<const int>(row.action_ids.data(),
                                                       row.action_count));

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
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
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
    condition_ranges_for_actions(player_a_range->get(), row.state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  } else if (condition_player_b_range) {
    condition_ranges_for_actions(player_b_range->get(), row.state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    std::optional<uint32_t> child_public_state_id =
        get_or_create_action_child_public_state(
            public_state_id, static_cast<int>(action_index));
    if (!child_public_state_id.has_value()) {
      continue;
    }

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
        *child_public_state_id, player_a_cards, player_b_cards,
        reach_probabilities, iteration, depth + 1, max_depth, scratch,
        child_player_a_range, child_player_b_range);
    action_values[action_index] = action_value;
    reach_probabilities[player] = previous_reach_probability;
    node_value += action_probabilities[action_index] * action_value;
  }

  cfr_update_count_.fetch_add(1, std::memory_order_relaxed);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.cfr_updates);
  POKER_RECORD_TRAVERSAL_STAT(
      traversal_stats_.max_decision_depth =
          std::max(traversal_stats_.max_decision_depth, depth));
  switch (street) {
    case StreetKind::kPreflop:
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.preflop_updates);
      break;
    case StreetKind::kFlop:
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.flop_updates);
      break;
    case StreetKind::kTurn:
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.turn_updates);
      break;
    case StreetKind::kRiver:
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.river_updates);
      break;
  }

  if (has_info_set_row) {
    const double opponent_reach_prob = reach_probabilities[1 - player];
    auto& regrets = cumulative_->cumulative_regrets;
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
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

double CFRSolver::chance_sampling_cfr(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
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
  // Child creation may grow public_state_rows, so keep a local copy instead
  // of holding a reference into the vector across samples.
  const CompactPublicState state = rows[public_state_id].state;

  const int samples = ChanceSamples(config_);
  POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.chance_samples += samples);
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  TrainingRangeView& public_player_a_range =
      scratch_frame.public_player_a_range;
  TrainingRangeView& public_player_b_range =
      scratch_frame.public_player_b_range;

  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const auto cards = SampleStreetCards(
        state, player_a_cards.mask() | player_b_cards.mask(), rng_);
    std::optional<uint32_t> child_public_state_id =
        get_or_create_chance_child_public_state(public_state_id, cards);
    if (!child_public_state_id.has_value()) {
      continue;
    }

    const PublicStateRow& child_row =
        frozen_tables_->public_state_rows[*child_public_state_id];
    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (player_a_range.has_value()) {
      PublicCompatibleRangeInto(
          player_a_range->get(), child_row.state.board_mask,
          public_player_a_range);
      child_player_a_range = std::cref(public_player_a_range);
    }
    if (player_b_range.has_value()) {
      PublicCompatibleRangeInto(
          player_b_range->get(), child_row.state.board_mask,
          public_player_b_range);
      child_player_b_range = std::cref(public_player_b_range);
    }

    value += cfr_with_ranges(*child_public_state_id, player_a_cards,
                             player_b_cards, reach_probabilities, iteration,
                             depth, max_depth, scratch, child_player_a_range,
                             child_player_b_range);
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
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
  const std::vector<float>& cumulative_regrets =
      cumulative_->cumulative_regrets;
  const std::vector<float>& cumulative_strategies =
      cumulative_->cumulative_strategies;

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
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
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
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
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

        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
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

ContinuationContext CFRSolver::build_continuation_context(
    const GameState& state,
    ComboId player_a_hand,
    ComboId player_b_hand,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) const {
  ContinuationContext context =
      ContinuationContext::ExactHands(state, player_a_hand, player_b_hand);
  if (player_a_range.has_value() && player_b_range.has_value()) {
    PublicCompatibleRangeInto(
        player_a_range->get(), state, context.player_a_range);
    PublicCompatibleRangeInto(
        player_b_range->get(), state, context.player_b_range);
  }
  return context;
}

namespace {

double StrategyWeight(const SolverConfig& config,
                      const std::vector<float>& cumulative_regrets,
                      const std::vector<float>& cumulative_strategies,
                      size_t table_index) {
  return config.regret_only_training
             ? std::max(0.0,
                        static_cast<double>(
                            AtomicFloatLoad(&cumulative_regrets[table_index])))
             : static_cast<double>(
                   AtomicFloatLoad(&cumulative_strategies[table_index]));
}

}  // namespace

CFRSolver::StrategyProfile CFRSolver::get_strategy_profile() const {
  const auto& slabs = frozen_tables_->public_info_set_slabs;
  const auto& action_ids = frozen_tables_->action_ids;
  const auto& regrets = cumulative_->cumulative_regrets;
  const auto& strategies = cumulative_->cumulative_strategies;

  StrategyProfile profile;
  profile.info_sets.reserve(get_info_set_count());

  auto export_row = [&](uint32_t public_state_id,
                        uint8_t player,
                        PrivateBucketId private_bucket,
                        const InfoSetRow& row) {
    double sum = 0.0;
    const size_t action_offset = row.action_offset;
    for (uint16_t action_index = 0; action_index < row.action_count;
         ++action_index) {
      const size_t table_index = action_offset + action_index;
      sum += StrategyWeight(config_, regrets, strategies, table_index);
    }

    StrategyInfoSet exported;
    exported.key.public_state_id = public_state_id;
    exported.key.private_bucket = private_bucket;
    exported.key.player = player;
    exported.action_ids.reserve(row.action_count);
    exported.probabilities.reserve(row.action_count);
    if (sum > 0.0) {
      for (uint16_t action_index = 0; action_index < row.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        exported.action_ids.push_back(action_ids[table_index]);
        exported.probabilities.push_back(
            StrategyWeight(config_, regrets, strategies, table_index) / sum);
      }
    } else if (row.action_count > 0) {
      double uniform_prob = 1.0 / row.action_count;
      for (uint16_t action_index = 0; action_index < row.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        exported.action_ids.push_back(action_ids[table_index]);
        exported.probabilities.push_back(uniform_prob);
      }
    }

    profile.info_sets.push_back(std::move(exported));
  };

  for (uint32_t public_state_id = 0; public_state_id < slabs.size();
       ++public_state_id) {
    const std::unique_ptr<PublicInfoSetSlab>& slab = slabs[public_state_id];
    if (slab == nullptr) {
      continue;
    }
    for (uint8_t player = 0; player < kPlayerCount; ++player) {
      const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
      for (size_t chunk_index = 0;
           chunk_index < player_slab.private_row_chunks.size();
           ++chunk_index) {
        const std::unique_ptr<PrivateRowChunk>& chunk =
            player_slab.private_row_chunks[chunk_index];
        if (chunk == nullptr) {
          continue;
        }
        for (size_t chunk_offset = 0; chunk_offset < chunk->rows.size();
             ++chunk_offset) {
          const PrivateBucketId private_bucket = static_cast<PrivateBucketId>(
              chunk_index * kPrivateBucketChunkSize + chunk_offset);
          if (private_bucket >= kComboCount) {
            break;
          }
          const int32_t row_id = chunk->rows[chunk_offset];
          if (row_id < 0 ||
              static_cast<size_t>(row_id) >= player_slab.rows.size()) {
            continue;
          }
          export_row(public_state_id, player, private_bucket,
                     player_slab.rows[row_id]);
        }
      }
    }
  }

  return profile;
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }
  return evaluate_strategy_node(*root_public_state_id,
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
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
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
                                       continuation_value_provider,
                                       shard_samples, seed]() mutable {
      CFRSolver worker(config, std::make_shared<TerminalUtilityCache>(),
                       continuation_value_provider);
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
    POKER_RECORD_TRAVERSAL_STAT(
        traversal_stats_.action_entry_touches += result.second);
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
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    total += evaluate_strategy_node(
        root_public_state_id, PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo));
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards) {
  const auto& rows = frozen_tables_->public_state_rows;
  if (public_state_id >= rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = rows[public_state_id];

  if (row.is_terminal) {
    return utility(row.state, player_a_cards, player_b_cards);
  }
  if (row.is_chance_node) {
    const CompactPublicState state = row.state;
    const int samples = ChanceSamples(config_);
    double value = 0.0;
    int evaluated = 0;
    for (int i = 0; i < samples; ++i) {
      const auto cards = SampleStreetCards(
          state, player_a_cards.mask() | player_b_cards.mask(), rng_);
      std::optional<uint32_t> child_public_state_id =
          get_or_create_chance_child_public_state(public_state_id, cards);
      if (!child_public_state_id.has_value()) {
        continue;
      }
      value += evaluate_strategy_node(
          *child_public_state_id, player_a_cards, player_b_cards);
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
    std::optional<uint32_t> child_public_state_id =
        get_or_create_action_child_public_state(public_state_id, action_index);
    if (!child_public_state_id.has_value()) {
      continue;
    }
    value += probabilities[action_index] *
             evaluate_strategy_node(*child_public_state_id, player_a_cards,
                                    player_b_cards);
  }
  return value;
}

double CFRSolver::calculate_exploitability() {
  return BestResponseEvaluator(*this).calculate_exploitability();
}

double CFRSolver::calculate_exploitability(int samples) {
  return BestResponseEvaluator(*this).calculate_exploitability(samples);
}

double CFRSolver::calculate_exploitability(int samples,
                                           const HandRange& player_a_range,
                                           const HandRange& player_b_range) {
  return BestResponseEvaluator(*this).calculate_exploitability(
      samples, player_a_range, player_b_range);
}

double CFRSolver::calculate_player_a_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return BestResponseEvaluator(*this).calculate_player_a_best_response_value(
      samples, player_a_range, player_b_range);
}

double CFRSolver::calculate_player_b_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return BestResponseEvaluator(*this).calculate_player_b_best_response_value(
      samples, player_a_range, player_b_range);
}

double CFRSolver::calculate_exploitability(ComboId player_a_hand,
                                           ComboId player_b_hand) {
  return BestResponseEvaluator(*this).calculate_exploitability(
      player_a_hand, player_b_hand);
}

GameAction CFRSolver::get_best_response_action(GameTree::Node& node,
                                               ComboId player_a_hand,
                                               ComboId player_b_hand,
                                               int best_response_player) {
  return BestResponseEvaluator(*this).get_best_response_action(
      node, player_a_hand, player_b_hand, best_response_player);
}

double CFRSolver::get_expected_value(int player_id) const {
  const int iters = iterations_run_.load(std::memory_order_relaxed);
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

double CFRSolver::utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_cards.size() + 2 < 5) {
    return 0.0;
  }

  return utility_cache_->get_or_compute(
      state, player_a_cards.combo, player_b_cards.combo, [&]() {
        return game_tree_->get_utility(
            state, player_a_cards.combo, player_b_cards.combo);
      });
}

double CFRSolver::utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_count + 2 < 5) {
    return 0.0;
  }

  return utility_cache_->get_or_compute(
      state, player_a_cards.combo, player_b_cards.combo, [&]() {
        return game_tree_->get_utility(
            state, player_a_cards.combo, player_b_cards.combo);
      });
}

double CFRSolver::uncached_utility(const GameState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_cards.size() + 2 < 5) {
    return 0.0;
  }

  return game_tree_->get_utility(
      state, player_a_cards.combo, player_b_cards.combo);
}

double CFRSolver::uncached_utility(const CompactPublicState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_count + 2 < 5) {
    return 0.0;
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
    POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
    const float delta = static_cast<float>(
        reach_prob * action_probabilities[action_index]);
    AtomicFloatAdd(&strategies[action_offset + action_index], delta);
  }
}

} // namespace poker
