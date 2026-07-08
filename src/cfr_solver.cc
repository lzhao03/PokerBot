#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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
#include <random>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kAutoWarmupNoGrowthLimit = 100;

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

class PublicStateBfs {
 public:
  struct Entry {
    uint32_t public_state_id = 0;
    int depth = 0;
  };

  PublicStateBfs(uint32_t root_public_state_id, size_t row_count) {
    queue_.reserve(1024);
    queued_.resize(row_count, 0);
    enqueue_growing(root_public_state_id, 0);
  }

  std::optional<Entry> next() {
    if (cursor_ >= queue_.size()) {
      return std::nullopt;
    }
    return queue_[cursor_++];
  }

  bool enqueue_existing(uint32_t public_state_id,
                        int depth,
                        size_t row_count) {
    if (public_state_id >= row_count ||
        public_state_id >= queued_.size()) {
      return false;
    }
    enqueue_known_index(public_state_id, depth);
    return true;
  }

  void enqueue_growing(uint32_t public_state_id, int depth) {
    if (public_state_id >= queued_.size()) {
      queued_.resize(static_cast<size_t>(public_state_id) + 1, 0);
    }
    enqueue_known_index(public_state_id, depth);
  }

 private:
  void enqueue_known_index(uint32_t public_state_id, int depth) {
    if (queued_[public_state_id]) {
      return;
    }
    queued_[public_state_id] = 1;
    queue_.push_back({public_state_id, depth});
  }

  std::vector<Entry> queue_;
  std::vector<char> queued_;
  size_t cursor_ = 0;
};

template <typename Callback>
bool ForEachCardCombination(int count, CardMask blocked_mask,
                            Callback callback) {
  absl::InlinedVector<CardId, 5> cards;
  cards.resize(static_cast<size_t>(count));
  auto choose = [&](auto& self, int start, int depth) -> bool {
    if (depth == count) {
      return callback(absl::Span<const CardId>(cards));
    }
    const int remaining = count - depth;
    for (int card_id = start; card_id <= kDeckCardCount - remaining;
         ++card_id) {
      const CardId card = static_cast<CardId>(card_id);
      if ((blocked_mask & CardBit(card)) != 0) {
        continue;
      }
      cards[static_cast<size_t>(depth)] = card;
      if (!self(self, card_id + 1, depth + 1)) {
        return false;
      }
    }
    return true;
  };
  return choose(choose, 0, 0);
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

  const int available_cards =
      kDeckCardCount -
      static_cast<int>(__builtin_popcountll(state.board_mask));
  if (available_cards < count) {
    throw std::runtime_error("Not enough cards to enumerate next street");
  }
  return ForEachCardCombination(count, state.board_mask, callback);
}

struct CoarseChanceTransitionTemplate {
  CompactPublicState parent_board_state;
  absl::InlinedVector<CardId, 5> cards;
};

using CoarseChanceTransitionMap =
    absl::flat_hash_map<PublicBucketId,
                        std::vector<CoarseChanceTransitionTemplate>>;

CoarseChanceTransitionMap BuildCoarseChanceTransitions(StreetKind street) {
  CoarseChanceTransitionMap transitions;
  const int board_count = BoardCardsForStreet(street);
  const int deal_count = CardsForNextStreet(street);
  if (deal_count <= 0) {
    return transitions;
  }

  CardAbstraction abstraction;
  absl::flat_hash_set<uint64_t> seen;
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
      if (!seen.insert(seen_key).second) {
        return true;
      }
      CoarseChanceTransitionTemplate transition;
      transition.parent_board_state = parent;
      transition.cards.assign(cards.begin(), cards.end());
      transitions[parent_bucket].push_back(std::move(transition));
      return true;
    });
    return true;
  });
  return transitions;
}

[[maybe_unused]] const CoarseChanceTransitionMap& CoarseChanceTransitions(
    StreetKind street) {
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
    storage_(),
    strategy_store_(config_, card_abstraction_, storage_, &traversal_stats_) {
  // Pre-allocate strategy table storage when limits are known upfront.
  // This gives fully deterministic peak memory: no reallocation after init.
  if (config_.max_info_sets > 0) {
    // avg ~4 actions per info set (between 2 and kMaxActionsPerNode).
    constexpr int kAvgActionsPerInfoSet = 4;
    const size_t info_set_cap = static_cast<size_t>(config_.max_info_sets);
    const size_t action_cap = info_set_cap * kAvgActionsPerInfoSet;
    storage_.mutable_ref().action_ids.reserve(action_cap);
    storage_.cumulative_ref().cumulative_regrets.reserve(action_cap);
    storage_.cumulative_ref().cumulative_strategies.reserve(action_cap);
  }
  if (config_.max_public_states > 0) {
    const size_t public_state_cap =
        static_cast<size_t>(config_.max_public_states);
    storage_.mutable_ref().public_state_ids.reserve(public_state_cap);
    storage_.mutable_ref().public_state_rows.reserve(public_state_cap);
    storage_.mutable_ref().public_chance_child_ids.reserve(public_state_cap);
    storage_.mutable_ref().chance_child_entries.reserve(public_state_cap);
    storage_.mutable_ref().private_bucket_rows.reserve(public_state_cap);
    storage_.mutable_ref().frozen_info_set_action_offsets.reserve(
        public_state_cap);
    storage_.mutable_ref().public_info_set_slabs.reserve(public_state_cap);
    storage_.mutable_ref().betting_history_ids.reserve(public_state_cap);
    storage_.mutable_ref().betting_history_rows.reserve(public_state_cap);
  }
}

FrozenStrategyTables& CFRSolver::mutable_tables() {
  return strategy_store_.mutable_tables();
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

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
  if (storage_.frozen) {
    auto existing = storage_.frozen_ref().public_state_ids.find(
        make_public_state_key(betting_history_id, state));
    if (existing == storage_.frozen_ref().public_state_ids.end()) {
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
  if (storage_.frozen) {
    BettingHistoryKey key = make_betting_history_key(state);
    const auto betting_history =
        storage_.frozen_ref().betting_history_ids.find(key);
    if (betting_history == storage_.frozen_ref().betting_history_ids.end()) {
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
  const auto& rows = storage_.frozen_ref().public_state_rows;
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
  const auto& rows = storage_.frozen_ref().public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = rows[public_state_id];
  const size_t begin = row.chance_child_offset;
  const size_t end = begin + row.chance_child_count;
  const auto& entries = storage_.frozen_ref().chance_child_entries;
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

template <typename Callback>
bool CFRSolver::for_each_required_chance_transition(
    const PublicStateRow& row,
    Callback&& callback) const {
  if constexpr (kCoarsePublicBuckets) {
    const auto& transitions = CoarseChanceTransitions(row.state.street);
    const auto existing = transitions.find(row.public_bucket);
    if (existing == transitions.end()) {
      return false;
    }
    for (const CoarseChanceTransitionTemplate& transition : existing->second) {
      CompactPublicState parent_state = row.state;
      CopyBoardFrom(parent_state, transition.parent_board_state);
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
  const auto& rows = storage_.frozen_ref().public_state_rows;
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
  const auto& betting_history_rows = storage_.frozen_ref().betting_history_rows;
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
            storage_.frozen_ref().public_state_ids.find(child_key);
        if (existing_public_child !=
            storage_.frozen_ref().public_state_ids.end()) {
          if (!storage_.frozen) {
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
  if (storage_.frozen) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(storage_.frozen_ref().public_state_rows.size()) >=
          config_.max_public_states) {
    mutable_tables()
        .public_state_rows[public_state_id]
        .action_child_ids[action_slot] = kCappedPublicStateId;
    record_betting_history_transition_miss();
    return std::nullopt;
  }

  const GameAction action =
      storage_.frozen_ref()
          .public_state_rows[public_state_id]
          .actions[action_slot];
  CompactPublicState child_state = game_tree_->apply_action(
      storage_.frozen_ref().public_state_rows[public_state_id].state,
      action);
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
  const auto& rows = storage_.frozen_ref().public_state_rows;
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = rows[public_state_id];
  const PublicBucketId outcome_id = chance_outcome_id(child_state);
  const ChanceTransitionKey transition_key{public_state_id, outcome_id};
  const auto& chance_children = storage_.frozen_ref().public_chance_child_ids;
  auto existing = chance_children.find(transition_key);
  if (existing != chance_children.end()) {
    record_betting_history_transition_hit();
    return existing->second;
  }
  if constexpr (kCoarsePublicBuckets) {
    const uint32_t cached_parent_betting_history_id = row.betting_history_id;
    const auto& betting_history_rows = storage_.frozen_ref().betting_history_rows;
    if (cached_parent_betting_history_id < betting_history_rows.size()) {
      const uint32_t child_betting_history_id =
          betting_history_rows[cached_parent_betting_history_id].chance_child_id;
      if (child_betting_history_id != GameTree::kInvalidBettingHistoryId) {
        const PublicStateKey child_public_key{
            child_betting_history_id,
            outcome_id,
        };
        auto existing_public_child =
            storage_.frozen_ref().public_state_ids.find(child_public_key);
        if (existing_public_child !=
            storage_.frozen_ref().public_state_ids.end()) {
          if (!storage_.frozen) {
            mutable_tables().public_chance_child_ids.emplace(
                transition_key, existing_public_child->second);
          }
          record_betting_history_transition_hit();
          return existing_public_child->second;
        }
      }
    }
  }
  if (storage_.frozen) {
    record_betting_history_transition_miss();
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(storage_.frozen_ref().public_state_rows.size()) >=
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

CFRSolver::ExactBoardState CFRSolver::ExactBoardFromState(
    const CompactPublicState& state) {
  return ExactBoardState{
      state.board_cards,
      state.board_count,
      state.board_mask,
  };
}

void CFRSolver::ApplyExactBoard(CompactPublicState& state,
                                const ExactBoardState& board) {
  state.board_cards = board.cards;
  state.board_count = board.count;
  state.board_mask = board.mask;
}

const CompactPublicState& CFRSolver::NodeCursor::exact_state() const {
  if (!exact_state_.has_value()) {
    CompactPublicState state = row_.state;
    CFRSolver::ApplyExactBoard(state, ref_.exact_board);
    exact_state_.emplace(std::move(state));
  }
  return *exact_state_;
}

std::optional<CFRSolver::NodeCursor> CFRSolver::cursor(NodeRef node) const {
  const auto& rows = storage_.frozen_ref().public_state_rows;
  if (node.public_state_id >= rows.size()) {
    return std::nullopt;
  }
  return NodeCursor{node, rows[node.public_state_id]};
}

std::optional<CFRSolver::NodeRef> CFRSolver::root_node_ref(
    uint32_t root_public_state_id) const {
  const auto& rows = storage_.frozen_ref().public_state_rows;
  if (root_public_state_id >= rows.size()) {
    return std::nullopt;
  }
  return NodeRef{root_public_state_id,
                 ExactBoardFromState(rows[root_public_state_id].state)};
}

CFRSolver::NodeGraphMode CFRSolver::default_node_graph_mode() const {
  if (!storage_.frozen) {
    return NodeGraphMode::kGrow;
  }
  return require_frozen_children_ ? NodeGraphMode::kRequirePresent
                                  : NodeGraphMode::kSkipMissing;
}

CFRSolver::NodeGraph::NodeGraph(CFRSolver& solver,
                                NodeGraphMode mode)
    : solver_(solver), mode_(mode) {}

CFRSolver::ChildResult
CFRSolver::NodeGraph::make_child_result(
    std::optional<uint32_t> child_id,
    ExactBoardState exact_board,
    const char* missing_message) const {
  const auto& rows = solver_.storage_.frozen_ref().public_state_rows;
  ChildStatus status = ChildStatus::kOk;
  if (!child_id.has_value() ||
      *child_id == GameTree::kInvalidPublicStateId) {
    status = ChildStatus::kMissing;
  } else if (*child_id == kCappedPublicStateId) {
    status = ChildStatus::kCapped;
  } else if (*child_id >= rows.size()) {
    status = ChildStatus::kInvalid;
  }

  if (status != ChildStatus::kOk) {
    if (mode_ == NodeGraphMode::kRequirePresent) {
      throw std::logic_error(missing_message);
    }
    return ChildResult{status, {}};
  }

  return ChildResult{ChildStatus::kOk, NodeRef{*child_id, exact_board}};
}

CFRSolver::ChildResult
CFRSolver::NodeGraph::action_child(NodeRef parent,
                                   int action_index) {
  std::optional<uint32_t> child_id;
  switch (mode_) {
    case NodeGraphMode::kGrow:
      child_id = solver_.get_or_create_action_child_public_state(
          parent.public_state_id, action_index);
      break;
    case NodeGraphMode::kSkipMissing:
    case NodeGraphMode::kRequirePresent:
      child_id =
          solver_.action_child_public_state(parent.public_state_id,
                                            action_index);
      break;
  }

  return make_child_result(child_id, parent.exact_board,
                           "required action child public state is missing");
}

CFRSolver::ChildResult
CFRSolver::NodeGraph::sample_chance_child(
    NodeRef parent,
    CardMask known_private_cards) {
  std::optional<NodeCursor> parent_cursor = solver_.cursor(parent);
  if (!parent_cursor.has_value()) {
    return ChildResult{ChildStatus::kInvalid, {}};
  }
  const CompactPublicState& exact_parent_state = parent_cursor->exact_state();
  const auto cards =
      SampleStreetCards(exact_parent_state, known_private_cards, solver_.rng_);
  CompactPublicState exact_child_state =
      solver_.game_tree_->apply_chance(exact_parent_state, cards);
  const ExactBoardState child_board =
      CFRSolver::ExactBoardFromState(exact_child_state);
  std::optional<uint32_t> child_id;
  switch (mode_) {
    case NodeGraphMode::kGrow:
      child_id = solver_.get_or_create_chance_child_public_state(
          parent.public_state_id, exact_child_state);
      break;
    case NodeGraphMode::kSkipMissing:
    case NodeGraphMode::kRequirePresent:
      child_id =
          solver_.chance_child_public_state(parent.public_state_id,
                                            exact_child_state);
      break;
  }

  return make_child_result(child_id, child_board,
                           "required chance child public state is missing");
}

bool CFRSolver::prebuild_public_state_rows(uint32_t root_public_state_id,
                                           int max_depth) {
  if (storage_.frozen) {
    return true;
  }
  if (root_public_state_id >= storage_.frozen_ref().public_state_rows.size()) {
    return false;
  }

  PublicStateBfs bfs(root_public_state_id,
                     storage_.frozen_ref().public_state_rows.size());
  while (std::optional<PublicStateBfs::Entry> maybe_entry = bfs.next()) {
    const PublicStateBfs::Entry entry = *maybe_entry;
    if (entry.public_state_id >= storage_.frozen_ref().public_state_rows.size()) {
      return false;
    }
    const PublicStateRow row =
        storage_.frozen_ref().public_state_rows[entry.public_state_id];
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
            bfs.enqueue_growing(*child_public_state_id, entry.depth);
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
      bfs.enqueue_growing(*child_public_state_id, entry.depth + 1);
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
  const auto& rows = storage_.frozen_ref().public_state_rows;
  const auto& history_rows = storage_.frozen_ref().betting_history_rows;
  if (root_public_state_id >= rows.size()) {
    return false;
  }
  stats.betting_history_transition_prebuild_complete = true;
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  PublicStateBfs bfs(root_public_state_id, rows.size());

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

  while (std::optional<PublicStateBfs::Entry> maybe_entry = bfs.next()) {
    const PublicStateBfs::Entry entry = *maybe_entry;
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
        if (valid_child &&
            !bfs.enqueue_existing(
                *child_public_state_id, entry.depth, rows.size())) {
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
          !bfs.enqueue_existing(
              child_public_state_id, entry.depth + 1, rows.size())) {
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
  if (storage_.frozen) {
    return true;
  }

  std::vector<uint32_t> seen_buckets;
  uint32_t seen_generation = 1;

  for (uint32_t public_state_id = 0;
       public_state_id < storage_.frozen_ref().public_state_rows.size();
       ++public_state_id) {
    const PublicStateRow& row =
        storage_.frozen_ref().public_state_rows[public_state_id];
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
      if (!strategy_store_
               .get_or_create({public_state_id, player, private_bucket},
                              action_ids)
               .has_value()) {
        return false;
      }
    }
  }

  return true;
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
                 storage_.frozen_ref().public_state_rows[*root_public_state_id].action_count)
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
      storage_.frozen_ref().public_state_rows[*root_public_state_id].state;
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
  TrainingRunStats& stats = last_training_run_stats_;
  const bool should_prebuild_public_states =
      !storage_.frozen && (num_threads > 1 || can_use_frozen_regret_only) &&
      (config_.max_public_states > 0 || max_depth > 0);
  if (!should_prebuild_public_states) {
    return false;
  }

  auto record_public_counts = [&] {
    stats.prebuild_public_states =
        static_cast<int64_t>(get_public_state_count());
    stats.prebuild_betting_histories =
        static_cast<int64_t>(storage_.frozen_ref().betting_history_rows.size());
  };
  auto record_action_counts = [&] {
    stats.prebuild_info_sets = static_cast<int64_t>(get_info_set_count());
    stats.prebuild_action_entries =
        static_cast<int64_t>(
            storage_.cumulative_ref().cumulative_regrets.size());
  };

  VLOG(1) << "Prebuilding compact public-state rows...";
  const auto prebuild_start = std::chrono::steady_clock::now();
  stats.public_state_prebuild_complete =
      prebuild_public_state_rows(root_public_state_id, max_depth);
  const auto prebuild_end = std::chrono::steady_clock::now();
  stats.prebuild_seconds =
      std::chrono::duration<double>(prebuild_end - prebuild_start).count();
  record_public_counts();
  if (!stats.public_state_prebuild_complete) {
    return false;
  }

  if (!validate_prebuilt_transitions(root_public_state_id, max_depth, stats)) {
    return false;
  }
  record_action_counts();

  VLOG(1) << "Prebuilding infoset rows...";
  const auto info_set_prebuild_start = std::chrono::steady_clock::now();
  stats.info_set_prebuild_complete =
      prebuild_info_set_rows(player_a_hands_view, player_b_hands_view);
  const auto info_set_prebuild_end = std::chrono::steady_clock::now();
  stats.info_set_prebuild_seconds =
      std::chrono::duration<double>(info_set_prebuild_end -
                                    info_set_prebuild_start)
          .count();
  record_action_counts();
  if (!stats.info_set_prebuild_complete) {
    return false;
  }

  stats.private_bucket_prebuild_complete =
      strategy_store_.prebuild_private_bucket_rows();
  if (!stats.private_bucket_prebuild_complete) {
    return false;
  }
  stats.prebuild_private_bucket_rows =
      static_cast<int64_t>(storage_.frozen_ref().private_bucket_rows.size());

  stats.frozen_info_set_lookup_prebuild_complete =
      strategy_store_.prebuild_frozen_info_set_action_offsets();
  if (!stats.frozen_info_set_lookup_prebuild_complete) {
    return false;
  }
  stats.prebuild_frozen_info_set_lookup_rows =
      static_cast<int64_t>(
          storage_.frozen_ref().frozen_info_set_action_offsets.size());
  return true;
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
      should_run_frozen_phase && !storage_.frozen && config_.warmup_iterations <= 0;
  int warmup_count = iterations;
  if (should_run_frozen_phase && !storage_.frozen &&
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
  NodeGraph graph(*this, NodeGraphMode::kGrow);
  const NodeRef root_node{root_public_state_id,
                          ExactBoardFromState(root_state)};
  TraversalScratch scratch(ScratchDepthReserve(config_, max_depth));
  const int64_t warmup_start_updates = cfr_update_count_;
  const auto warmup_start = std::chrono::steady_clock::now();
  int completed_warmup = 0;
  int no_growth_iterations = 0;
  size_t previous_info_sets = get_info_set_count();
  size_t previous_public_states = get_public_state_count();

  for (int i = 0; i < warmup_count; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    const TraversalDeal traversal_deal{{
        PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo),
    }};

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_;
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    TraversalOptions options;
    options.update_player = cfr_iteration % kPlayerCount;
    options.iteration = cfr_iteration;
    options.max_depth = max_depth;
    options.write_average_strategy = !config_.regret_only_training;
    TraversalContext ctx(traversal_deal, options, scratch,
                         player_a_context_range, player_b_context_range);
    double dealt_value = cfr_with_ranges(root_node, ctx, graph);

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

  storage_.freeze();
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
  std::shared_ptr<const FrozenStrategyTables> frozen_tables =
      storage_.frozen_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative = storage_.cumulative;
  const bool use_frozen_regret_only =
      storage_.frozen && require_frozen_children_ &&
      config_.regret_only_training && config_.max_depth == 0;
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
          worker.storage_.bind_frozen(frozen_tables, cumulative);
          worker.require_frozen_children_ = true;
          worker.rng_.seed(seed);

          TrainingRangeView player_a_hands_view;
          TrainingRangeView player_b_hands_view;
          const int max_depth = config_.max_depth;
          const size_t scratch_depth =
              use_frozen_regret_only ? 0
                                     : ScratchDepthReserve(config_, max_depth);
          TraversalScratch scratch(scratch_depth);
          if (!use_frozen_regret_only) {
            // Per-worker range views (read-only, built from shared training data).
            player_a_hands_view.reset_to_all(player_a_training_range);
            player_b_hands_view.reset_to_all(player_b_training_range);
          }

          double local_utility = 0.0;
          const CompactPublicState root_state =
              worker.storage_.frozen_ref()
                  .public_state_rows[root_public_state_id]
                  .state;
          NodeGraph graph(worker, NodeGraphMode::kRequirePresent);
          const NodeRef root_node{root_public_state_id,
                                  ExactBoardFromState(root_state)};
          for (int i = 0; i < shard; ++i) {
            const RangeDeal deal = range_sampler.sample(worker.rng_);
            const TraversalDeal traversal_deal{{
                PrivateCards::FromCombo(deal.player_a_combo),
                PrivateCards::FromCombo(deal.player_b_combo),
            }};

            const int cfr_iteration = iteration_begin + i;
            OptionalTrainingRange player_a_context_range;
            OptionalTrainingRange player_b_context_range;
            if (max_depth > 0) {
              player_a_context_range = std::cref(player_a_hands_view);
              player_b_context_range = std::cref(player_b_hands_view);
            }
            TraversalOptions options;
            options.update_player = cfr_iteration % kPlayerCount;
            options.iteration = cfr_iteration;
            options.max_depth = max_depth;
            options.write_average_strategy = !config_.regret_only_training;
            if (use_frozen_regret_only) {
              options.regret_load_mode =
                  use_atomic_updates ? RegretLoadMode::kAtomic
                                     : RegretLoadMode::kPlain;
              options.regret_update_mode =
                  use_atomic_updates ? RegretUpdateMode::kAtomic
                                     : RegretUpdateMode::kPlain;
              options.write_average_strategy = false;
              options.record_atomic_retry_stats = use_atomic_updates;
            }
            TraversalContext ctx(traversal_deal, options, scratch,
                                 player_a_context_range,
                                 player_b_context_range);
            if (use_frozen_regret_only) {
              local_utility +=
                  worker.cfr_frozen_regret_only(root_node, ctx, graph);
              continue;
            }

            local_utility += worker.cfr_with_ranges(root_node, ctx, graph);
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

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::cfr_traversal(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  const uint32_t public_state_id = node.public_state_id;
  std::optional<NodeCursor> node_cursor;
  const PublicStateRow* row_ptr = nullptr;
  if constexpr (mode == CfrTraversalMode::kNormal) {
    node_cursor = cursor(node);
    if (!node_cursor.has_value()) {
      return 0.0;
    }
    row_ptr = &node_cursor->row();
  } else {
    const auto& public_state_rows = storage_.frozen_ref().public_state_rows;
    if (public_state_id >= public_state_rows.size()) {
      return 0.0;
    }
    row_ptr = &public_state_rows[public_state_id];
  }
  const PublicStateRow& row = *row_ptr;

  if (row.is_terminal) {
    if constexpr (mode == CfrTraversalMode::kNormal) {
      const CompactPublicState& state = node_cursor->exact_state();
      record_terminal_utility(state.folded_player < 0);
      if (!ctx.use_terminal_cache() || ctx.max_depth() > 0) {
        return uncached_utility(state, ctx.cards(0), ctx.cards(1));
      }
      return utility(state, ctx.cards(0), ctx.cards(1));
    } else {
      record_terminal_utility(row.state.folded_player < 0);
      return frozen_utility(row, node.exact_board, ctx.cards(0),
                            ctx.cards(1));
    }
  }

  if (row.is_chance_node) {
    if constexpr (mode == CfrTraversalMode::kNormal) {
      return chance_sampling_cfr(node, ctx, graph);
    } else {
      return chance_sampling_frozen_regret_only(node, ctx, graph);
    }
  }

  if constexpr (mode == CfrTraversalMode::kNormal) {
    if (ctx.depth_limited()) {
      const CompactPublicState& state = node_cursor->exact_state();
      return game_tree_->is_betting_round_over(state)
                 ? uncached_utility(state, ctx.cards(0), ctx.cards(1))
                 : 0.0;
    }
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player) || row.action_count == 0) {
    return 0.0;
  }
  const StreetKind street = row.state.street;
  const PrivateCards& player_cards = ctx.cards(player);

  const bool is_update_player = ctx.is_update_player(player);
  const size_t action_count = row.action_count;
  const absl::Span<const int> legal_action_ids(row.action_ids.data(),
                                               action_count);
  std::optional<ActionBlock> action_block;
  if constexpr (mode == CfrTraversalMode::kNormal) {
    const InfoSetAddress info_set_address{
        public_state_id, player,
        card_abstraction_.private_bucket(player_cards.combo, row.state)};
    action_block =
        is_update_player
            ? strategy_store_.get_or_create(info_set_address, legal_action_ids)
            : strategy_store_.find(info_set_address, action_count);
  } else {
    action_block =
        strategy_store_.find_frozen(public_state_id, player,
                                    player_cards.combo, action_count);
  }

  ActionScratch action_scratch;
  absl::Span<double> action_probabilities =
      action_scratch.probs(action_count);
  absl::Span<double> action_values = action_scratch.vals(action_count);
  strategy_store_.regret_matching_or_uniform(
      action_block, action_count, ctx.options().regret_load_mode,
      action_probabilities);

  double node_value = 0.0;
  std::optional<ActionRangeConditioning> range_conditioning;
  if constexpr (mode == CfrTraversalMode::kNormal) {
    range_conditioning.emplace(
        *this, ctx, *node_cursor, public_state_id, player, legal_action_ids);
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    const ChildResult child =
        graph.action_child(node, static_cast<int>(action_index));
    if (child.status != ChildStatus::kOk) {
      continue;
    }

    double action_value = 0.0;
    {
      auto reach_scope =
          ctx.enter_action(player, action_probabilities[action_index]);
      auto depth_scope = ctx.descend();
      if constexpr (mode == CfrTraversalMode::kNormal) {
        if (range_conditioning->enabled()) {
          auto range_scope =
              ctx.set_ranges(
                  range_conditioning->player_a_range_for(action_index),
                  range_conditioning->player_b_range_for(action_index));
          action_value = cfr_traversal<mode>(child.node, ctx, graph);
        } else {
          action_value = cfr_traversal<mode>(child.node, ctx, graph);
        }
      } else {
        action_value = cfr_traversal<mode>(child.node, ctx, graph);
      }
    }
    action_values[action_index] = action_value;
    node_value += action_probabilities[action_index] * action_value;
  }

  ++cfr_update_count_;
  record_cfr_update(street, ctx.depth());

  if (action_block.has_value() && is_update_player) {
    const double opponent_reach_prob = ctx.opponent_reach(player);
    const RegretUpdateOptions regret_update_options =
        ctx.regret_update_options();
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      action_block->add_cfr_plus_regret(
          action_index, static_cast<float>(regret), regret_update_options);
    }

    if constexpr (mode == CfrTraversalMode::kNormal) {
      if (ctx.options().write_average_strategy) {
        action_block->add_average_strategy(
            action_probabilities, ctx.average_strategy_weight(player),
            ctx.options().regret_update_mode);
      }
    }
  }

  return node_value;
}

double CFRSolver::cfr_with_ranges(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  return cfr_traversal<CfrTraversalMode::kNormal>(node, ctx, graph);
}

double CFRSolver::chance_sampling_cfr(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  const auto& rows = storage_.frozen_ref().public_state_rows;
  if (node.public_state_id >= rows.size()) {
    return 0.0;
  }

  const int samples = ChanceSamples(config_);
  record_chance_samples(samples);
  TrainingRangeView* public_player_a_range = nullptr;
  TrainingRangeView* public_player_b_range = nullptr;
  if (ctx.range(0).has_value() || ctx.range(1).has_value()) {
    RangeScratchFrame& scratch_frame = ctx.scratch_frame();
    public_player_a_range = &scratch_frame.public_player_a_range;
    public_player_b_range = &scratch_frame.public_player_b_range;
  }

  double value = 0.0;
  int evaluated = 0;
  const CardMask known_private_cards = ctx.known_private_cards();
  for (int i = 0; i < samples; ++i) {
    const ChildResult child =
        graph.sample_chance_child(node, known_private_cards);
    if (child.status != ChildStatus::kOk) {
      continue;
    }

    OptionalTrainingRange child_player_a_range = ctx.range(0);
    OptionalTrainingRange child_player_b_range = ctx.range(1);
    if (child_player_a_range.has_value()) {
      child_player_a_range =
          std::cref(child_player_a_range->get().without_mask(
              child.node.exact_board.mask, *public_player_a_range));
    }
    if (child_player_b_range.has_value()) {
      child_player_b_range =
          std::cref(child_player_b_range->get().without_mask(
              child.node.exact_board.mask, *public_player_b_range));
    }

    if (child_player_a_range.has_value() || child_player_b_range.has_value()) {
      auto range_scope =
          ctx.set_ranges(child_player_a_range, child_player_b_range);
      value += cfr_with_ranges(child.node, ctx, graph);
    } else {
      value += cfr_with_ranges(child.node, ctx, graph);
    }
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

double CFRSolver::cfr_frozen_regret_only(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  return cfr_traversal<CfrTraversalMode::kFrozenRegretOnly>(
      node, ctx, graph);
}

double CFRSolver::chance_sampling_frozen_regret_only(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  const int samples = ChanceSamples(config_);
  record_chance_samples(samples);
  return average_sampled_chance(
      samples, node, ctx.known_private_cards(), graph,
      [&](NodeRef child) {
        return cfr_frozen_regret_only(child, ctx, graph);
      });
}

template <typename EvalChild>
double CFRSolver::average_sampled_chance(
    int samples,
    NodeRef node,
    CardMask known_private_cards,
    NodeGraph& graph,
    EvalChild&& eval_child) {
  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const ChildResult child =
        graph.sample_chance_child(node, known_private_cards);
    if (child.status != ChildStatus::kOk) {
      continue;
    }
    value += eval_child(child.node);
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

CFRSolver::ActionRangeConditioning::ActionRangeConditioning(
    CFRSolver& solver,
    TraversalContext& ctx,
    const NodeCursor& node_cursor,
    uint32_t public_state_id,
    int player,
    absl::Span<const int> legal_action_ids)
    : original_player_a_range_(ctx.range(0)),
      original_player_b_range_(ctx.range(1)),
      condition_player_a_(player == 0 &&
                          original_player_a_range_.has_value()),
      condition_player_b_(player == 1 &&
                          original_player_b_range_.has_value()) {
  if (!enabled()) {
    return;
  }

  conditioned_ranges_ = &ctx.scratch_frame().conditioned_ranges;
  const CompactPublicState& state = node_cursor.exact_state();
  if (condition_player_a_) {
    solver.condition_ranges_for_actions(
        original_player_a_range_->get(), state, public_state_id, player,
        legal_action_ids, *conditioned_ranges_);
  } else {
    solver.condition_ranges_for_actions(
        original_player_b_range_->get(), state, public_state_id, player,
        legal_action_ids, *conditioned_ranges_);
  }
}

CFRSolver::OptionalTrainingRange
CFRSolver::ActionRangeConditioning::player_a_range_for(
    size_t action_index) const {
  if (condition_player_a_) {
    return std::cref((*conditioned_ranges_)[action_index]);
  }
  return original_player_a_range_;
}

CFRSolver::OptionalTrainingRange
CFRSolver::ActionRangeConditioning::player_b_range_for(
    size_t action_index) const {
  if (condition_player_b_) {
    return std::cref((*conditioned_ranges_)[action_index]);
  }
  return original_player_b_range_;
}

void CFRSolver::condition_ranges_for_actions(
    const TrainingRangeView& range,
    const CompactPublicState& state,
    uint32_t public_state_id,
    int player,
    absl::Span<const int> conditioned_action_ids,
    std::vector<TrainingRangeView>& conditioned_ranges) {
  const size_t action_count = conditioned_action_ids.size();
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

  const CardMask board_mask = state.board_mask;
  ActionScratch action_scratch;
  absl::Span<double> action_probabilities =
      action_scratch.probs(action_count);
  for (size_t i = 0; i < range_size; ++i) {
    const float range_weight = range.weight(i);
    const ComboId combo_id = range.combo(i);
    if (range_weight <= 0.0 || (ComboMask(combo_id) & board_mask) != 0) {
      continue;
    }

    const PrivateBucketId private_bucket =
        card_abstraction_.private_bucket(combo_id, state);
    strategy_store_.regret_matching_for_bucket(
        public_state_id, player, private_bucket, conditioned_action_ids,
        action_probabilities);

    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double conditioned_weight =
          range_weight * action_probabilities[action_index];
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
      storage_.frozen_ref().public_state_rows[*root_public_state_id].state;
  const NodeRef root_node{*root_public_state_id,
                          ExactBoardFromState(root_state)};
  NodeGraph graph(*this, default_node_graph_mode());
  EvaluationContext ctx{TraversalDeal{{
      PrivateCards::FromCombo(player_a_hand),
      PrivateCards::FromCombo(player_b_hand),
  }}};
  return evaluate_strategy_node(root_node, ctx, graph);
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
  std::shared_ptr<const FrozenStrategyTables> frozen_tables =
      storage_.frozen_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative = storage_.cumulative;
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
      worker.storage_.bind_frozen(frozen_tables, cumulative);
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
  std::optional<NodeRef> root_node = root_node_ref(root_public_state_id);
  if (!root_node.has_value()) {
    return 0.0;
  }
  NodeGraph graph(*this, default_node_graph_mode());
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    EvaluationContext ctx{TraversalDeal{{
        PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo),
    }}};
    total += evaluate_strategy_node(*root_node, ctx, graph);
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    NodeRef node,
    EvaluationContext& ctx,
    NodeGraph& graph) {
  const std::optional<NodeCursor> node_cursor = cursor(node);
  if (!node_cursor.has_value()) {
    return 0.0;
  }
  const uint32_t public_state_id = node.public_state_id;
  const PublicStateRow& row = node_cursor->row();

  if (row.is_terminal) {
    const CompactPublicState& state = node_cursor->exact_state();
    return utility(state, ctx.cards(0), ctx.cards(1));
  }
  if (row.is_chance_node) {
    const int samples = ChanceSamples(config_);
    return average_sampled_chance(
        samples, node, ctx.known_private_cards(), graph,
        [&](NodeRef child) {
          return evaluate_strategy_node(child, ctx, graph);
        });
  }
  if (row.action_count == 0) {
    return 0.0;
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player)) {
    return 0.0;
  }

  const PrivateCards& player_cards = ctx.cards(player);
  ActionScratch action_scratch;
  absl::Span<double> probabilities =
      action_scratch.probs(row.action_count);
  const PrivateBucketId private_bucket =
      card_abstraction_.private_bucket(player_cards.combo, row.state);
  strategy_store_.average_strategy(
      public_state_id, row, player, private_bucket,
      config_.regret_only_training, probabilities);

  double value = 0.0;
  const int action_count = row.action_count;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    const ChildResult child =
        graph.action_child(node, action_index);
    if (child.status != ChildStatus::kOk) {
      continue;
    }
    value += probabilities[action_index] *
             evaluate_strategy_node(child.node, ctx, graph);
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

} // namespace poker
