#include "src/public_state_graph.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "src/build_flags.h"
#include "src/card_utils.h"
#include "src/game_tree.h"

namespace poker {
namespace {

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
    if (public_state_id >= row_count || public_state_id >= queued_.size()) {
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
bool ForEachCardCombination(int count,
                            CardMask blocked_mask,
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
      static_cast<int>(std::popcount(state.board_mask));
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

}  // namespace

PublicStateGraph::PublicStateGraph(
    const SolverConfig& config,
    SolverStorage& storage,
    const CardAbstraction& card_abstraction,
    const BettingAbstraction& betting_abstraction,
    TraversalStats& stats)
    : config_(config),
      storage_(storage),
      card_abstraction_(card_abstraction),
      betting_abstraction_(betting_abstraction),
      stats_(stats) {}

PublicStateGraph::BettingHistoryKey
PublicStateGraph::betting_history_key(
    const CompactPublicState& state) const {
  return betting_abstraction_.make_history_key(state);
}

PublicStateGraph::BettingHistoryRow
PublicStateGraph::make_betting_history_row(
    const BettingHistoryKey& key) const {
  return betting_abstraction_.make_history_row(key);
}

PublicStateGraph::PublicStateKey
PublicStateGraph::row_key(
    uint32_t betting_history_id,
    const CompactPublicState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

std::optional<uint32_t> PublicStateGraph::find_row(
    uint32_t betting_history_id,
    const CompactPublicState& state) const {
  const auto existing =
      tables().public_state_ids.find(row_key(betting_history_id, state));
  if (existing == tables().public_state_ids.end()) {
    return std::nullopt;
  }
  return existing->second;
}

PublicStateGraph::PublicStateRow PublicStateGraph::make_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
  state = betting_abstraction_.public_state_for_row(std::move(state));
  PublicStateRow row;
  row.betting_history_id = betting_history_id;
  row.public_bucket = card_abstraction_.public_bucket(state);
  row.is_terminal = IsTerminal(state);
  row.player_to_act = GetPlayerToAct(state);
  row.state = std::move(state);
  row.is_chance_node = !row.is_terminal && row.player_to_act == -1;

  if (!row.is_terminal && !row.is_chance_node && IsPlayer(row.player_to_act)) {
    row.action_count = betting_abstraction_.actions_for_betting_node(
        row.state, row.player_to_act, row.actions);
    for (int i = 0; i < row.action_count; ++i) {
      row.action_ids[static_cast<size_t>(i)] =
          betting_abstraction_.action_key(
              row.actions[static_cast<size_t>(i)]);
    }
  }

  return row;
}

std::optional<uint32_t> PublicStateGraph::get_or_create_row(
    uint32_t betting_history_id,
    CompactPublicState state) {
  if (std::optional<uint32_t> existing = find_row(betting_history_id, state)) {
    return existing;
  }

  if (storage_.frozen || row_limit_reached()) {
    return std::nullopt;
  }

  StrategyTables& tables = mtables();
  const uint32_t public_state_id =
      static_cast<uint32_t>(tables.public_state_rows.size());
  const auto [state_iter, inserted] = tables.public_state_ids.try_emplace(
      row_key(betting_history_id, state), public_state_id);
  if (!inserted) {
    return state_iter->second;
  }
  tables.public_state_rows.push_back(
      make_row(betting_history_id, std::move(state)));
  cache_betting_history_actions(betting_history_id,
                                tables.public_state_rows.back());
  return public_state_id;
}

std::optional<uint32_t> PublicStateGraph::get_or_create_row(
    const CompactPublicState& state) {
  if (storage_.frozen) {
    BettingHistoryKey key = betting_history_key(state);
    const auto betting_history = tables().betting_history_ids.find(key);
    if (betting_history == tables().betting_history_ids.end()) {
      return std::nullopt;
    }
    return get_or_create_row(betting_history->second, state);
  }

  const uint32_t betting_history_id = get_or_create_betting_history(state);
  return get_or_create_row(betting_history_id, state);
}

uint32_t PublicStateGraph::get_or_create_betting_history(
    const CompactPublicState& state) {
  BettingHistoryKey key = betting_history_key(state);
  BettingHistoryRow row = make_betting_history_row(key);
  return get_or_create_betting_history(std::move(key), std::move(row));
}

uint32_t PublicStateGraph::get_or_create_betting_history(
    BettingHistoryKey key,
    BettingHistoryRow row) {
  StrategyTables& tables = mtables();
  const auto [history_iter, inserted] = tables.betting_history_ids.try_emplace(
      std::move(key), static_cast<uint32_t>(tables.betting_history_ids.size()));
  const uint32_t betting_history_id = history_iter->second;
  if (!inserted) {
    if (tables.betting_history_rows.size() <= betting_history_id) {
      throw std::logic_error(
          "betting-history ID map and row table are out of sync");
    }
    return betting_history_id;
  }
  tables.betting_history_rows.push_back(std::move(row));
  return betting_history_id;
}

uint32_t PublicStateGraph::get_or_create_action_history_child(
    uint32_t parent_history_id,
    int action_index,
    const CompactPublicState& child_state) {
  StrategyTables& tables = mtables();
  if (parent_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      const uint32_t child_id =
          parent_row.action_child_ids[static_cast<size_t>(action_index)];
      if (child_id != kInvalidBettingHistoryId) {
        return child_id;
      }
    }
  }

  BettingHistoryKey child_key;
  if (parent_history_id < tables.betting_history_rows.size()) {
    const BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      child_key = betting_abstraction_.make_action_child_history_key(
          parent_row, action_index, child_state);
    } else {
      child_key = betting_history_key(child_state);
    }
  } else {
    child_key = betting_history_key(child_state);
  }
  BettingHistoryRow child_row = make_betting_history_row(child_key);
  const uint32_t child_id = get_or_create_betting_history(
      std::move(child_key), std::move(child_row));
  if (parent_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      parent_row.action_child_ids[static_cast<size_t>(action_index)] =
          child_id;
    }
  }
  return child_id;
}

uint32_t PublicStateGraph::get_or_create_chance_history_child(
    uint32_t parent_history_id,
    const CompactPublicState& child_state) {
  StrategyTables& tables = mtables();
  if (parent_history_id < tables.betting_history_rows.size()) {
    BettingHistoryRow& parent_row =
        tables.betting_history_rows[parent_history_id];
    if (parent_row.chance_child_id != kInvalidBettingHistoryId) {
      return parent_row.chance_child_id;
    }
  }

  const uint32_t child_id = get_or_create_betting_history(child_state);
  if (parent_history_id < tables.betting_history_rows.size()) {
    tables.betting_history_rows[parent_history_id].chance_child_id =
        child_id;
  }
  return child_id;
}

void PublicStateGraph::cache_betting_history_actions(
    uint32_t betting_history_id,
    const PublicStateRow& row) {
  StrategyTables& tables = mtables();
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

std::optional<uint32_t> PublicStateGraph::find_action_child(
    uint32_t parent_public_state_id,
    int action_index) const {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent_public_state_id];
  if (action_index < 0 || action_index >= row.action_count) {
    throw std::logic_error("action child index out of range");
  }
  const uint32_t child_id =
      row.action_child_ids[static_cast<size_t>(action_index)];
  if (child_id == kInvalidPublicStateId ||
      child_id == kCappedPublicStateId) {
    return std::nullopt;
  }
  return child_id;
}

std::optional<uint32_t> PublicStateGraph::find_chance_child(
    uint32_t parent_public_state_id,
    const CompactPublicState& child_state) const {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent_public_state_id];
  const size_t begin = row.chance_child_offset;
  const size_t end = begin + row.chance_child_count;
  const auto& entries = tables().chance_child_entries;
  if (begin > entries.size() || end > entries.size()) {
    return std::nullopt;
  }
  const PublicBucketId outcome_id = chance_outcome_id(child_state);
  const auto first =
      entries.begin() + static_cast<std::ptrdiff_t>(begin);
  const auto last =
      entries.begin() + static_cast<std::ptrdiff_t>(end);
  const auto iter = std::lower_bound(
      first, last, outcome_id,
      [](const auto& entry, PublicBucketId target_outcome_id) {
        return entry.outcome_id < target_outcome_id;
      });
  if (iter == last || iter->outcome_id != outcome_id) {
    return std::nullopt;
  }
  return iter->public_state_id;
}

template <typename Callback>
bool PublicStateGraph::for_each_required_chance_transition(
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
          ApplyChance(parent_state, transition.cards);
      if (!callback(child_state, absl::Span<const CardId>(transition.cards))) {
        return false;
      }
    }
    return true;
  } else {
    return ForEachNextStreetDeal(row.state, [&](absl::Span<const CardId> cards) {
      return callback(ApplyChance(row.state, cards), cards);
    });
  }
}

PublicStateGraph::PublicBucketId PublicStateGraph::chance_outcome_id(
    const CompactPublicState& child_state) const {
  return card_abstraction_.public_bucket(child_state);
}

std::optional<uint32_t> PublicStateGraph::find_or_cache_action_child(
    uint32_t parent_public_state_id,
    int action_index) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent_public_state_id];
  if (action_index < 0 || action_index >= row.action_count) {
    throw std::logic_error(
        "find_or_cache_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const uint32_t row_child_id = row.action_child_ids[action_slot];
  if (row_child_id != kInvalidPublicStateId) {
    return row_child_id;
  }

  const uint32_t parent_betting_history_id = row.betting_history_id;
  const auto& betting_history_rows = tables().betting_history_rows;
  if (parent_betting_history_id >= betting_history_rows.size()) {
    return std::nullopt;
  }

  const BettingHistoryRow& parent_betting_history =
      betting_history_rows[parent_betting_history_id];
  if (action_index >= parent_betting_history.action_count) {
    return std::nullopt;
  }

  const uint32_t child_betting_history_id =
      parent_betting_history.action_child_ids[action_slot];
  if (child_betting_history_id == kInvalidBettingHistoryId) {
    return std::nullopt;
  }

  const PublicStateKey child_key{
      child_betting_history_id,
      row.public_bucket,
  };
  auto existing_public_child = tables().public_state_ids.find(child_key);
  if (existing_public_child == tables().public_state_ids.end()) {
    return std::nullopt;
  }

  if (!storage_.frozen) {
    mtables()
        .public_state_rows[parent_public_state_id]
        .action_child_ids[action_slot] = existing_public_child->second;
  }
  return existing_public_child->second;
}

bool PublicStateGraph::row_limit_reached() const {
  return config_.max_public_states > 0 &&
         static_cast<int>(rows().size()) >= config_.max_public_states;
}

bool PublicStateGraph::can_insert_row() const {
  return !storage_.frozen && !row_limit_reached();
}

std::optional<uint32_t>
PublicStateGraph::get_or_create_action_child(
    uint32_t parent_public_state_id,
    int action_index) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& read_row = public_rows[parent_public_state_id];
  if (action_index < 0 || action_index >= read_row.action_count) {
    throw std::logic_error(
        "get_or_create_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  if (std::optional<uint32_t> existing_child_id =
          find_or_cache_action_child(parent_public_state_id, action_index)) {
    stats_.record_transition_hit();
    if (*existing_child_id == kCappedPublicStateId) {
      return std::nullopt;
    }
    return existing_child_id;
  }

  if (!can_insert_row()) {
    if (!storage_.frozen) {
      mtables()
          .public_state_rows[parent_public_state_id]
          .action_child_ids[action_slot] = kCappedPublicStateId;
    }
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const uint32_t parent_history_id = read_row.betting_history_id;
  CompactPublicState child_state =
      ApplyAction(read_row.state, read_row.actions[action_slot]);
  const uint32_t child_history_id = get_or_create_action_history_child(
      parent_history_id, action_index, child_state);
  auto child_id = get_or_create_row(child_history_id, std::move(child_state));
  if (!child_id.has_value()) {
    mtables()
        .public_state_rows[parent_public_state_id]
        .action_child_ids[action_slot] = kCappedPublicStateId;
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mtables()
      .public_state_rows[parent_public_state_id]
      .action_child_ids[action_slot] = *child_id;
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

std::optional<uint32_t> PublicStateGraph::find_or_cache_chance_child(
    uint32_t parent_public_state_id,
    const CompactPublicState& child_state) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent_public_state_id];
  const PublicBucketId outcome_id = chance_outcome_id(child_state);
  const ChanceTransitionKey transition_key{parent_public_state_id, outcome_id};
  auto existing = tables().public_chance_child_ids.find(transition_key);
  if (existing != tables().public_chance_child_ids.end()) {
    return existing->second;
  }

  if constexpr (kCoarsePublicBuckets) {
    const uint32_t parent_betting_history_id = row.betting_history_id;
    const auto& betting_history_rows = tables().betting_history_rows;
    if (parent_betting_history_id >= betting_history_rows.size()) {
      return std::nullopt;
    }

    const uint32_t child_betting_history_id =
        betting_history_rows[parent_betting_history_id].chance_child_id;
    if (child_betting_history_id == kInvalidBettingHistoryId) {
      return std::nullopt;
    }

    const PublicStateKey child_public_key{
        child_betting_history_id,
        outcome_id,
    };
    auto existing_public_child =
        tables().public_state_ids.find(child_public_key);
    if (existing_public_child == tables().public_state_ids.end()) {
      return std::nullopt;
    }

    if (!storage_.frozen) {
      mtables().public_chance_child_ids.emplace(
          transition_key, existing_public_child->second);
    }
    return existing_public_child->second;
  }

  return std::nullopt;
}

std::optional<uint32_t>
PublicStateGraph::get_or_create_chance_child(
    uint32_t parent_public_state_id,
    const CompactPublicState& exact_child_state) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicBucketId outcome_id = chance_outcome_id(exact_child_state);
  if (std::optional<uint32_t> existing_child_id =
          find_or_cache_chance_child(parent_public_state_id,
                                     exact_child_state)) {
    stats_.record_transition_hit();
    return *existing_child_id;
  }

  if (!can_insert_row()) {
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const PublicStateRow& parent_row = public_rows[parent_public_state_id];
  CompactPublicState stored_child_state = exact_child_state;
  const uint32_t parent_history_id = parent_row.betting_history_id;
  const uint32_t child_history_id = get_or_create_chance_history_child(
      parent_history_id, stored_child_state);
  auto child_id = get_or_create_row(child_history_id,
                                    std::move(stored_child_state));
  if (!child_id.has_value()) {
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mtables().public_chance_child_ids.emplace(
      ChanceTransitionKey{parent_public_state_id, outcome_id}, *child_id);
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

bool PublicStateGraph::prebuild_reachable_rows(
    uint32_t root_public_state_id,
    int max_depth) {
  if (storage_.frozen) {
    return true;
  }
  if (root_public_state_id >= rows().size()) {
    return false;
  }

  PublicStateBfs bfs(root_public_state_id, rows().size());
  while (std::optional<PublicStateBfs::Entry> maybe_entry = bfs.next()) {
    const PublicStateBfs::Entry entry = *maybe_entry;
    if (entry.public_state_id >= rows().size()) {
      return false;
    }
    // Copy before creating children; child creation can append to rows and
    // invalidate references.
    const PublicStateRow row = rows()[entry.public_state_id];
    if (row.is_terminal) {
      continue;
    }

    if (row.is_chance_node) {
      const bool complete = for_each_required_chance_transition(
          row, [&](const CompactPublicState& child_state,
                   absl::Span<const CardId>) {
            std::optional<uint32_t> child_public_state_id =
                get_or_create_chance_child(entry.public_state_id,
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
          get_or_create_action_child(entry.public_state_id,
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

void PublicStateGraph::rebuild_chance_child_entries() {
  StrategyTables& tables = mtables();
  struct PendingChanceChild {
    uint32_t parent_id = 0;
    PublicBucketId outcome_id = 0;
    uint32_t public_state_id = kInvalidPublicStateId;

    bool operator<(const PendingChanceChild& other) const {
      return std::tie(parent_id, outcome_id) <
             std::tie(other.parent_id, other.outcome_id);
    }
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

  std::sort(pending.begin(), pending.end());

  for (PublicStateRow& row : tables.public_state_rows) {
    row.chance_child_offset = 0;
    row.chance_child_count = 0;
  }
  tables.chance_child_entries.clear();
  tables.chance_child_entries.reserve(pending.size());
  uint32_t current_parent_id = kInvalidPublicStateId;
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

bool PublicStateGraph::validate_prebuilt_rows(
    uint32_t root_public_state_id,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& public_rows = rows();
  const auto& history_rows = tables().betting_history_rows;
  if (root_public_state_id >= public_rows.size()) {
    return false;
  }
  stats.betting_history_transition_prebuild_complete = true;
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  PublicStateBfs bfs(root_public_state_id, public_rows.size());

  auto valid_public_child = [&](uint32_t public_state_id) {
    return public_state_id != kInvalidPublicStateId &&
           public_state_id != kCappedPublicStateId &&
           public_state_id < public_rows.size();
  };
  auto matching_betting_child = [&](bool valid_child,
                                    uint32_t public_state_id,
                                    uint32_t betting_history_id) {
    return valid_child && betting_history_id != kInvalidBettingHistoryId &&
           betting_history_id < history_rows.size() &&
           public_rows[public_state_id].betting_history_id ==
               betting_history_id;
  };
  auto enqueue_child = [&](uint32_t public_state_id, int depth) {
    return bfs.enqueue_existing(public_state_id, depth, public_rows.size());
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
    if (entry.public_state_id >= public_rows.size()) {
      stats.betting_history_transition_prebuild_complete = false;
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const PublicStateRow& row = public_rows[entry.public_state_id];
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
                   absl::Span<const CardId>) {
        ++stats.prebuild_chance_transitions;
        ++stats.prebuild_betting_history_transitions;
        const std::optional<uint32_t> child_public_state_id =
            find_chance_child(entry.public_state_id, child_state);
        const uint32_t child_id =
            child_public_state_id.value_or(kInvalidPublicStateId);
        const bool valid_child = valid_public_child(child_id);
        if (!valid_child) {
          mark_missing_chance();
        }
        if (!matching_betting_child(valid_child, child_id,
                                    history_row.chance_child_id)) {
          mark_missing_betting_history();
        }
        if (valid_child && !enqueue_child(child_id, entry.depth)) {
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
          history_row.action_ids[action_slot] == row.action_ids[action_slot] &&
          matching_betting_child(valid_action_child, child_public_state_id,
                                 child_betting_history_id);
      if (!valid_betting_child) {
        mark_missing_betting_history();
      }
      if (valid_action_child &&
          !enqueue_child(child_public_state_id, entry.depth + 1)) {
        mark_missing_action();
        mark_missing_betting_history();
      }
    }
  }

  return stats.betting_history_transition_prebuild_complete &&
         stats.action_transition_prebuild_complete &&
         stats.chance_transition_prebuild_complete;
}

}  // namespace poker
