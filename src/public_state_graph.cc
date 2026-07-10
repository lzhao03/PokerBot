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
    uint32_t node_id = 0;
    Board board;
    int depth = 0;
  };

  PublicStateBfs(uint32_t root_id, Board root_board, size_t row_count) {
    queue_.reserve(1024);
    queued_.resize(row_count, 0);
    enqueue_growing(root_id, std::move(root_board), 0);
  }

  std::optional<Entry> next() {
    if (cursor_ >= queue_.size()) {
      return std::nullopt;
    }
    return queue_[cursor_++];
  }

  bool enqueue_existing(uint32_t node_id,
                        Board board,
                        int depth,
                        size_t row_count) {
    if (node_id >= row_count || node_id >= queued_.size()) {
      return false;
    }
    enqueue_known_index(node_id, std::move(board), depth);
    return true;
  }

  void enqueue_growing(uint32_t node_id, Board board, int depth) {
    if (node_id >= queued_.size()) {
      queued_.resize(static_cast<size_t>(node_id) + 1, 0);
    }
    enqueue_known_index(node_id, std::move(board), depth);
  }

 private:
  void enqueue_known_index(uint32_t node_id, Board board, int depth) {
    if (queued_[node_id]) {
      return;
    }
    queued_[node_id] = 1;
    queue_.push_back({node_id, std::move(board), depth});
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
bool ForEachNextStreetDeal(StreetKind street,
                           const Board& board,
                           Callback callback) {
  const int remaining_board_slots =
      std::max(0, kMaxBoardCards - static_cast<int>(board.count));
  const int count =
      std::min(CardsForNextStreet(street), remaining_board_slots);
  if (count <= 0) {
    return callback(absl::Span<const CardId>());
  }

  const int available_cards =
      kDeckCardCount -
      static_cast<int>(std::popcount(board.mask));
  if (available_cards < count) {
    throw std::runtime_error("Not enough cards to enumerate next street");
  }
  return ForEachCardCombination(count, board.mask, callback);
}

struct CoarseChanceTransitionTemplate {
  Board parent_board;
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
    Board parent;
    for (CardId card : board) {
      parent.add(card);
    }
    const PublicBucketId parent_bucket =
        abstraction.public_bucket(street, parent);
    ForEachCardCombination(deal_count, parent.mask,
                           [&](absl::Span<const CardId> cards) {
      Board child = parent;
      for (CardId card : cards) {
        child.add(card);
      }
      const PublicBucketId child_bucket =
          abstraction.public_bucket(StreetAfterChance(street), child);
      const uint64_t seen_key = (parent_bucket << 32) | child_bucket;
      if (!seen.insert(seen_key).second) {
        return true;
      }
      CoarseChanceTransitionTemplate transition;
      transition.parent_board = parent;
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

int FirstBettingPlayer(StreetKind street) {
  return street == StreetKind::kPreflop ? 0 : 1;
}

StrategyTables::NodeKind BettingNodeKind(const BettingState& state) {
  const bool betting_over = IsBettingRoundOver(state);
  if (state.folded_player >= 0 ||
      (betting_over && state.street == StreetKind::kRiver)) {
    return StrategyTables::NodeKind::kTerminal;
  }
  if (betting_over) {
    return StrategyTables::NodeKind::kChance;
  }
  return StrategyTables::NodeKind::kDecision;
}

int BettingNodePlayerToAct(const BettingState& state) {
  if (BettingNodeKind(state) != StrategyTables::NodeKind::kDecision) {
    return -1;
  }
  return IsPlayer(state.player_to_act) ? state.player_to_act
                                       : FirstBettingPlayer(state.street);
}

bool SameBettingState(const BettingState& first, const BettingState& second) {
  return first.stack == second.stack &&
         first.contribution == second.contribution &&
         first.pot == second.pot &&
         first.street == second.street &&
         first.player_to_act == second.player_to_act &&
         first.folded_player == second.folded_player &&
         first.actions_this_street == second.actions_this_street &&
         first.last_action.kind == second.last_action.kind &&
         first.last_action.amount == second.last_action.amount &&
         first.last_action.player == second.last_action.player &&
         first.all_in == second.all_in;
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
    const BettingState& state) const {
  return betting_abstraction_.make_history_key(state);
}

PublicStateGraph::BettingHistoryRow
PublicStateGraph::make_betting_history_row(
    const BettingHistoryKey& key) const {
  return betting_abstraction_.make_history_row(key);
}

PublicStateGraph::BettingNodeId PublicStateGraph::append_betting_node(
    const BettingState& state) {
  StrategyTables& tables = mtables();
  BettingNode node;
  node.state = state;
  node.kind = BettingNodeKind(state);
  node.player_to_act = BettingNodePlayerToAct(state);

  if (node.kind == StrategyTables::NodeKind::kDecision) {
    const auto menu =
        betting_abstraction_.actions_for_betting_node(state, node.player_to_act);
    node.action_begin = static_cast<uint32_t>(tables.betting_edges.size());
    node.action_count = menu.count;
    for (uint8_t i = 0; i < menu.count; ++i) {
      const GameAction action = menu.actions[static_cast<size_t>(i)];
      tables.betting_edges.push_back({
          action,
          betting_abstraction_.action_key(action),
          kInvalidBettingNodeId,
      });
    }
  }

  const BettingNodeId node_id =
      static_cast<BettingNodeId>(tables.betting_nodes.size());
  tables.betting_nodes.push_back(std::move(node));
  return node_id;
}

PublicStateGraph::BettingNodeId
PublicStateGraph::get_or_create_root_betting_node(const BettingState& state) {
  StrategyTables& tables = mtables();
  if (tables.root_betting_node_id == kInvalidBettingNodeId) {
    tables.root_betting_node_id = append_betting_node(state);
    return tables.root_betting_node_id;
  }

  const BettingNodeId node_id = tables.root_betting_node_id;
  if (node_id >= tables.betting_nodes.size() ||
      !SameBettingState(tables.betting_nodes[node_id].state, state)) {
    throw std::logic_error("root betting node does not match root state");
  }
  return node_id;
}

PublicStateGraph::BettingNodeId
PublicStateGraph::get_or_create_action_betting_child(
    BettingNodeId parent_node_id,
    int action_index) {
  StrategyTables& tables = mtables();
  if (parent_node_id >= tables.betting_nodes.size()) {
    throw std::logic_error("action betting parent node is invalid");
  }
  const BettingNode& parent = tables.betting_nodes[parent_node_id];
  if (action_index < 0 || action_index >= parent.action_count) {
    throw std::logic_error("action betting child index out of range");
  }

  const size_t edge_index = static_cast<size_t>(parent.action_begin) +
                            static_cast<size_t>(action_index);
  const BettingNodeId existing_child = tables.betting_edges[edge_index].child;
  if (existing_child != kInvalidBettingNodeId) {
    return existing_child;
  }

  const GameAction action = tables.betting_edges[edge_index].action;
  const BettingState child_state = ApplyAction(parent.state, action);
  const BettingNodeId child_node_id = append_betting_node(child_state);
  tables.betting_edges[edge_index].child = child_node_id;
  return child_node_id;
}

PublicStateGraph::BettingNodeId
PublicStateGraph::get_or_create_chance_betting_child(
    BettingNodeId parent_node_id,
    const BettingState& child_state) {
  StrategyTables& tables = mtables();
  if (parent_node_id >= tables.betting_nodes.size()) {
    throw std::logic_error("chance betting parent node is invalid");
  }
  const BettingNodeId existing_child =
      tables.betting_nodes[parent_node_id].chance_child;
  if (existing_child == kInvalidBettingNodeId) {
    const BettingNodeId child_node_id = append_betting_node(child_state);
    tables.betting_nodes[parent_node_id].chance_child = child_node_id;
    return child_node_id;
  }

  if (existing_child >= tables.betting_nodes.size() ||
      !SameBettingState(tables.betting_nodes[existing_child].state,
                        child_state)) {
    throw std::logic_error("chance betting child state mismatch");
  }
  return existing_child;
}

PublicStateGraph::PublicStateKey
PublicStateGraph::row_key(
    BettingNodeId betting_node_id,
    StreetKind street,
    const Board& board) const {
  return {betting_node_id, card_abstraction_.public_bucket(street, board)};
}

std::optional<uint32_t> PublicStateGraph::find_row(
    BettingNodeId betting_node_id,
    StreetKind street,
    const Board& board) const {
  const auto existing =
      tables().public_state_ids.find(row_key(betting_node_id, street, board));
  if (existing == tables().public_state_ids.end()) {
    return std::nullopt;
  }
  return existing->second;
}

PublicStateGraph::PublicStateRow PublicStateGraph::make_row(
    uint32_t betting_history_id,
    BettingNodeId betting_node_id,
    const ExactGameState& state) {
  PublicStateRow row;
  row.betting_history_id = betting_history_id;
  row.betting_node_id = betting_node_id;
  row.public_bucket =
      card_abstraction_.public_bucket(state.betting.street, state.board);
  row.betting = state.betting;
  row.is_terminal = IsTerminal(row.betting, state.board);
  row.player_to_act = GetPlayerToAct(row.betting, state.board);
  row.is_chance_node = !row.is_terminal && row.player_to_act == -1;

  const auto& nodes = tables().betting_nodes;
  if (betting_node_id >= nodes.size()) {
    throw std::logic_error("public row betting node is invalid");
  }
  const BettingNode& node = nodes[betting_node_id];
  if (!SameBettingState(node.state, row.betting)) {
    throw std::logic_error("public row betting node state mismatch");
  }
  if (node.player_to_act != row.player_to_act) {
    throw std::logic_error("public row betting node metadata mismatch");
  }

  return row;
}

std::optional<uint32_t> PublicStateGraph::get_or_create_row(
    uint32_t betting_history_id,
    BettingNodeId betting_node_id,
    const ExactGameState& state) {
  if (std::optional<uint32_t> existing =
          find_row(betting_node_id, state.betting.street, state.board)) {
    return existing;
  }

  if (storage_.frozen || row_limit_reached()) {
    return std::nullopt;
  }

  StrategyTables& tables = mtables();
  const uint32_t public_state_id =
      static_cast<uint32_t>(tables.public_state_rows.size());
  const auto [state_iter, inserted] = tables.public_state_ids.try_emplace(
      row_key(betting_node_id, state.betting.street, state.board),
      public_state_id);
  if (!inserted) {
    return state_iter->second;
  }
  tables.public_state_rows.push_back(
      make_row(betting_history_id, betting_node_id, state));
  cache_betting_history_actions(betting_history_id,
                                tables.public_state_rows.back());
  return public_state_id;
}

std::optional<uint32_t> PublicStateGraph::get_or_create_row(
    const ExactGameState& state) {
  if (storage_.frozen) {
    BettingHistoryKey key = betting_history_key(state.betting);
    const auto betting_history = tables().betting_history_ids.find(key);
    if (betting_history == tables().betting_history_ids.end()) {
      return std::nullopt;
    }
    const BettingNodeId betting_node_id = tables().root_betting_node_id;
    if (betting_node_id == kInvalidBettingNodeId) {
      return std::nullopt;
    }
    return get_or_create_row(betting_history->second, betting_node_id, state);
  }

  const uint32_t betting_history_id =
      get_or_create_betting_history(state.betting);
  const BettingNodeId betting_node_id =
      get_or_create_root_betting_node(state.betting);
  return get_or_create_row(betting_history_id, betting_node_id, state);
}

uint32_t PublicStateGraph::get_or_create_betting_history(
    const BettingState& state) {
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
    const BettingState& child_state) {
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
    child_key = betting_abstraction_.make_action_child_history_key(
        parent_row, action_index, child_state);
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
    const BettingState& child_state) {
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
  if (row.betting_node_id >= tables.betting_nodes.size() ||
      betting_history_id >= tables.betting_history_rows.size()) {
    return;
  }

  const BettingNode& node = tables.betting_nodes[row.betting_node_id];
  if (node.action_count == 0) {
    return;
  }
  BettingHistoryRow& betting_history =
      tables.betting_history_rows[static_cast<size_t>(betting_history_id)];
  betting_history.action_count = node.action_count;
  for (uint8_t i = 0; i < node.action_count; ++i) {
    const size_t edge_index = static_cast<size_t>(node.action_begin) + i;
    betting_history.action_ids[static_cast<size_t>(i)] =
        tables.betting_edges[edge_index].action_id;
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
  if (row.betting_node_id >= tables().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node = tables().betting_nodes[row.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
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
    const ExactGameState& child_state) const {
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
    const Board& board,
    Callback&& callback) const {
  if constexpr (kCoarsePublicBuckets) {
    const auto& transitions = CoarseChanceTransitions(row.betting.street);
    const auto existing = transitions.find(row.public_bucket);
    if (existing == transitions.end()) {
      return false;
    }
    for (const CoarseChanceTransitionTemplate& transition : existing->second) {
      const ExactGameState parent{row.betting, transition.parent_board};
      const ExactGameState child_state = ApplyChance(parent, transition.cards);
      if (!callback(child_state, absl::Span<const CardId>(transition.cards))) {
        return false;
      }
    }
    return true;
  } else {
    return ForEachNextStreetDeal(
        row.betting.street, board, [&](absl::Span<const CardId> cards) {
      return callback(ApplyChance({row.betting, board}, cards), cards);
    });
  }
}

PublicStateGraph::PublicBucketId PublicStateGraph::chance_outcome_id(
    const ExactGameState& child_state) const {
  return card_abstraction_.public_bucket(child_state.betting.street,
                                         child_state.board);
}

std::optional<uint32_t> PublicStateGraph::find_or_cache_action_child(
    uint32_t parent_public_state_id,
    int action_index) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent_public_state_id];
  if (row.betting_node_id >= tables().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node = tables().betting_nodes[row.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error(
        "find_or_cache_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const uint32_t row_child_id = row.action_child_ids[action_slot];
  if (row_child_id != kInvalidPublicStateId) {
    return row_child_id;
  }

  const auto& betting_nodes = tables().betting_nodes;
  const auto& betting_edges = tables().betting_edges;
  if (row.betting_node_id >= betting_nodes.size()) {
    return std::nullopt;
  }

  const BettingNode& parent = betting_nodes[row.betting_node_id];
  if (action_index >= parent.action_count) {
    return std::nullopt;
  }

  const size_t edge_index =
      static_cast<size_t>(parent.action_begin) + action_slot;
  if (edge_index >= betting_edges.size()) {
    return std::nullopt;
  }
  const BettingNodeId child_node_id = betting_edges[edge_index].child;
  if (child_node_id == kInvalidBettingNodeId) {
    return std::nullopt;
  }

  const PublicStateKey child_key{
      child_node_id,
      row.public_bucket,
  };
  auto child_id = tables().public_state_ids.find(child_key);
  if (child_id == tables().public_state_ids.end()) {
    return std::nullopt;
  }

  if (!storage_.frozen) {
    mtables()
        .public_state_rows[parent_public_state_id]
        .action_child_ids[action_slot] = child_id->second;
  }
  return child_id->second;
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
    int action_index,
    const Board& parent_board) {
  const auto& public_rows = rows();
  if (parent_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& read_row = public_rows[parent_public_state_id];
  if (read_row.betting_node_id >= tables().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& parent = tables().betting_nodes[read_row.betting_node_id];
  if (action_index < 0 || action_index >= parent.action_count) {
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
  const size_t edge_index =
      static_cast<size_t>(parent.action_begin) + action_slot;
  const GameAction action = tables().betting_edges[edge_index].action;
  const BettingState child_betting = ApplyAction(read_row.betting, action);
  const ExactGameState child_state{child_betting, parent_board};
  const BettingNodeId child_betting_node_id =
      get_or_create_action_betting_child(read_row.betting_node_id,
                                         action_index);
  if (!SameBettingState(
          tables().betting_nodes[child_betting_node_id].state,
          child_betting)) {
    throw std::logic_error("action betting child state mismatch");
  }
  const uint32_t child_history_id = get_or_create_action_history_child(
      parent_history_id, action_index, child_betting);
  auto child_id =
      get_or_create_row(child_history_id, child_betting_node_id, child_state);
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
    const ExactGameState& child_state) {
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
    const auto& betting_nodes = tables().betting_nodes;
    if (row.betting_node_id >= betting_nodes.size()) {
      return std::nullopt;
    }

    const BettingNodeId child_node_id =
        betting_nodes[row.betting_node_id].chance_child;
    if (child_node_id == kInvalidBettingNodeId) {
      return std::nullopt;
    }

    const PublicStateKey child_public_key{
        child_node_id,
        outcome_id,
    };
    auto child_id = tables().public_state_ids.find(child_public_key);
    if (child_id == tables().public_state_ids.end()) {
      return std::nullopt;
    }

    if (!storage_.frozen) {
      mtables().public_chance_child_ids.emplace(
          transition_key, child_id->second);
    }
    return child_id->second;
  }

  return std::nullopt;
}

std::optional<uint32_t>
PublicStateGraph::get_or_create_chance_child(
    uint32_t parent_public_state_id,
    const ExactGameState& exact_child_state) {
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
  const uint32_t parent_history_id = parent_row.betting_history_id;
  const BettingNodeId child_betting_node_id =
      get_or_create_chance_betting_child(parent_row.betting_node_id,
                                         exact_child_state.betting);
  const uint32_t child_history_id = get_or_create_chance_history_child(
      parent_history_id, exact_child_state.betting);
  auto child_id = get_or_create_row(child_history_id, child_betting_node_id,
                                    exact_child_state);
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
    uint32_t root_id,
    const Board& root_board,
    int max_depth,
    std::vector<std::optional<Board>>& row_boards) {
  if (storage_.frozen) {
    return true;
  }
  if (root_id >= rows().size()) {
    return false;
  }

  row_boards.clear();
  row_boards.resize(rows().size());
  row_boards[root_id] = root_board;
  PublicStateBfs bfs(root_id, root_board, rows().size());
  while (std::optional<PublicStateBfs::Entry> maybe_entry = bfs.next()) {
    const PublicStateBfs::Entry entry = *maybe_entry;
    if (entry.node_id >= rows().size()) {
      return false;
    }
    // Copy before creating children; child creation can append to rows and
    // invalidate references.
    const PublicStateRow row = rows()[entry.node_id];
    if (row.is_terminal) {
      continue;
    }

    if (row.is_chance_node) {
      const bool complete = for_each_required_chance_transition(
          row, entry.board, [&](const ExactGameState& child_state,
                                absl::Span<const CardId>) {
            auto child_id = get_or_create_chance_child(entry.node_id, child_state);
            if (!child_id.has_value()) {
              return false;
            }
            if (row_boards.size() <= *child_id) {
              row_boards.resize(static_cast<size_t>(*child_id) + 1);
            }
            if (!row_boards[*child_id].has_value()) {
              row_boards[*child_id] = child_state.board;
            }
            bfs.enqueue_growing(*child_id, child_state.board, entry.depth);
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

    if (row.betting_node_id >= tables().betting_nodes.size()) {
      return false;
    }
    const int action_count =
        tables().betting_nodes[row.betting_node_id].action_count;
    for (int action_index = 0; action_index < action_count; ++action_index) {
      auto child_id = get_or_create_action_child(entry.node_id, action_index,
                                                 entry.board);
      if (!child_id.has_value()) {
        return false;
      }
      if (row_boards.size() <= *child_id) {
        row_boards.resize(static_cast<size_t>(*child_id) + 1);
      }
      if (!row_boards[*child_id].has_value()) {
        row_boards[*child_id] = entry.board;
      }
      bfs.enqueue_growing(*child_id, entry.board, entry.depth + 1);
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
    uint32_t child_id = kInvalidPublicStateId;

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
    if (child.child_id >= tables.public_state_rows.size()) {
      continue;
    }
    if (child.parent_id != current_parent_id) {
      current_parent_id = child.parent_id;
      tables.public_state_rows[current_parent_id].chance_child_offset =
          static_cast<uint32_t>(tables.chance_child_entries.size());
    }
    tables.chance_child_entries.push_back({
        child.outcome_id,
        child.child_id,
    });
    ++tables.public_state_rows[current_parent_id].chance_child_count;
  }
}

bool PublicStateGraph::validate_prebuilt_rows(
    uint32_t root_id,
    const Board& root_board,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& public_rows = rows();
  if (root_id >= public_rows.size()) {
    return false;
  }
  stats.betting_history_transition_prebuild_complete = true;
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  PublicStateBfs bfs(root_id, root_board, public_rows.size());

  auto valid_public_child = [&](uint32_t node_id) {
    return node_id != kInvalidPublicStateId &&
           node_id != kCappedPublicStateId &&
           node_id < public_rows.size();
  };
  auto enqueue_child = [&](uint32_t node_id, Board board, int depth) {
    return bfs.enqueue_existing(node_id, std::move(board), depth,
                                public_rows.size());
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
    if (entry.node_id >= public_rows.size()) {
      stats.betting_history_transition_prebuild_complete = false;
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const PublicStateRow& row = public_rows[entry.node_id];
    if (row.is_terminal) {
      continue;
    }

    if (row.is_chance_node) {
      const bool chance_complete = for_each_required_chance_transition(
          row, entry.board, [&](const ExactGameState& child_state,
                                absl::Span<const CardId>) {
        ++stats.prebuild_chance_transitions;
        const auto child = find_chance_child(entry.node_id, child_state);
        const uint32_t child_id = child.value_or(kInvalidPublicStateId);
        const bool valid_child = valid_public_child(child_id);
        if (!valid_child) {
          mark_missing_chance();
        }
        if (valid_child &&
            !enqueue_child(child_id, child_state.board, entry.depth)) {
          mark_missing_chance();
        }
        return true;
      });
      if (!chance_complete) {
        mark_missing_chance();
      }
      continue;
    }

    if (max_depth > 0 && entry.depth >= max_depth) {
      continue;
    }

    if (row.betting_node_id >= tables().betting_nodes.size()) {
      stats.action_transition_prebuild_complete = false;
      return false;
    }
    const BettingNode& node = tables().betting_nodes[row.betting_node_id];
    for (int action_index = 0; action_index < node.action_count;
         ++action_index) {
      ++stats.prebuild_action_transitions;
      const size_t action_slot = static_cast<size_t>(action_index);
      const uint32_t child_id = row.action_child_ids[action_slot];
      const bool valid_child = valid_public_child(child_id);
      if (!valid_child) {
        mark_missing_action();
      }
      if (valid_child && !enqueue_child(child_id, entry.board,
                                        entry.depth + 1)) {
        mark_missing_action();
      }
    }
  }

  return stats.betting_history_transition_prebuild_complete &&
         stats.action_transition_prebuild_complete &&
         stats.chance_transition_prebuild_complete;
}

}  // namespace poker
