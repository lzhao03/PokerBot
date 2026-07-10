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
#include "absl/container/inlined_vector.h"
#include "src/build_flags.h"
#include "src/card_utils.h"
#include "src/coarse_chance_transitions.h"
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

template <size_t N, typename Callback>
bool ForEachCoarseChanceTransition(
    const std::array<CoarseChanceTransition, N>& transitions,
    PublicBucketId parent_bucket,
    const BettingState& betting,
    Callback&& callback) {
  for (const CoarseChanceTransition& transition : transitions) {
    if (transition.parent_bucket != parent_bucket) {
      continue;
    }
    Board parent;
    for (uint8_t i = 0; i < transition.parent_count; ++i) {
      parent.add(transition.parent_cards[static_cast<size_t>(i)]);
    }
    const absl::Span<const CardId> cards(
        transition.cards.data(), transition.card_count);
    if (!callback(ApplyChance({betting, parent}, cards), cards)) {
      return false;
    }
  }
  return true;
}

template <typename Callback>
bool ForEachCoarseChanceTransition(StreetKind street,
                                   PublicBucketId parent_bucket,
                                   const BettingState& betting,
                                   Callback&& callback) {
  switch (street) {
    case StreetKind::kPreflop:
      return ForEachCoarseChanceTransition(kPreflopTextureTransitions,
                                           parent_bucket, betting,
                                           std::forward<Callback>(callback));
    case StreetKind::kFlop:
      return ForEachCoarseChanceTransition(kFlopTextureTransitions,
                                           parent_bucket, betting,
                                           std::forward<Callback>(callback));
    case StreetKind::kTurn:
      return ForEachCoarseChanceTransition(kTurnTextureTransitions,
                                           parent_bucket, betting,
                                           std::forward<Callback>(callback));
    case StreetKind::kRiver:
      return true;
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
         first.committed == second.committed &&
         first.street == second.street &&
         first.player_to_act == second.player_to_act &&
         first.folded_player == second.folded_player &&
         first.pending_action_mask == second.pending_action_mask;
}

}  // namespace

GraphBuilder::GraphBuilder(
    const SolverConfig& config,
    SolverStorage& storage,
    const BettingAbstraction& betting_abstraction,
    TraversalStats& stats)
    : config_(config),
      storage_(storage),
      betting_abstraction_(betting_abstraction),
      stats_(stats) {}

GraphBuilder::BettingNodeId GraphBuilder::append_betting_node(
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
          kInvalidBettingNodeId,
      });
    }
  }

  const BettingNodeId node_id =
      static_cast<BettingNodeId>(tables.betting_nodes.size());
  tables.betting_nodes.push_back(std::move(node));
  return node_id;
}

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_root_betting_node(const BettingState& state) {
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

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_action_betting_child(
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
  const BettingState child_state =
      ApplyLegalActionUnchecked(parent.state, action);
  const BettingNodeId child_node_id = append_betting_node(child_state);
  tables.betting_edges[edge_index].child = child_node_id;
  return child_node_id;
}

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_chance_betting_child(
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

GraphBuilder::PublicStateKey
GraphBuilder::row_key(
    BettingNodeId betting_node_id,
    StreetKind street,
    const Board& board) const {
  return {betting_node_id, public_bucket(street, board)};
}

std::optional<uint32_t> GraphBuilder::find_row(
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

GraphBuilder::PublicStateRow GraphBuilder::make_row(
    BettingNodeId betting_node_id,
    const ExactGameState& state) {
  PublicStateRow row;
  row.betting_node_id = betting_node_id;
  row.public_bucket = public_bucket(state.betting.street, state.board);

  StrategyTables& tables = mtables();
  const auto& nodes = tables.betting_nodes;
  if (betting_node_id >= nodes.size()) {
    throw std::logic_error("public row betting node is invalid");
  }
  const BettingNode& node = nodes[betting_node_id];
  if (!SameBettingState(node.state, state.betting)) {
    throw std::logic_error("public row betting node state mismatch");
  }
  if (node.action_count > 0) {
    row.action_child_offset =
        static_cast<uint32_t>(tables.action_child_ids.size());
    const size_t new_size = tables.action_child_ids.size() +
                            node.action_count;
    tables.action_child_ids.resize(new_size, kInvalidPublicStateId);
  }

  return row;
}

std::optional<uint32_t> GraphBuilder::get_or_create_row(
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
  tables.public_state_rows.push_back(make_row(betting_node_id, state));
  return public_state_id;
}

std::optional<uint32_t> GraphBuilder::get_or_create_row(
    const ExactGameState& state) {
  if (storage_.frozen) {
    const uint32_t root_id = tables().root_public_state_id;
    if (root_id == kInvalidPublicStateId ||
        root_id >= tables().public_state_rows.size()) {
      return std::nullopt;
    }
    return root_id;
  }

  const BettingNodeId betting_node_id =
      get_or_create_root_betting_node(state.betting);
  auto root_id = get_or_create_row(betting_node_id, state);
  if (root_id.has_value()) {
    mtables().root_public_state_id = *root_id;
  }
  return root_id;
}

template <typename Callback>
bool GraphBuilder::for_each_required_chance_transition(
    const PublicStateRow& row,
    const Board& board,
    Callback&& callback) const {
  if (row.betting_node_id >= tables().betting_nodes.size()) {
    return false;
  }
  const BettingState& betting =
      tables().betting_nodes[row.betting_node_id].state;
  if constexpr (kCoarsePublicBuckets) {
    return ForEachCoarseChanceTransition(
        betting.street, row.public_bucket, betting,
        std::forward<Callback>(callback));
  } else {
    return ForEachNextStreetDeal(
        betting.street, board, [&](absl::Span<const CardId> cards) {
      return callback(ApplyChance({betting, board}, cards), cards);
    });
  }
}

GraphBuilder::PublicBucketId GraphBuilder::chance_outcome_id(
    const ExactGameState& child_state) const {
  return public_bucket(child_state.betting.street, child_state.board);
}

std::optional<uint32_t> GraphBuilder::find_or_cache_action_child(
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
  const size_t child_slot = static_cast<size_t>(row.action_child_offset) +
                            action_slot;
  if (child_slot >= tables().action_child_ids.size()) {
    return std::nullopt;
  }
  const uint32_t row_child_id = tables().action_child_ids[child_slot];
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
    mtables().action_child_ids[child_slot] = child_id->second;
  }
  return child_id->second;
}

bool GraphBuilder::row_limit_reached() const {
  return config_.max_public_states > 0 &&
         static_cast<int>(rows().size()) >= config_.max_public_states;
}

bool GraphBuilder::can_insert_row() const {
  return !storage_.frozen && !row_limit_reached();
}

std::optional<uint32_t>
GraphBuilder::get_or_create_action_child(
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
  const size_t child_slot = static_cast<size_t>(read_row.action_child_offset) +
                            action_slot;
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
      mtables().action_child_ids[child_slot] = kCappedPublicStateId;
    }
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const size_t edge_index =
      static_cast<size_t>(parent.action_begin) + action_slot;
  const GameAction action = tables().betting_edges[edge_index].action;
  const BettingState child_betting =
      ApplyLegalActionUnchecked(parent.state, action);
  const ExactGameState child_state{child_betting, parent_board};
  const BettingNodeId child_betting_node_id =
      get_or_create_action_betting_child(read_row.betting_node_id,
                                         action_index);
  if (!SameBettingState(
          tables().betting_nodes[child_betting_node_id].state,
          child_betting)) {
    throw std::logic_error("action betting child state mismatch");
  }
  auto child_id = get_or_create_row(child_betting_node_id, child_state);
  if (!child_id.has_value()) {
    mtables().action_child_ids[child_slot] = kCappedPublicStateId;
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mtables().action_child_ids[child_slot] = *child_id;
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

std::optional<uint32_t> GraphBuilder::find_or_cache_chance_child(
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
GraphBuilder::get_or_create_chance_child(
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
  const BettingNodeId child_betting_node_id =
      get_or_create_chance_betting_child(parent_row.betting_node_id,
                                         exact_child_state.betting);
  auto child_id = get_or_create_row(child_betting_node_id, exact_child_state);
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

bool GraphBuilder::prebuild_reachable_rows(
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
    if (row.betting_node_id >= tables().betting_nodes.size()) {
      return false;
    }
    const BettingNode node = tables().betting_nodes[row.betting_node_id];
    if (node.kind == StrategyTables::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == StrategyTables::NodeKind::kChance) {
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

    const int action_count = node.action_count;
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

void GraphBuilder::rebuild_chance_child_entries() {
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

bool GraphBuilder::validate_prebuilt_rows(
    uint32_t root_id,
    const Board& root_board,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& public_rows = rows();
  if (root_id >= public_rows.size()) {
    return false;
  }
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
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const PublicStateRow& row = public_rows[entry.node_id];
    if (row.betting_node_id >= tables().betting_nodes.size()) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const BettingNode& node = tables().betting_nodes[row.betting_node_id];
    if (node.kind == StrategyTables::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == StrategyTables::NodeKind::kChance) {
      const bool chance_complete = for_each_required_chance_transition(
          row, entry.board, [&](const ExactGameState& child_state,
                                absl::Span<const CardId>) {
        ++stats.prebuild_chance_transitions;
        const auto child = tables().chance_child(
            entry.node_id, chance_outcome_id(child_state));
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

    for (int action_index = 0; action_index < node.action_count;
         ++action_index) {
      ++stats.prebuild_action_transitions;
      const uint32_t child_id =
          tables().action_child(entry.node_id, action_index)
              .value_or(kInvalidPublicStateId);
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

  return stats.action_transition_prebuild_complete &&
         stats.chance_transition_prebuild_complete;
}

}  // namespace poker
