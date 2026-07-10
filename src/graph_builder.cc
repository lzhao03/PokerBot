#include "src/graph_builder.h"

#include <algorithm>
#include <bit>
#include <cassert>
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
#include "src/game_rules.h"

namespace poker {
namespace {

class NodeBfs {
 public:
  struct Entry {
    NodeId node_id = 0;
    Board board;
    int depth = 0;
  };

  NodeBfs(NodeId root_id, Board root_board, size_t node_count) {
    queue_.reserve(1024);
    queued_.resize(node_count, 0);
    enqueue_growing(root_id, std::move(root_board), 0);
  }

  std::optional<Entry> next() {
    if (cursor_ >= queue_.size()) {
      return std::nullopt;
    }
    return queue_[cursor_++];
  }

  bool enqueue_existing(NodeId node_id,
                        Board board,
                        int depth,
                        size_t node_count) {
    if (node_id >= node_count || node_id >= queued_.size()) {
      return false;
    }
    enqueue_known_index(node_id, std::move(board), depth);
    return true;
  }

  void enqueue_growing(NodeId node_id, Board board, int depth) {
    if (node_id >= queued_.size()) {
      queued_.resize(static_cast<size_t>(node_id) + 1, 0);
    }
    enqueue_known_index(node_id, std::move(board), depth);
  }

 private:
  void enqueue_known_index(NodeId node_id, Board board, int depth) {
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
    BoardBucketId parent_bucket,
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
                                   BoardBucketId parent_bucket,
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

void ValidateBettingNode(const StrategyTables::BettingNode& node) {
  if (node.kind == StrategyTables::NodeKind::kDecision) {
    assert(IsPlayer(node.state.player_to_act));
  } else {
    assert(node.state.player_to_act == -1);
  }
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
  ValidateBettingNode(node);

  if (node.kind == StrategyTables::NodeKind::kDecision) {
    const auto menu = betting_abstraction_.actions_for_betting_node(state);
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
      tables.betting_nodes[node_id].state != state) {
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
  const BettingState child_state = ApplyAction(parent.state, action);
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
      tables.betting_nodes[existing_child].state != child_state) {
    throw std::logic_error("chance betting child state mismatch");
  }
  return existing_child;
}

GraphBuilder::NodeKey
GraphBuilder::node_key(
    BettingNodeId betting_node_id,
    StreetKind street,
    const Board& board) const {
  return {betting_node_id, board_bucket(street, board)};
}

std::optional<NodeId> GraphBuilder::find_node(
    BettingNodeId betting_node_id,
    StreetKind street,
    const Board& board) const {
  const auto existing =
      tables().node_ids.find(node_key(betting_node_id, street, board));
  if (existing == tables().node_ids.end()) {
    return std::nullopt;
  }
  return existing->second;
}

GraphBuilder::Node GraphBuilder::make_node(
    BettingNodeId betting_node_id,
    const ExactGameState& state) {
  Node graph_node;
  graph_node.betting_node_id = betting_node_id;
  graph_node.board_bucket = board_bucket(state.betting.street, state.board);

  StrategyTables& tables = mtables();
  const auto& nodes = tables.betting_nodes;
  if (betting_node_id >= nodes.size()) {
    throw std::logic_error("graph node betting node is invalid");
  }
  const BettingNode& node = nodes[betting_node_id];
  if (node.state != state.betting) {
    throw std::logic_error("graph node betting state mismatch");
  }
  if (node.action_count > 0) {
    graph_node.action_child_offset =
        static_cast<uint32_t>(tables.action_child_ids.size());
    const size_t new_size = tables.action_child_ids.size() +
                            node.action_count;
    tables.action_child_ids.resize(new_size, kInvalidNodeId);
  }

  return graph_node;
}

std::optional<NodeId> GraphBuilder::get_or_create_node(
    BettingNodeId betting_node_id,
    const ExactGameState& state) {
  if (std::optional<NodeId> existing =
          find_node(betting_node_id, state.betting.street, state.board)) {
    return existing;
  }

  if (storage_.frozen || node_limit_reached()) {
    return std::nullopt;
  }

  StrategyTables& tables = mtables();
  const NodeId node_id =
      static_cast<uint32_t>(tables.nodes.size());
  const auto [state_iter, inserted] = tables.node_ids.try_emplace(
      node_key(betting_node_id, state.betting.street, state.board),
      node_id);
  if (!inserted) {
    return state_iter->second;
  }
  tables.nodes.push_back(make_node(betting_node_id, state));
  return node_id;
}

std::optional<NodeId> GraphBuilder::get_or_create_node(
    const ExactGameState& state) {
  if (storage_.frozen) {
    const NodeId root_id = tables().root_node_id;
    if (root_id == kInvalidNodeId ||
        root_id >= tables().nodes.size()) {
      return std::nullopt;
    }
    return root_id;
  }

  const BettingNodeId betting_node_id =
      get_or_create_root_betting_node(state.betting);
  auto root_id = get_or_create_node(betting_node_id, state);
  if (root_id.has_value()) {
    mtables().root_node_id = *root_id;
  }
  return root_id;
}

template <typename Callback>
bool GraphBuilder::for_each_required_chance_transition(
    const Node& graph_node,
    const Board& board,
    Callback&& callback) const {
  if (graph_node.betting_node_id >= tables().betting_nodes.size()) {
    return false;
  }
  const BettingState& betting =
      tables().betting_nodes[graph_node.betting_node_id].state;
  if constexpr (kCoarsePublicBuckets) {
    return ForEachCoarseChanceTransition(
        betting.street, graph_node.board_bucket, betting,
        std::forward<Callback>(callback));
  } else {
    return ForEachNextStreetDeal(
        betting.street, board, [&](absl::Span<const CardId> cards) {
      return callback(ApplyChance({betting, board}, cards), cards);
    });
  }
}

GraphBuilder::BoardBucketId GraphBuilder::chance_outcome_id(
    const ExactGameState& child_state) const {
  return board_bucket(child_state.betting.street, child_state.board);
}

std::optional<NodeId> GraphBuilder::find_or_cache_action_child(
    NodeId parent_node_id,
    int action_index) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = graph_nodes[parent_node_id];
  if (graph_node.betting_node_id >= tables().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node = tables().betting_nodes[graph_node.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error(
        "find_or_cache_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const size_t child_slot = static_cast<size_t>(graph_node.action_child_offset) +
                            action_slot;
  if (child_slot >= tables().action_child_ids.size()) {
    return std::nullopt;
  }
  const NodeId cached_child_id = tables().action_child_ids[child_slot];
  if (cached_child_id != kInvalidNodeId) {
    return cached_child_id;
  }

  const auto& betting_nodes = tables().betting_nodes;
  const auto& betting_edges = tables().betting_edges;
  if (graph_node.betting_node_id >= betting_nodes.size()) {
    return std::nullopt;
  }

  const BettingNode& parent = betting_nodes[graph_node.betting_node_id];
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

  const NodeKey child_key{
      child_node_id,
      graph_node.board_bucket,
  };
  auto child_id = tables().node_ids.find(child_key);
  if (child_id == tables().node_ids.end()) {
    return std::nullopt;
  }

  if (!storage_.frozen) {
    mtables().action_child_ids[child_slot] = child_id->second;
  }
  return child_id->second;
}

bool GraphBuilder::node_limit_reached() const {
  return config_.max_public_states > 0 &&
         static_cast<int>(nodes().size()) >= config_.max_public_states;
}

bool GraphBuilder::can_insert_node() const {
  return !storage_.frozen && !node_limit_reached();
}

std::optional<NodeId>
GraphBuilder::get_or_create_action_child(
    NodeId parent_node_id,
    int action_index,
    const Board& parent_board) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& parent_node = graph_nodes[parent_node_id];
  if (parent_node.betting_node_id >= tables().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& parent = tables().betting_nodes[parent_node.betting_node_id];
  if (action_index < 0 || action_index >= parent.action_count) {
    throw std::logic_error(
        "get_or_create_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const size_t child_slot = static_cast<size_t>(parent_node.action_child_offset) +
                            action_slot;
  if (std::optional<NodeId> existing_child_id =
          find_or_cache_action_child(parent_node_id, action_index)) {
    stats_.record_transition_hit();
    if (*existing_child_id == kCappedNodeId) {
      return std::nullopt;
    }
    return existing_child_id;
  }

  if (!can_insert_node()) {
    if (!storage_.frozen) {
      mtables().action_child_ids[child_slot] = kCappedNodeId;
    }
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const size_t edge_index =
      static_cast<size_t>(parent.action_begin) + action_slot;
  const GameAction action = tables().betting_edges[edge_index].action;
  const BettingState child_betting = ApplyAction(parent.state, action);
  const ExactGameState child_state{child_betting, parent_board};
  const BettingNodeId child_betting_node_id =
      get_or_create_action_betting_child(parent_node.betting_node_id,
                                         action_index);
  if (tables().betting_nodes[child_betting_node_id].state != child_betting) {
    throw std::logic_error("action betting child state mismatch");
  }
  auto child_id = get_or_create_node(child_betting_node_id, child_state);
  if (!child_id.has_value()) {
    mtables().action_child_ids[child_slot] = kCappedNodeId;
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mtables().action_child_ids[child_slot] = *child_id;
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

std::optional<NodeId> GraphBuilder::find_or_cache_chance_child(
    NodeId parent_node_id,
    const ExactGameState& child_state) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = graph_nodes[parent_node_id];
  const BoardBucketId outcome_id = chance_outcome_id(child_state);
  const ChanceTransitionKey transition_key{parent_node_id, outcome_id};
  auto existing = tables().public_chance_child_ids.find(transition_key);
  if (existing != tables().public_chance_child_ids.end()) {
    return existing->second;
  }

  if constexpr (kCoarsePublicBuckets) {
    const auto& betting_nodes = tables().betting_nodes;
    if (graph_node.betting_node_id >= betting_nodes.size()) {
      return std::nullopt;
    }

    const BettingNodeId child_node_id =
        betting_nodes[graph_node.betting_node_id].chance_child;
    if (child_node_id == kInvalidBettingNodeId) {
      return std::nullopt;
    }

    const NodeKey child_key{
        child_node_id,
        outcome_id,
    };
    auto child_id = tables().node_ids.find(child_key);
    if (child_id == tables().node_ids.end()) {
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

std::optional<NodeId>
GraphBuilder::get_or_create_chance_child(
    NodeId parent_node_id,
    const ExactGameState& exact_child_state) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const BoardBucketId outcome_id = chance_outcome_id(exact_child_state);
  if (std::optional<NodeId> existing_child_id =
          find_or_cache_chance_child(parent_node_id,
                                     exact_child_state)) {
    stats_.record_transition_hit();
    return *existing_child_id;
  }

  if (!can_insert_node()) {
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const Node& parent_node = graph_nodes[parent_node_id];
  const BettingNodeId child_betting_node_id =
      get_or_create_chance_betting_child(parent_node.betting_node_id,
                                         exact_child_state.betting);
  auto child_id = get_or_create_node(child_betting_node_id, exact_child_state);
  if (!child_id.has_value()) {
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mtables().public_chance_child_ids.emplace(
      ChanceTransitionKey{parent_node_id, outcome_id}, *child_id);
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

bool GraphBuilder::prebuild_reachable_nodes(
    NodeId root_id,
    const Board& root_board,
    int max_depth,
    std::vector<std::optional<Board>>& node_boards) {
  if (storage_.frozen) {
    return true;
  }
  if (root_id >= nodes().size()) {
    return false;
  }

  node_boards.clear();
  node_boards.resize(nodes().size());
  node_boards[root_id] = root_board;
  NodeBfs bfs(root_id, root_board, nodes().size());
  while (std::optional<NodeBfs::Entry> maybe_entry = bfs.next()) {
    const NodeBfs::Entry entry = *maybe_entry;
    if (entry.node_id >= nodes().size()) {
      return false;
    }
    // Copy before creating children; child creation can append to nodes and
    // invalidate references.
    const Node graph_node = nodes()[entry.node_id];
    if (graph_node.betting_node_id >= tables().betting_nodes.size()) {
      return false;
    }
    const BettingNode node = tables().betting_nodes[graph_node.betting_node_id];
    if (node.kind == StrategyTables::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == StrategyTables::NodeKind::kChance) {
      const bool complete = for_each_required_chance_transition(
          graph_node, entry.board, [&](const ExactGameState& child_state,
                                absl::Span<const CardId>) {
            auto child_id = get_or_create_chance_child(entry.node_id, child_state);
            if (!child_id.has_value()) {
              return false;
            }
            if (node_boards.size() <= *child_id) {
              node_boards.resize(static_cast<size_t>(*child_id) + 1);
            }
            if (!node_boards[*child_id].has_value()) {
              node_boards[*child_id] = child_state.board;
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
      if (node_boards.size() <= *child_id) {
        node_boards.resize(static_cast<size_t>(*child_id) + 1);
      }
      if (!node_boards[*child_id].has_value()) {
        node_boards[*child_id] = entry.board;
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
    BoardBucketId outcome_id = 0;
    NodeId child_id = kInvalidNodeId;

    bool operator<(const PendingChanceChild& other) const {
      return std::tie(parent_id, outcome_id) <
             std::tie(other.parent_id, other.outcome_id);
    }
  };

  std::vector<PendingChanceChild> pending;
  pending.reserve(tables.public_chance_child_ids.size());
  for (const auto& [transition_key, child_id] :
       tables.public_chance_child_ids) {
    const uint32_t parent_id = transition_key.parent_node_id;
    if (parent_id >= tables.nodes.size()) {
      continue;
    }
    pending.push_back({
        parent_id,
        transition_key.outcome_id,
        child_id,
    });
  }

  std::sort(pending.begin(), pending.end());

  for (Node& graph_node : tables.nodes) {
    graph_node.chance_child_offset = 0;
    graph_node.chance_child_count = 0;
  }
  tables.chance_child_entries.clear();
  tables.chance_child_entries.reserve(pending.size());
  NodeId current_parent_id = kInvalidNodeId;
  for (const PendingChanceChild& child : pending) {
    if (child.child_id >= tables.nodes.size()) {
      continue;
    }
    if (child.parent_id != current_parent_id) {
      current_parent_id = child.parent_id;
      tables.nodes[current_parent_id].chance_child_offset =
          static_cast<uint32_t>(tables.chance_child_entries.size());
    }
    tables.chance_child_entries.push_back({
        child.outcome_id,
        child.child_id,
    });
    ++tables.nodes[current_parent_id].chance_child_count;
  }
}

bool GraphBuilder::validate_prebuilt_nodes(
    NodeId root_id,
    const Board& root_board,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& graph_nodes = nodes();
  if (root_id >= graph_nodes.size()) {
    return false;
  }
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  NodeBfs bfs(root_id, root_board, graph_nodes.size());

  auto valid_child_id = [&](NodeId node_id) {
    return node_id != kInvalidNodeId &&
           node_id != kCappedNodeId &&
           node_id < graph_nodes.size();
  };
  auto enqueue_child = [&](NodeId node_id, Board board, int depth) {
    return bfs.enqueue_existing(node_id, std::move(board), depth,
                                graph_nodes.size());
  };
  auto mark_missing_action = [&] {
    ++stats.missing_action_transitions;
    stats.action_transition_prebuild_complete = false;
  };
  auto mark_missing_chance = [&] {
    ++stats.missing_chance_transitions;
    stats.chance_transition_prebuild_complete = false;
  };

  while (std::optional<NodeBfs::Entry> maybe_entry = bfs.next()) {
    const NodeBfs::Entry entry = *maybe_entry;
    if (entry.node_id >= graph_nodes.size()) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const Node& graph_node = graph_nodes[entry.node_id];
    if (graph_node.betting_node_id >= tables().betting_nodes.size()) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const BettingNode& node = tables().betting_nodes[graph_node.betting_node_id];
    ValidateBettingNode(node);
    if (node.kind == StrategyTables::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == StrategyTables::NodeKind::kChance) {
      const bool chance_complete = for_each_required_chance_transition(
          graph_node, entry.board, [&](const ExactGameState& child_state,
                                absl::Span<const CardId>) {
        ++stats.prebuild_chance_transitions;
        const auto child = tables().chance_child(
            entry.node_id, chance_outcome_id(child_state));
        const NodeId child_id = child.value_or(kInvalidNodeId);
        const bool valid_child = valid_child_id(child_id);
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
      const NodeId child_id =
          tables().action_child(entry.node_id, action_index)
              .value_or(kInvalidNodeId);
      const bool valid_child = valid_child_id(child_id);
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
