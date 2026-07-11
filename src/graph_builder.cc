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
    BoardRunout board = BoardRunout::Preflop();
    int depth = 0;
  };

  NodeBfs(NodeId root_id, BoardRunout root_board, size_t node_count) {
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
                        BoardRunout board,
                        int depth,
                        size_t node_count) {
    if (node_id >= node_count || node_id >= queued_.size()) {
      return false;
    }
    enqueue_known_index(node_id, std::move(board), depth);
    return true;
  }

  void enqueue_growing(NodeId node_id, BoardRunout board, int depth) {
    if (node_id >= queued_.size()) {
      queued_.resize(static_cast<size_t>(node_id) + 1, 0);
    }
    enqueue_known_index(node_id, std::move(board), depth);
  }

 private:
  void enqueue_known_index(NodeId node_id, BoardRunout board, int depth) {
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
                           const BoardRunout& board,
                           Callback callback) {
  const int remaining_board_slots =
      std::max(0, kMaxBoardCards - static_cast<int>(board.count()));
  const int count =
      std::min(CardsForNextStreet(street), remaining_board_slots);
  if (count <= 0) {
    return callback(absl::Span<const CardId>());
  }

  const int available_cards =
      kDeckCardCount -
      static_cast<int>(std::popcount(board.mask()));
  if (available_cards < count) {
    throw std::runtime_error("Not enough cards to enumerate next street");
  }
  return ForEachCardCombination(count, board.mask(), callback);
}

[[maybe_unused]] BoardRunout RunoutFromCards(
    absl::Span<const CardId> cards) {
  BoardRunout runout = BoardRunout::Preflop();
  if (cards.empty()) {
    return runout;
  }
  if (cards.size() < 3 || cards.size() > kMaxBoardCards) {
    throw std::logic_error("stored board has an invalid card count");
  }
  runout.deal_flop(cards.first(3));
  if (cards.size() >= 4) {
    runout.deal_turn(cards[3]);
  }
  if (cards.size() == 5) {
    runout.deal_river(cards[4]);
  }
  return runout;
}

template <size_t N, typename Callback>
bool ForEachCoarseChanceTransition(
    const std::array<CoarseChanceTransition, N>& transitions,
    BoardBucketId parent_bucket,
    const BettingState& betting,
    const BettingRules& rules,
    Callback&& callback) {
  for (const CoarseChanceTransition& transition : transitions) {
    if (transition.parent_bucket != parent_bucket) {
      continue;
    }
    const absl::Span<const CardId> parent_cards(
        transition.parent_cards.data(), transition.parent_count);
    const BoardRunout parent = RunoutFromCards(parent_cards);
    const absl::Span<const CardId> cards(
        transition.cards.data(), transition.card_count);
    if (!callback(ApplyChance({betting, parent}, cards, rules), cards)) {
      return false;
    }
  }
  return true;
}

template <typename Callback>
bool ForEachCoarseChanceTransition(StreetKind street,
                                   PublicStreetObservation parent_observation,
                                   const BettingState& betting,
                                   const BettingRules& rules,
                                   Callback&& callback) {
  BoardBucketId parent_bucket = 0;
  if (street != StreetKind::kPreflop) {
    const auto street_id = static_cast<BoardBucketId>(street);
    const auto stride = kCoarsePublicStreetObservationCount;
    parent_bucket = 1 + street_id * stride + parent_observation.value;
  }
  switch (street) {
    case StreetKind::kPreflop:
      return ForEachCoarseChanceTransition(kPreflopTextureTransitions,
                                           parent_bucket, betting, rules,
                                           std::forward<Callback>(callback));
    case StreetKind::kFlop:
      return ForEachCoarseChanceTransition(kFlopTextureTransitions,
                                           parent_bucket, betting, rules,
                                           std::forward<Callback>(callback));
    case StreetKind::kTurn:
      return ForEachCoarseChanceTransition(kTurnTextureTransitions,
                                           parent_bucket, betting, rules,
                                           std::forward<Callback>(callback));
    case StreetKind::kRiver:
      return true;
  }
}

PublicGraph::NodeKind BettingNodeKind(const BettingState& state) {
  const bool betting_over = IsBettingRoundOver(state);
  if (state.folded_player >= 0 ||
      (betting_over && state.street == StreetKind::kRiver)) {
    return PublicGraph::NodeKind::kTerminal;
  }
  if (betting_over) {
    return PublicGraph::NodeKind::kChance;
  }
  return PublicGraph::NodeKind::kDecision;
}

void ValidateBettingNode(const PublicGraph::BettingNode& node) {
  if (node.kind == PublicGraph::NodeKind::kDecision) {
    assert(IsPlayer(node.state.player_to_act));
  } else {
    assert(node.state.player_to_act == -1);
  }
}

}  // namespace

GraphBuilder::GraphBuilder(
    const SolverConfig& config,
    const BettingRules& rules,
    SolverStorage& storage,
    const BettingAbstraction& betting_abstraction,
    TraversalStats& stats)
    : config_(config),
      rules_(rules),
      storage_(storage),
      betting_abstraction_(betting_abstraction),
      stats_(stats) {
  if (config_.max_public_states <= 0) {
    return;
  }
  const size_t node_cap = static_cast<size_t>(config_.max_public_states);
  build_state_.node_ids.reserve(node_cap);
  build_state_.chance_children.reserve(node_cap);
  PublicGraph& public_graph = mgraph();
  public_graph.nodes.reserve(node_cap);
  public_graph.action_children.reserve(node_cap * 4);
  public_graph.chance_children.reserve(node_cap);
  public_graph.betting_nodes.reserve(node_cap);
  public_graph.betting_edges.reserve(node_cap);
}

GraphBuilder::BettingNodeId GraphBuilder::append_betting_node(
    const BettingState& state) {
  PublicGraph& graph = mgraph();
  BettingNode node;
  node.state = state;
  node.kind = BettingNodeKind(state);
  ValidateBettingNode(node);

  if (node.kind == PublicGraph::NodeKind::kDecision) {
    const auto menu = betting_abstraction_.actions_for_betting_node(state);
    node.action_begin = static_cast<uint32_t>(graph.betting_edges.size());
    node.action_count = menu.count;
    for (uint8_t i = 0; i < menu.count; ++i) {
      const GameAction action = menu.actions[static_cast<size_t>(i)];
      graph.betting_edges.push_back({
          action,
          kInvalidBettingNodeId,
      });
    }
  }

  const BettingNodeId node_id =
      static_cast<BettingNodeId>(graph.betting_nodes.size());
  graph.betting_nodes.push_back(std::move(node));
  return node_id;
}

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_root_betting_node(const BettingState& state) {
  PublicGraph& graph = mgraph();
  if (graph.root_betting_node == kInvalidBettingNodeId) {
    graph.root_betting_node = append_betting_node(state);
    return graph.root_betting_node;
  }

  const BettingNodeId node_id = graph.root_betting_node;
  if (node_id >= graph.betting_nodes.size() ||
      graph.betting_nodes[node_id].state != state) {
    throw std::logic_error("root betting node does not match root state");
  }
  return node_id;
}

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_action_betting_child(
    BettingNodeId parent_node_id,
    int action_index) {
  PublicGraph& graph = mgraph();
  if (parent_node_id >= graph.betting_nodes.size()) {
    throw std::logic_error("action betting parent node is invalid");
  }
  const BettingNode& parent = graph.betting_nodes[parent_node_id];
  if (action_index < 0 || action_index >= parent.action_count) {
    throw std::logic_error("action betting child index out of range");
  }

  const size_t edge_index = static_cast<size_t>(parent.action_begin) +
                            static_cast<size_t>(action_index);
  const BettingNodeId existing_child = graph.betting_edges[edge_index].child;
  if (existing_child != kInvalidBettingNodeId) {
    return existing_child;
  }

  const GameAction action = graph.betting_edges[edge_index].action;
  const BettingState child_state = ApplyAction(parent.state, action);
  const BettingNodeId child_node_id = append_betting_node(child_state);
  graph.betting_edges[edge_index].child = child_node_id;
  return child_node_id;
}

GraphBuilder::BettingNodeId
GraphBuilder::get_or_create_chance_betting_child(
    BettingNodeId parent_node_id,
    const BettingState& child_state) {
  PublicGraph& graph = mgraph();
  if (parent_node_id >= graph.betting_nodes.size()) {
    throw std::logic_error("chance betting parent node is invalid");
  }
  const BettingNodeId existing_child =
      graph.betting_nodes[parent_node_id].chance_child;
  if (existing_child == kInvalidBettingNodeId) {
    const BettingNodeId child_node_id = append_betting_node(child_state);
    graph.betting_nodes[parent_node_id].chance_child = child_node_id;
    return child_node_id;
  }

  if (existing_child >= graph.betting_nodes.size() ||
      graph.betting_nodes[existing_child].state != child_state) {
    throw std::logic_error("chance betting child state mismatch");
  }
  return existing_child;
}

PublicNodeKey
GraphBuilder::node_key(
    BettingNodeId betting_node_id,
    PublicObservationId public_observation) const {
  return {betting_node_id, public_observation};
}

std::optional<NodeId> GraphBuilder::find_node(
    BettingNodeId betting_node_id,
    PublicObservationId public_observation) const {
  const auto existing =
      build_state_.node_ids.find(node_key(betting_node_id, public_observation));
  if (existing == build_state_.node_ids.end()) {
    return std::nullopt;
  }
  return existing->second;
}

GraphBuilder::Node GraphBuilder::make_node(
    BettingNodeId betting_node_id,
    const ExactPublicState& state,
    PublicObservationId public_observation) {
  Node graph_node;
  graph_node.betting_node_id = betting_node_id;
  graph_node.public_observation = public_observation;

  PublicGraph& graph = mgraph();
  const auto& nodes = graph.betting_nodes;
  if (betting_node_id >= nodes.size()) {
    throw std::logic_error("graph node betting node is invalid");
  }
  const BettingNode& node = nodes[betting_node_id];
  if (node.state != state.betting) {
    throw std::logic_error("graph node betting state mismatch");
  }
  if (node.action_count > 0) {
    graph_node.action_child_begin =
        static_cast<uint32_t>(graph.action_children.size());
    const size_t new_size = graph.action_children.size() +
                            node.action_count;
    graph.action_children.resize(new_size, kInvalidNodeId);
  }

  return graph_node;
}

std::optional<NodeId> GraphBuilder::get_or_create_node(
    BettingNodeId betting_node_id,
    const ExactPublicState& state,
    PublicObservationId public_observation) {
  if (std::optional<NodeId> existing =
          find_node(betting_node_id, public_observation)) {
    return existing;
  }

  if (storage_.is_frozen() || node_limit_reached()) {
    return std::nullopt;
  }

  PublicGraph& graph = mgraph();
  const NodeId node_id =
      static_cast<uint32_t>(graph.nodes.size());
  const auto [state_iter, inserted] = build_state_.node_ids.try_emplace(
      node_key(betting_node_id, public_observation), node_id);
  if (!inserted) {
    return state_iter->second;
  }
  graph.nodes.push_back(make_node(betting_node_id, state,
                                  public_observation));
  return node_id;
}

std::optional<NodeId> GraphBuilder::get_or_create_node(
    const ExactPublicState& state) {
  if (storage_.is_frozen()) {
    const NodeId root_id = graph().root;
    if (root_id == kInvalidNodeId ||
        root_id >= graph().nodes.size()) {
      return std::nullopt;
    }
    return root_id;
  }

  const BettingNodeId betting_node_id =
      get_or_create_root_betting_node(state.betting);
  const PublicObservationId public_observation =
      public_observation_id(state.betting.street, state.board);
  auto root_id = get_or_create_node(betting_node_id, state,
                                    public_observation);
  if (root_id.has_value()) {
    mgraph().root = *root_id;
  }
  return root_id;
}

template <typename Callback>
bool GraphBuilder::for_each_required_chance_transition(
    const Node& graph_node,
    const BoardRunout& board,
    Callback&& callback) const {
  if (graph_node.betting_node_id >= graph().betting_nodes.size()) {
    return false;
  }
  const BettingState& betting =
      graph().betting_nodes[graph_node.betting_node_id].state;
  if constexpr (kCoarsePublicBuckets) {
    const PublicStreetObservation current_observation =
        current_public_street_observation(graph_node.public_observation,
                                          betting.street);
    return ForEachCoarseChanceTransition(
        betting.street, current_observation, betting, rules_,
        std::forward<Callback>(callback));
  } else {
    return ForEachNextStreetDeal(
        betting.street, board, [&](absl::Span<const CardId> cards) {
      return callback(ApplyChance({betting, board}, cards, rules_), cards);
    });
  }
}

std::optional<NodeId> GraphBuilder::find_or_cache_action_child(
    NodeId parent_node_id,
    int action_index) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = graph_nodes[parent_node_id];
  if (graph_node.betting_node_id >= graph().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node =
      graph().betting_nodes[graph_node.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error(
        "find_or_cache_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const size_t child_slot = static_cast<size_t>(graph_node.action_child_begin) +
                            action_slot;
  if (child_slot >= graph().action_children.size()) {
    return std::nullopt;
  }
  const NodeId cached_child_id = graph().action_children[child_slot];
  if (cached_child_id != kInvalidNodeId) {
    return cached_child_id;
  }

  const auto& betting_nodes = graph().betting_nodes;
  const auto& betting_edges = graph().betting_edges;
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

  const PublicNodeKey child_key{
      child_node_id,
      graph_node.public_observation,
  };
  auto child_id = build_state_.node_ids.find(child_key);
  if (child_id == build_state_.node_ids.end()) {
    return std::nullopt;
  }

  if (!storage_.is_frozen()) {
    mgraph().action_children[child_slot] = child_id->second;
  }
  return child_id->second;
}

bool GraphBuilder::node_limit_reached() const {
  return config_.max_public_states > 0 &&
         static_cast<int>(nodes().size()) >= config_.max_public_states;
}

bool GraphBuilder::can_insert_node() const {
  return !storage_.is_frozen() && !node_limit_reached();
}

std::optional<NodeId>
GraphBuilder::get_or_create_action_child(
    NodeId parent_node_id,
    int action_index,
    const BoardRunout& parent_board) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& parent_node = graph_nodes[parent_node_id];
  if (parent_node.betting_node_id >= graph().betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& parent =
      graph().betting_nodes[parent_node.betting_node_id];
  if (action_index < 0 || action_index >= parent.action_count) {
    throw std::logic_error(
        "get_or_create_action_child: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const size_t child_slot = static_cast<size_t>(parent_node.action_child_begin) +
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
    if (!storage_.is_frozen()) {
      mgraph().action_children[child_slot] = kCappedNodeId;
    }
    stats_.record_transition_miss();
    return std::nullopt;
  }

  const size_t edge_index =
      static_cast<size_t>(parent.action_begin) + action_slot;
  const GameAction action = graph().betting_edges[edge_index].action;
  const BettingState child_betting = ApplyAction(parent.state, action);
  const ExactPublicState child_state{child_betting, parent_board};
  const BettingNodeId child_betting_node_id =
      get_or_create_action_betting_child(parent_node.betting_node_id,
                                         action_index);
  if (graph().betting_nodes[child_betting_node_id].state != child_betting) {
    throw std::logic_error("action betting child state mismatch");
  }
  auto child_id = get_or_create_node(
      child_betting_node_id, child_state, parent_node.public_observation);
  if (!child_id.has_value()) {
    mgraph().action_children[child_slot] = kCappedNodeId;
    stats_.record_transition_miss();
    return std::nullopt;
  }

  mgraph().action_children[child_slot] = *child_id;
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

std::optional<NodeId> GraphBuilder::find_or_cache_chance_child(
    NodeId parent_node_id,
    const ExactPublicState& child_state) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = graph_nodes[parent_node_id];
  const PublicObservationId child_public_observation =
      public_observation_after_chance(
          graph_node.public_observation, child_state.betting.street,
          child_state.board);
  const ChanceTransitionKey transition_key{
      parent_node_id, child_public_observation};
  auto existing = build_state_.chance_children.find(transition_key);
  if (existing != build_state_.chance_children.end()) {
    return existing->second;
  }

  if constexpr (kCoarsePublicBuckets) {
    const auto& betting_nodes = graph().betting_nodes;
    if (graph_node.betting_node_id >= betting_nodes.size()) {
      return std::nullopt;
    }

    const BettingNodeId child_node_id =
        betting_nodes[graph_node.betting_node_id].chance_child;
    if (child_node_id == kInvalidBettingNodeId) {
      return std::nullopt;
    }

    const PublicNodeKey child_key{
        child_node_id,
        child_public_observation,
    };
    auto child_id = build_state_.node_ids.find(child_key);
    if (child_id == build_state_.node_ids.end()) {
      return std::nullopt;
    }

    if (!storage_.is_frozen()) {
      build_state_.chance_children.emplace(
          transition_key, child_id->second);
    }
    return child_id->second;
  }

  return std::nullopt;
}

std::optional<NodeId>
GraphBuilder::get_or_create_chance_child(
    NodeId parent_node_id,
    const ExactPublicState& exact_child_state) {
  const auto& graph_nodes = nodes();
  if (parent_node_id >= graph_nodes.size()) {
    return std::nullopt;
  }
  const Node& parent_node = graph_nodes[parent_node_id];
  const PublicObservationId child_public_observation =
      public_observation_after_chance(
          parent_node.public_observation, exact_child_state.betting.street,
          exact_child_state.board);
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

  const BettingNodeId child_betting_node_id =
      get_or_create_chance_betting_child(parent_node.betting_node_id,
                                         exact_child_state.betting);
  auto child_id = get_or_create_node(
      child_betting_node_id, exact_child_state, child_public_observation);
  if (!child_id.has_value()) {
    stats_.record_transition_miss();
    return std::nullopt;
  }

  build_state_.chance_children.emplace(
      ChanceTransitionKey{parent_node_id, child_public_observation},
      *child_id);
  stats_.record_transition_miss();
  stats_.record_child_node_created();
  return child_id;
}

bool GraphBuilder::prebuild_reachable_nodes(
    NodeId root_id,
    const BoardRunout& root_board,
    int max_depth,
    std::vector<std::optional<BoardRunout>>& node_boards) {
  if (storage_.is_frozen()) {
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
    if (graph_node.betting_node_id >= graph().betting_nodes.size()) {
      return false;
    }
    const BettingNode node =
        graph().betting_nodes[graph_node.betting_node_id];
    if (node.kind == PublicGraph::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == PublicGraph::NodeKind::kChance) {
      const bool complete = for_each_required_chance_transition(
          graph_node, entry.board, [&](const ExactPublicState& child_state,
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
  PublicGraph& graph = mgraph();
  struct PendingChanceChild {
    uint32_t parent_id = 0;
    PublicObservationId child_public_observation = 0;
    NodeId child_id = kInvalidNodeId;

    bool operator<(const PendingChanceChild& other) const {
      return std::tie(parent_id, child_public_observation) <
             std::tie(other.parent_id, other.child_public_observation);
    }
  };

  std::vector<PendingChanceChild> pending;
  pending.reserve(build_state_.chance_children.size());
  for (const auto& [transition_key, child_id] :
       build_state_.chance_children) {
    const uint32_t parent_id = transition_key.parent_node_id;
    if (parent_id >= graph.nodes.size()) {
      continue;
    }
    pending.push_back({
        parent_id,
        transition_key.child_public_observation,
        child_id,
    });
  }

  std::sort(pending.begin(), pending.end());

  for (Node& graph_node : graph.nodes) {
    graph_node.chance_child_begin = 0;
    graph_node.chance_child_count = 0;
  }
  graph.chance_children.clear();
  graph.chance_children.reserve(pending.size());
  NodeId current_parent_id = kInvalidNodeId;
  for (const PendingChanceChild& child : pending) {
    if (child.child_id >= graph.nodes.size()) {
      continue;
    }
    if (child.parent_id != current_parent_id) {
      current_parent_id = child.parent_id;
      graph.nodes[current_parent_id].chance_child_begin =
          static_cast<uint32_t>(graph.chance_children.size());
    }
    graph.chance_children.push_back({
        child.child_public_observation,
        child.child_id,
    });
    ++graph.nodes[current_parent_id].chance_child_count;
  }
}

bool GraphBuilder::validate_prebuilt_nodes(
    NodeId root_id,
    const BoardRunout& root_board,
    int max_depth,
    TrainingRunStats& stats) const {
  const auto& graph_nodes = nodes();
  if (root_id >= graph_nodes.size()) {
    return false;
  }
  stats.action_transition_prebuild_complete = true;
  stats.chance_transition_prebuild_complete = true;

  if (build_state_.node_ids.size() != graph_nodes.size()) {
    stats.action_transition_prebuild_complete = false;
    stats.chance_transition_prebuild_complete = false;
    return false;
  }
  for (NodeId node_id = 0; node_id < graph_nodes.size(); ++node_id) {
    const Node& node = graph_nodes[node_id];
    const PublicNodeKey key{node.betting_node_id, node.public_observation};
    const auto stored = build_state_.node_ids.find(key);
    if (stored == build_state_.node_ids.end() || stored->second != node_id) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
  }

  NodeBfs bfs(root_id, root_board, graph_nodes.size());

  auto valid_child_id = [&](NodeId node_id) {
    return node_id != kInvalidNodeId &&
           node_id != kCappedNodeId &&
           node_id < graph_nodes.size();
  };
  auto enqueue_child = [&](NodeId node_id, BoardRunout board, int depth) {
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
    if (graph_node.betting_node_id >= graph().betting_nodes.size()) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    const BettingNode& node =
        graph().betting_nodes[graph_node.betting_node_id];
    ValidateBettingNode(node);
    if (node.kind != BettingNodeKind(node.state)) {
      stats.action_transition_prebuild_complete = false;
      stats.chance_transition_prebuild_complete = false;
      return false;
    }
    if (node.kind == PublicGraph::NodeKind::kDecision) {
      const ActionMenu menu =
          betting_abstraction_.actions_for_betting_node(node.state);
      if (menu.count != node.action_count) {
        stats.action_transition_prebuild_complete = false;
        return false;
      }
      for (uint8_t action = 0; action < menu.count; ++action) {
        const size_t edge = static_cast<size_t>(node.action_begin) + action;
        if (edge >= graph().betting_edges.size() ||
            graph().betting_edges[edge].action != menu.actions[action]) {
          stats.action_transition_prebuild_complete = false;
          return false;
        }
      }
    } else if (node.action_count != 0) {
      stats.action_transition_prebuild_complete = false;
      return false;
    }
    if (node.kind == PublicGraph::NodeKind::kTerminal) {
      continue;
    }

    if (node.kind == PublicGraph::NodeKind::kChance) {
      const bool chance_complete = for_each_required_chance_transition(
          graph_node, entry.board, [&](const ExactPublicState& child_state,
                                absl::Span<const CardId>) {
        ++stats.prebuild_chance_transitions;
        const PublicObservationId expected_observation =
            public_observation_after_chance(
                graph_node.public_observation, child_state.betting.street,
                child_state.board);
        NodeId child_id = kInvalidNodeId;
        try {
          child_id = graph().required_chance_child(
              entry.node_id, expected_observation);
        } catch (const std::logic_error&) {
        }
        const bool valid_child = valid_child_id(child_id) &&
            graph_nodes[child_id].public_observation == expected_observation &&
            graph_nodes[child_id].betting_node_id == node.chance_child &&
            node.chance_child < graph().betting_nodes.size() &&
            graph().betting_nodes[node.chance_child].state ==
                child_state.betting;
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
          graph().action_child(entry.node_id, action_index);
      const size_t edge_index = static_cast<size_t>(node.action_begin) +
                                static_cast<size_t>(action_index);
      const BettingEdge& edge = graph().betting_edges[edge_index];
      const bool valid_child = valid_child_id(child_id) &&
          graph_nodes[child_id].public_observation ==
              graph_node.public_observation &&
          graph_nodes[child_id].betting_node_id == edge.child &&
          edge.child < graph().betting_nodes.size() &&
          graph().betting_nodes[edge.child].state ==
              ApplyAction(node.state, edge.action);
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
