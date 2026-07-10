#include "tests/identity_test_support.h"

#include <array>

#include "doctest/doctest.h"

namespace poker {
namespace {

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }

TEST_CASE("betting history remains distinct after equivalent chip states") {
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  NodeId node = graph.root(state);
  CHECK(graph.access().root_node() == node);
  node = graph.action_child(node, state, {ActionKind::kCall, 2});
  node = graph.action_child(node, state, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(7, S::kDiamonds), C(12, S::kClubs)};
  node = graph.chance_child(node, state, flop);

  ExactPublicState state_a = state;
  ExactPublicState state_b = state;
  NodeId node_a = graph.action_child(
      node, state_a, {ActionKind::kBet, 2});
  node_a = graph.action_child(
      node_a, state_a, {ActionKind::kRaise, 5});
  node_a = graph.action_child(
      node_a, state_a, {ActionKind::kCall, 5});

  NodeId node_b = graph.action_child(
      node, state_b, {ActionKind::kBet, 5});
  node_b = graph.action_child(
      node_b, state_b, {ActionKind::kCall, 5});
  CHECK(state_a.betting != state_b.betting);

  const std::array<CardId, 1> turn = {C(9, S::kSpades)};
  node_a = graph.chance_child(node_a, state_a, turn);
  node_b = graph.chance_child(node_b, state_b, turn);
  CHECK(state_a.betting == state_b.betting);
  CHECK(state_a.board == state_b.board);
  CHECK(graph.access().betting_history(node_a) !=
        graph.access().betting_history(node_b));
  CHECK(node_a != node_b);
}

}  // namespace
}  // namespace poker
