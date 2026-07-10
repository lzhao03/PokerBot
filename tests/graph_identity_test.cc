#include "tests/identity_test_support.h"

#include <array>

#include "doctest/doctest.h"

namespace poker {
namespace {

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }

struct SameMaskHistory {
  BoardRunout runout_a = BoardRunout::Preflop();
  BoardRunout runout_b = BoardRunout::Preflop();
  NodeId node_a = kInvalidNodeId;
  NodeId node_b = kInvalidNodeId;
  PublicObservationId observation_a = 0;
  PublicObservationId observation_b = 0;
  BettingNodeId betting_a = kInvalidBettingNodeId;
  BettingNodeId betting_b = kInvalidBettingNodeId;
};

SameMaskHistory BuildSameMaskHistories() {
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId preflop_node = test::ClosePreflop(graph, preflop);

  const std::array<CardId, 3> flop_a = {
      C(14, S::kSpades), C(13, S::kSpades), C(12, S::kSpades)};
  const std::array<CardId, 3> flop_b = {
      C(14, S::kSpades), C(13, S::kSpades), C(11, S::kHearts)};
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  NodeId node_a = graph.chance_child(preflop_node, state_a, flop_a);
  NodeId node_b = graph.chance_child(preflop_node, state_b, flop_b);
  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);

  const std::array<CardId, 1> turn_a = {C(11, S::kHearts)};
  const std::array<CardId, 1> turn_b = {C(12, S::kSpades)};
  node_a = graph.chance_child(node_a, state_a, turn_a);
  node_b = graph.chance_child(node_b, state_b, turn_b);

  const auto& access = graph.access();
  return {
      state_a.board,
      state_b.board,
      node_a,
      node_b,
      access.public_observation(node_a),
      access.public_observation(node_b),
      access.betting_history(node_a),
      access.betting_history(node_b),
  };
}

TEST_CASE("current graph merges equal masks with different reveal history") {
  const SameMaskHistory histories = BuildSameMaskHistories();
  CHECK(histories.runout_a.mask() == histories.runout_b.mask());
  CHECK(histories.runout_a != histories.runout_b);
  CHECK(histories.observation_a == histories.observation_b);
  CHECK(histories.betting_a == histories.betting_b);
  CHECK(histories.node_a == histories.node_b);
}

// TODO(identity-commit-3): Enable when public observation IDs retain prefixes.
TEST_CASE("different reveal histories have distinct graph identity" *
          doctest::skip()) {
  const SameMaskHistory histories = BuildSameMaskHistories();
  CHECK(histories.node_a != histories.node_b);
}

}  // namespace
}  // namespace poker
