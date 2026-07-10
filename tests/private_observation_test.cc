#include "tests/identity_test_support.h"

#include <array>

#include "doctest/doctest.h"
#include "src/combo.h"

namespace poker {
namespace {

#if !POKER_COARSE_PRIVATE_BUCKETS || !POKER_COARSE_PUBLIC_BUCKETS
#error "private_observation_test requires coarse card buckets"
#endif

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }

TEST_CASE("private observation traces retain colliding street prefixes") {
  const ComboId hand = CardsToComboId(
      C(14, S::kSpades), C(13, S::kSpades));
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId preflop_node = test::ClosePreflop(graph, preflop);

  const std::array<CardId, 3> flop_a = {
      C(2, S::kHearts), C(3, S::kHearts), C(5, S::kHearts)};
  const std::array<CardId, 3> flop_b = {
      C(2, S::kHearts), C(3, S::kHearts), C(6, S::kHearts)};
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  NodeId node_a = graph.chance_child(preflop_node, state_a, flop_a);
  NodeId node_b = graph.chance_child(preflop_node, state_b, flop_b);
  const PrivateBucketId flop_observation_a =
      test::PrivateObservation(hand, state_a);
  const PrivateBucketId flop_observation_b =
      test::PrivateObservation(hand, state_b);

  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);
  const std::array<CardId, 1> turn_a = {C(6, S::kHearts)};
  const std::array<CardId, 1> turn_b = {C(4, S::kHearts)};
  node_a = graph.chance_child(node_a, state_a, turn_a);
  node_b = graph.chance_child(node_b, state_b, turn_b);
  const PrivateBucketId turn_observation_a =
      test::PrivateObservation(hand, state_a);
  const PrivateBucketId turn_observation_b =
      test::PrivateObservation(hand, state_b);

  const test::PrivateObservationTrace trace_a = {
      flop_observation_a, turn_observation_a};
  const test::PrivateObservationTrace trace_b = {
      flop_observation_b, turn_observation_b};
  CHECK(flop_observation_a != flop_observation_b);
  CHECK(turn_observation_a == turn_observation_b);
  CHECK(trace_a != trace_b);
  CHECK(node_a == node_b);
}

}  // namespace
}  // namespace poker
