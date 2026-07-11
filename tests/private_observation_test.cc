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
  const ComboId pair = CardsToComboId(
      C(14, S::kHearts), C(14, S::kDiamonds));
  const PrivateObservationId preflop_observation =
      initial_private_observation(hand);
  CHECK(preflop_observation != initial_private_observation(pair));

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
  const PrivateStreetObservation local_flop_a =
      observe_private_street(hand, StreetKind::kFlop, state_a.board);
  const PrivateStreetObservation local_flop_b =
      observe_private_street(hand, StreetKind::kFlop, state_b.board);
  const PrivateObservationId flop_observation_a =
      advance_private_observation(
          preflop_observation, hand, StreetKind::kFlop, state_a.board,
          graph.access().public_observation(node_a));
  const PrivateObservationId flop_observation_b =
      advance_private_observation(
          preflop_observation, hand, StreetKind::kFlop, state_b.board,
          graph.access().public_observation(node_b));
  CHECK(local_flop_a.value != local_flop_b.value);
  CHECK(local_flop_a.value <
        kCoarsePrivateStreetObservationCount);
  CHECK(preflop_observation != flop_observation_a);

  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);
  CHECK(private_observation_for_runout(
            hand, state_a.board,
            graph.access().public_observation(node_a)) ==
        flop_observation_a);
  const std::array<CardId, 1> turn_a = {C(6, S::kHearts)};
  const std::array<CardId, 1> turn_b = {C(4, S::kHearts)};
  node_a = graph.chance_child(node_a, state_a, turn_a);
  node_b = graph.chance_child(node_b, state_b, turn_b);
  const PrivateStreetObservation local_turn_a =
      observe_private_street(hand, StreetKind::kTurn, state_a.board);
  const PrivateStreetObservation local_turn_b =
      observe_private_street(hand, StreetKind::kTurn, state_b.board);
  const PrivateObservationId turn_observation_a =
      advance_private_observation(
          flop_observation_a, hand, StreetKind::kTurn, state_a.board,
          graph.access().public_observation(node_a));
  const PrivateObservationId turn_observation_b =
      advance_private_observation(
          flop_observation_b, hand, StreetKind::kTurn, state_b.board,
          graph.access().public_observation(node_b));
  CHECK(flop_observation_a != flop_observation_b);
  CHECK(local_turn_a == local_turn_b);
  CHECK(turn_observation_a != turn_observation_b);
  CHECK(turn_observation_a != flop_observation_a);
  CHECK(private_observation_for_runout(
            hand, state_a.board,
            graph.access().public_observation(node_a)) ==
        turn_observation_a);
  CHECK(node_a == node_b);

  node_a = test::CheckCheck(graph, node_a, state_a);
  const std::array<CardId, 1> river = {C(9, S::kDiamonds)};
  node_a = graph.chance_child(node_a, state_a, river);
  const PrivateObservationId river_observation =
      advance_private_observation(
          turn_observation_a, hand, StreetKind::kRiver, state_a.board,
          graph.access().public_observation(node_a));
  CHECK(river_observation != turn_observation_a);
  CHECK(private_observation_for_runout(
            hand, state_a.board,
            graph.access().public_observation(node_a)) ==
        river_observation);
}

}  // namespace
}  // namespace poker
