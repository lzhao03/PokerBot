#include "tests/identity_test_support.h"

#include <array>

#include "doctest/doctest.h"
#include "src/coarse_chance_transitions.h"

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "public_observation_test requires coarse public buckets"
#endif

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }

TEST_CASE("public observation traces retain colliding street prefixes") {
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId preflop_node = test::ClosePreflop(graph, preflop);

  const std::array<CardId, 3> flop_a = {
      C(2, S::kHearts), C(3, S::kHearts), C(6, S::kHearts)};
  const std::array<CardId, 3> flop_b = {
      C(2, S::kHearts), C(3, S::kHearts), C(7, S::kHearts)};
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  NodeId node_a = graph.chance_child(preflop_node, state_a, flop_a);
  NodeId node_b = graph.chance_child(preflop_node, state_b, flop_b);
  const PublicObservationId flop_observation_a =
      graph.access().public_observation(node_a);
  const PublicObservationId flop_observation_b =
      graph.access().public_observation(node_b);
  const PublicStreetObservation local_flop_observation =
      observe_public_street(StreetKind::kFlop, state_a.board);
  CHECK(current_public_street_observation(
            flop_observation_a, StreetKind::kFlop) ==
        local_flop_observation);
  CHECK(local_flop_observation.exact_cards.count == 0);
  CHECK(local_flop_observation.value <
        kCoarsePublicStreetObservationCount);

  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);
  CHECK(graph.access().public_observation(node_a) == flop_observation_a);
  CHECK(graph.access().public_observation(node_b) == flop_observation_b);
  const std::array<CardId, 1> turn_a = {C(10, S::kHearts)};
  const std::array<CardId, 1> turn_b = {C(4, S::kHearts)};
  node_a = graph.chance_child(node_a, state_a, turn_a);
  node_b = graph.chance_child(node_b, state_b, turn_b);
  const PublicObservationId turn_observation_a =
      graph.access().public_observation(node_a);
  const PublicObservationId turn_observation_b =
      graph.access().public_observation(node_b);
  const PublicStreetObservation local_turn_observation_a =
      current_public_street_observation(
          turn_observation_a, StreetKind::kTurn);
  const PublicStreetObservation local_turn_observation_b =
      current_public_street_observation(
          turn_observation_b, StreetKind::kTurn);
  CHECK(flop_observation_a != flop_observation_b);
  CHECK(local_turn_observation_a == local_turn_observation_b);
  CHECK(turn_observation_a != turn_observation_b);
  CHECK(turn_observation_a == advance_public_observation(
                                  flop_observation_a, StreetKind::kTurn,
                                  local_turn_observation_a));
  CHECK(turn_observation_b == advance_public_observation(
                                  flop_observation_b, StreetKind::kTurn,
                                  local_turn_observation_b));
  CHECK(node_a != node_b);
}

TEST_CASE("coarse public history packs and recovers every street") {
  const PublicStreetObservation flop{7, {}};
  const PublicStreetObservation turn{12, {}};
  const PublicStreetObservation river{3, {}};
  const PublicObservationId flop_history = advance_public_observation(
      initial_public_observation(), StreetKind::kFlop, flop);
  const PublicObservationId turn_history = advance_public_observation(
      flop_history, StreetKind::kTurn, turn);
  const PublicObservationId river_history = advance_public_observation(
      turn_history, StreetKind::kRiver, river);

  CHECK(current_public_street_observation(
            river_history, StreetKind::kFlop) == flop);
  CHECK(current_public_street_observation(
            river_history, StreetKind::kTurn) == turn);
  CHECK(current_public_street_observation(
            river_history, StreetKind::kRiver) == river);
  CHECK(advance_public_observation(
            flop_history, StreetKind::kTurn, turn) == turn_history);
}

TEST_CASE("coarse transition tables use the current local observation") {
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  NodeId node = test::ClosePreflop(graph, state);
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(3, S::kHearts), C(6, S::kHearts)};
  node = graph.chance_child(node, state, flop);
  node = test::CheckCheck(graph, node, state);
  const PublicObservationId parent_observation =
      graph.access().public_observation(node);
  REQUIRE(graph.prebuild(node, state.board, 1));
  REQUIRE(graph.validate(node, state.board, 1));

  const BoardBucketId local_parent = current_public_street_observation(
      parent_observation, StreetKind::kFlop).value;
  const BoardBucketId generated_parent =
      1 + static_cast<BoardBucketId>(StreetKind::kFlop) *
              kCoarsePublicStreetObservationCount +
      local_parent;
  int transition_count = 0;
  for (const CoarseChanceTransition& transition :
       kFlopTextureTransitions) {
    if (transition.parent_bucket != generated_parent) {
      continue;
    }
    ++transition_count;
    const BoardBucketId local_child =
        transition.child_bucket - 1 -
        static_cast<BoardBucketId>(StreetKind::kTurn) *
            kCoarsePublicStreetObservationCount;
    const PublicObservationId child_observation =
        advance_public_observation(
            parent_observation, StreetKind::kTurn,
            PublicStreetObservation{local_child, {}});
    CHECK(graph.access().chance_child(node, child_observation) !=
          kInvalidNodeId);
  }
  CHECK(transition_count > 0);
}

}  // namespace
}  // namespace poker
