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

TEST_CASE("exact graph preserves reveal history") {
  const SameMaskHistory histories = BuildSameMaskHistories();
  const PublicStreetObservation observation_a =
      observe_public_street(StreetKind::kTurn, histories.runout_a);
  const PublicStreetObservation observation_b =
      observe_public_street(StreetKind::kTurn, histories.runout_b);
  CHECK(histories.runout_a.mask() == histories.runout_b.mask());
  CHECK(histories.runout_a != histories.runout_b);
  CHECK(observation_a.value == observation_b.value);
  CHECK(observation_a.exact_cards.count == 1);
  CHECK(observation_b.exact_cards.count == 1);
  CHECK(observation_a.exact_cards != observation_b.exact_cards);
  CHECK(observation_a != observation_b);
  CHECK(histories.observation_a != histories.observation_b);
  CHECK(histories.betting_a == histories.betting_b);
  CHECK(histories.node_a != histories.node_b);
}

TEST_CASE("public observation follows graph transitions") {
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  const NodeId root = graph.root(state);
  const NodeId action = graph.action_child(
      root, state, {ActionKind::kCall, 2});
  CHECK(graph.access().public_observation(action) ==
        graph.access().public_observation(root));

  const NodeId preflop = graph.action_child(
      action, state, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      C(14, S::kSpades), C(13, S::kSpades), C(12, S::kSpades)};
  const NodeId flop_node = graph.chance_child(preflop, state, flop);
  CHECK(graph.access().public_observation(flop_node) !=
        graph.access().public_observation(preflop));
}

TEST_CASE("canonical flop order deduplicates graph nodes") {
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId parent = test::ClosePreflop(graph, preflop);
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  const std::array<CardId, 3> flop_a = {
      C(14, S::kSpades), C(13, S::kSpades), C(12, S::kSpades)};
  const std::array<CardId, 3> flop_b = {
      C(12, S::kSpades), C(14, S::kSpades), C(13, S::kSpades)};
  CHECK(graph.chance_child(parent, state_a, flop_a) ==
        graph.chance_child(parent, state_b, flop_b));
}

TEST_CASE("mutable and frozen chance lookup agree") {
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  const NodeId parent = test::ClosePreflop(graph, state);
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(7, S::kDiamonds), C(12, S::kClubs)};
  const NodeId child = graph.chance_child(parent, state, flop);
  const PublicObservationId observation =
      graph.access().public_observation(child);
  CHECK(graph.access().chance_child(parent, observation) == child);
  graph.access().freeze();
  CHECK(graph.access().frozen_chance_child(parent, observation) == child);
}

TEST_CASE("validation rejects a chance edge with the wrong observation") {
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  NodeId node = test::ClosePreflop(graph, state);
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(7, S::kDiamonds), C(12, S::kClubs)};
  node = graph.chance_child(node, state, flop);
  node = test::CheckCheck(graph, node, state);
  const BoardRunout flop_board = state.board;
  REQUIRE(graph.prebuild(node, flop_board, 1));
  REQUIRE(graph.validate(node, flop_board, 1));

  const std::array<CardId, 1> turn = {C(9, S::kSpades)};
  const NodeId child = graph.chance_child(node, state, turn);
  graph.access().set_public_observation(
      child, graph.access().public_observation(child) + 1);
  CHECK_FALSE(graph.validate(node, flop_board, 1));
}

TEST_CASE("exact observations contain only the current street reveal") {
  BoardRunout board = BoardRunout::Preflop();
  const std::array<CardId, 3> flop = {
      C(12, S::kSpades), C(13, S::kSpades), C(14, S::kSpades)};
  board.deal_flop(flop);
  const PublicStreetObservation flop_observation =
      observe_public_street(StreetKind::kFlop, board);
  CHECK(flop_observation.exact_cards.count == 3);
  CHECK(flop_observation.exact_cards.cards == flop);

  board.deal_turn(C(11, S::kHearts));
  const PublicStreetObservation turn_observation =
      observe_public_street(StreetKind::kTurn, board);
  CHECK(turn_observation.exact_cards.count == 1);
  CHECK(turn_observation.exact_cards.cards[0] == C(11, S::kHearts));

  board.deal_river(C(10, S::kDiamonds));
  const PublicStreetObservation river_observation =
      observe_public_street(StreetKind::kRiver, board);
  CHECK(river_observation.exact_cards.count == 1);
  CHECK(river_observation.exact_cards.cards[0] == C(10, S::kDiamonds));

  const ComboId hand = CardsToComboId(
      C(14, S::kHearts), C(13, S::kHearts));
  CHECK(observe_private_street(hand, StreetKind::kRiver, board).value == hand);
}

TEST_CASE("exact private observation is always the hand") {
  const ComboId hand = CardsToComboId(
      C(14, S::kHearts), C(13, S::kHearts));
  test::IdentityGraph graph;
  ExactPublicState state = graph.initial_state();
  NodeId node = graph.root(state);
  const PrivateObservationId initial = initial_private_observation(hand);
  CHECK(initial == hand);

  node = graph.action_child(node, state, {ActionKind::kCall, 2});
  CHECK(private_observation_for_runout(
            hand, state.board,
            graph.access().public_observation(node)) == initial);
  node = graph.action_child(node, state, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(7, S::kDiamonds), C(12, S::kClubs)};
  node = graph.chance_child(node, state, flop);
  const PublicObservationId public_observation =
      graph.access().public_observation(node);
  CHECK(advance_private_observation(
            initial, hand, StreetKind::kFlop, state.board,
            public_observation) == hand);
  CHECK(private_observation_for_runout(
            hand, state.board, public_observation) == hand);
}

}  // namespace
}  // namespace poker
