#include "src/poker.h"
#include "src/bet_abstraction.h"

#include "doctest/doctest.h"

#include <array>
#include <stdexcept>
#include <string>

namespace poker {
namespace {

using S = Suit;

Card C(int rank, S suit) { return Card(static_cast<Rank>(rank - 2), suit); }

BettingData& B(BettingState& state) {
  return std::visit([](auto& phase) -> BettingData& { return phase.data; },
                    state);
}

BettingData& B(ExactPublicState& state) { return B(state.betting); }

BettingState Apply(const BettingState& state, GameAction action) {
  const auto* decision = std::get_if<DecisionState>(&state);
  if (decision == nullptr) {
    throw std::invalid_argument("expected decision state");
  }
  const auto child = ApplyAction(*decision, action);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

ExactPublicState DealChance(const ExactPublicState& state,
                            absl::Span<const Card> cards,
                            const BettingRules& rules) {
  const auto child = TryApplyChance(state, cards, rules);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

ExactPublicState Root() {
  ExactPublicState state;
  B(state).stack = {19, 18};
  B(state).total_committed = {1, 2};
  B(state).street_committed = {1, 2};
  B(state).last_full_raise = 2;
  return state;
}

constexpr BettingRules kRules{2};

TEST_CASE("boundary actions, chance transitions, and sizing are enforced") {
  ExactPublicState state = Root();
  const auto* root_decision = std::get_if<DecisionState>(&state.betting);
  REQUIRE(root_decision != nullptr);
  const LegalActionSpace root_legal = LegalActions(*root_decision);
  CHECK(root_legal.current_to == 1);
  CHECK(root_legal.highest_to == 2);
  CHECK(root_legal.call_to == 2);
  CHECK(root_legal.min_full_raise_to == 4);
  CHECK(root_legal.all_in_to == 20);
  CHECK_THROWS(Apply(state.betting, {ActionKind::kCheck}));
  CHECK_THROWS(Apply(state.betting, {ActionKind::kRaise, 1}));
  state.betting = Apply(state.betting, {ActionKind::kCall, 2});
  state.betting = Apply(state.betting, {ActionKind::kCheck});
  CHECK_THROWS_AS(DealChance(state, {C(14, S::kSpades)}, kRules),
                  std::invalid_argument);

  ExactPublicState short_call = Root();
  B(short_call).stack = {3, 12};
  B(short_call).total_committed = {1, 8};
  B(short_call).street_committed = {1, 8};
  const auto* short_decision =
      std::get_if<DecisionState>(&short_call.betting);
  REQUIRE(short_decision != nullptr);
  const LegalActionSpace short_legal = LegalActions(*short_decision);
  CHECK(short_legal.call_to == 4);
  CHECK_FALSE(short_legal.can_aggress());
  short_call.betting = Apply(short_call.betting,
                                   {ActionKind::kCall, 4});
  CHECK(std::holds_alternative<ChanceState>(short_call.betting));
  CHECK(B(short_call).stack[0] == 0);
  CHECK(B(short_call).total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(short_call).street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(short_call).stack[1] == 16);

  BetAbstractionConfig config;
  config.bet_sizes[static_cast<size_t>(StreetKind::kPreflop)] = {0.5};
  config.bet_sizes[static_cast<size_t>(StreetKind::kFlop)] = {1.0};
  BettingData flop_data;
  flop_data.stack = {98, 98};
  flop_data.total_committed = {2, 2};
  flop_data.last_full_raise = 2;
  flop_data.street = StreetKind::kFlop;
  const DecisionState flop{flop_data, Player::kB};
  const LegalActionSpace legal = LegalActions(flop);
  const AbstractActions actions = SelectAbstractActions(config, flop, legal);
  bool bet_four = false;
  bool bet_two = false;
  bool all_in = false;
  for (const GameAction& action : actions) {
    bet_four |= action == GameAction{ActionKind::kBet, 4};
    bet_two |= action == GameAction{ActionKind::kBet, 2};
    all_in |= action == GameAction{ActionKind::kAllIn, 98};
  }
  CHECK(bet_four);
  CHECK_FALSE(bet_two);
  CHECK(all_in);
}

}  // namespace
}  // namespace poker
