#include "src/poker.h"
#include "src/bet_abstraction.h"

#include "doctest/doctest.h"

#include <array>
#include <stdexcept>
#include <string>

namespace poker {
namespace {

using S = Suit;

TEST_CASE("small betting uses half-pot and pot-sized actions") {
  const BetAbstractionConfig config = SmallBettingConfig();
  for (const auto& fractions : config.pot_fractions) {
    REQUIRE(fractions.size() == 2);
    CHECK(fractions[0] == 0.5);
    CHECK(fractions[1] == 1.0);
  }
}

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
  CHECK(IsLegalAction(*root_decision, {ActionKind::Call, 2}));
  CHECK(IsLegalAction(*root_decision, {ActionKind::Raise, 4}));
  CHECK(IsLegalAction(*root_decision, {ActionKind::AllIn, 20}));
  CHECK_FALSE(IsLegalAction(*root_decision, {ActionKind::Raise, 3}));
  CHECK_THROWS(Apply(state.betting, {ActionKind::Check}));
  CHECK_THROWS(Apply(state.betting, {ActionKind::Raise, 1}));
  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  state.betting = Apply(state.betting, {ActionKind::Check});
  CHECK_THROWS_AS(DealChance(state, {C(14, S::Spades)}, kRules),
                  std::invalid_argument);

  ExactPublicState short_call = Root();
  B(short_call).stack = {3, 12};
  B(short_call).total_committed = {1, 8};
  B(short_call).street_committed = {1, 8};
  const auto* short_decision =
      std::get_if<DecisionState>(&short_call.betting);
  REQUIRE(short_decision != nullptr);
  CHECK(IsLegalAction(*short_decision, {ActionKind::Call, 4}));
  CHECK_FALSE(IsLegalAction(*short_decision, {ActionKind::AllIn, 4}));
  short_call.betting = Apply(short_call.betting,
                                   {ActionKind::Call, 4});
  CHECK(std::holds_alternative<ChanceState>(short_call.betting));
  CHECK(B(short_call).stack[0] == 0);
  CHECK(B(short_call).total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(short_call).street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(short_call).stack[1] == 16);

  BetAbstractionConfig config;
  config.pot_fractions[static_cast<size_t>(StreetKind::Preflop)] = {0.5};
  config.pot_fractions[static_cast<size_t>(StreetKind::Flop)] = {1.0};
  BettingData flop_data;
  flop_data.stack = {98, 98};
  flop_data.total_committed = {2, 2};
  flop_data.last_full_raise = 2;
  flop_data.street = StreetKind::Flop;
  const DecisionState flop{flop_data, Player::B};
  const AbstractActions actions = SelectAbstractActions(config, flop);
  bool bet_four = false;
  bool bet_two = false;
  bool all_in = false;
  for (const GameAction& action : actions) {
    bet_four |= action == GameAction{ActionKind::Bet, 4};
    bet_two |= action == GameAction{ActionKind::Bet, 2};
    all_in |= action == GameAction{ActionKind::AllIn, 98};
  }
  CHECK(bet_four);
  CHECK_FALSE(bet_two);
  CHECK(all_in);
}

}  // namespace
}  // namespace poker
