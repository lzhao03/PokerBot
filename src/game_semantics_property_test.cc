#include "src/betting_abstraction.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "src/card_utils.h"
#include "src/game_rules.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <ostream>
#include <vector>

namespace poker {
namespace {

struct ReachableScenario {
  int starting_stack = 20;
  std::vector<uint16_t> choices;
};

std::ostream& operator<<(std::ostream& out,
                         const ReachableScenario& scenario) {
  out << "stack=" << scenario.starting_stack << " choices=[";
  for (size_t i = 0; i < scenario.choices.size(); ++i) {
    out << (i == 0 ? "" : ",") << scenario.choices[i];
  }
  return out << "]";
}

auto ReachableScenarios() {
  return rc::gen::build<ReachableScenario>(
      rc::gen::set(&ReachableScenario::starting_stack,
                   rc::gen::inRange(4, 33)),
      rc::gen::set(&ReachableScenario::choices,
                   rc::gen::resize(
                       64, rc::gen::arbitrary<std::vector<uint16_t>>())))
      .as("scenario");
}

SolverConfig TestConfig() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {0.5, 1.0};
  return config;
}

ExactGameState InitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;

  ExactGameState state;
  state.betting.stack[0] =
      std::max(0, config.starting_stack_size - small_blind);
  state.betting.stack[1] =
      std::max(0, config.starting_stack_size - big_blind);
  state.betting.committed = {small_blind, big_blind};
  return state;
}

bool Terminal(const ExactGameState& state) {
  return IsTerminal(state.betting, state.board);
}

int TotalChips(const ExactGameState& state) {
  return Pot(state.betting) + state.betting.stack[0] +
         state.betting.stack[1];
}

void CheckState(const ExactGameState& state, int total_chips) {
  const BettingState& betting = state.betting;
  RC_ASSERT(IsValidBettingState(betting));
  RC_ASSERT(TotalChips(state) == total_chips);
  RC_ASSERT(Pot(betting) == betting.committed[0] + betting.committed[1]);
  RC_ASSERT(std::popcount(state.board.mask) == state.board.count);
  RC_ASSERT(BoardCardsForStreet(betting.street) == state.board.count);
  RC_ASSERT(IsBettingRoundOver(betting) == !IsPlayer(betting.player_to_act));

  CardMask seen = 0;
  for (CardId card : state.board.span()) {
    RC_ASSERT((seen & CardBit(card)) == 0);
    RC_ASSERT(state.board.contains(card));
    seen |= CardBit(card);
  }
}

void CheckActionTransition(const ExactGameState& parent,
                           const ExactGameState& child,
                           int player,
                           int total_chips) {
  CheckState(child, total_chips);
  const Chips committed =
      parent.betting.stack[player] - child.betting.stack[player];
  RC_ASSERT(committed >= 0);
  RC_ASSERT(Pot(child.betting) == Pot(parent.betting) + committed);
  RC_ASSERT(child.betting.committed[player] ==
            parent.betting.committed[player] + committed);
  RC_ASSERT(child.betting.committed[Opponent(player)] ==
            parent.betting.committed[Opponent(player)]);
  RC_ASSERT(child.board.mask == parent.board.mask);
}

CardId SelectAvailableCard(CardMask blocked, uint16_t choice) {
  std::array<CardId, kDeckCardCount> available = {};
  size_t count = 0;
  for (int id = 0; id < kDeckCardCount; ++id) {
    const CardId card = static_cast<CardId>(id);
    if ((blocked & CardBit(card)) == 0) {
      available[count++] = card;
    }
  }
  RC_ASSERT(count > 0);
  return available[choice % count];
}

void CheckChanceTransition(const ExactGameState& parent,
                           const ExactGameState& child,
                           absl::Span<const CardId> cards,
                           int total_chips) {
  CheckState(child, total_chips);
  RC_ASSERT(child.betting.stack == parent.betting.stack);
  RC_ASSERT(child.betting.committed == parent.betting.committed);
  RC_ASSERT(child.betting.street == StreetAfterChance(parent.betting.street));
  RC_ASSERT(child.betting.pending_action_mask == kAllPlayersMask);
  RC_ASSERT((child.board.mask & parent.board.mask) == parent.board.mask);

  CardMask dealt = 0;
  for (CardId card : cards) {
    RC_ASSERT((parent.board.mask & CardBit(card)) == 0);
    RC_ASSERT((dealt & CardBit(card)) == 0);
    dealt |= CardBit(card);
  }
}

void ReplayAndValidate(const ReachableScenario& scenario) {
  SolverConfig config = TestConfig();
  config.starting_stack_size = scenario.starting_stack;
  const BettingAbstraction betting(config);
  ExactGameState state = InitialState(config);
  const int total_chips = TotalChips(state);
  size_t cursor = 0;

  while (true) {
    CheckState(state, total_chips);
    if (Terminal(state)) {
      return;
    }

    const int player = state.betting.player_to_act;
    if (IsPlayer(player)) {
      const ActionMenu menu =
          betting.actions_for_betting_node(state.betting);
      RC_ASSERT(menu.count > 0);
      for (uint8_t i = 0; i < menu.count; ++i) {
        ExactGameState child = state;
        child.betting = ApplyAction(state.betting, menu.actions[i]);
        CheckActionTransition(state, child, player, total_chips);
      }
      if (cursor == scenario.choices.size()) {
        return;
      }
      const size_t action = scenario.choices[cursor++] % menu.count;
      state.betting = ApplyAction(state.betting, menu.actions[action]);
      continue;
    }

    const int card_count = CardsForNextStreet(state.betting.street);
    if (cursor + static_cast<size_t>(card_count) > scenario.choices.size()) {
      return;
    }

    std::vector<CardId> cards;
    cards.reserve(static_cast<size_t>(card_count));
    CardMask blocked = state.board.mask;
    for (int i = 0; i < card_count; ++i) {
      const CardId card =
          SelectAvailableCard(blocked, scenario.choices[cursor++]);
      cards.push_back(card);
      blocked |= CardBit(card);
    }

    const ExactGameState parent = state;
    state = ApplyChance(state, cards);
    CheckChanceTransition(parent, state, cards, total_chips);
  }
}

TEST_CASE("reachable states preserve poker invariants") {
  const bool passed = rc::check("reachable game trace", [] {
    const ReachableScenario scenario = *ReachableScenarios();
    ReplayAndValidate(scenario);
  });
  CHECK(passed);
}

}  // namespace
}  // namespace poker
