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
  std::vector<uint8_t> bet_size_points;
  std::vector<uint16_t> choices;
};

std::ostream& operator<<(std::ostream& out,
                         const ReachableScenario& scenario) {
  out << "stack=" << scenario.starting_stack << " choices=[";
  for (size_t i = 0; i < scenario.choices.size(); ++i) {
    out << (i == 0 ? "" : ",") << scenario.choices[i];
  }
  out << "] sizes=[";
  for (size_t i = 0; i < scenario.bet_size_points.size(); ++i) {
    out << (i == 0 ? "" : ",")
        << static_cast<int>(scenario.bet_size_points[i]);
  }
  return out << "]";
}

auto ReachableScenarios() {
  return rc::gen::build<ReachableScenario>(
      rc::gen::set(&ReachableScenario::starting_stack,
                   rc::gen::inRange(4, 33)),
      rc::gen::set(&ReachableScenario::bet_size_points,
                   rc::gen::resize(
                       5, rc::gen::arbitrary<std::vector<uint8_t>>())),
      rc::gen::set(&ReachableScenario::choices,
                   rc::gen::resize(
                       64, rc::gen::arbitrary<std::vector<uint16_t>>())))
      .as("scenario");
}

ExactGameState InitialState(int starting_stack) {
  ExactGameState state;
  state.betting.stack = {starting_stack - 1, starting_stack - 2};
  state.betting.committed = {1, 2};
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

ActionMenu CheckLegalMenu(const ExactGameState& state,
                          absl::Span<const double> bet_sizes,
                          int total_chips) {
  const BettingState& betting = state.betting;
  const ActionMenu menu = LegalActions(betting, bet_sizes);
  RC_ASSERT(menu.count > 0);

  std::vector<double> sorted_sizes(bet_sizes.begin(), bet_sizes.end());
  std::sort(sorted_sizes.begin(), sorted_sizes.end());
  const ActionMenu sorted_menu = LegalActions(betting, sorted_sizes);
  RC_ASSERT(menu.count == sorted_menu.count);

  for (uint8_t i = 0; i < menu.count; ++i) {
    const GameAction action = menu.actions[i];
    RC_ASSERT(action == sorted_menu.actions[i]);
    for (uint8_t j = 0; j < i; ++j) {
      RC_ASSERT(action != menu.actions[j]);
    }

    ExactGameState child = state;
    child.betting = ApplyAction(betting, action);
    CheckActionTransition(state, child, betting.player_to_act, total_chips);
  }

  uint8_t index = 0;
  const Chips to_call = ToCall(betting, betting.player_to_act);
  if (to_call > 0) {
    RC_ASSERT(menu.actions[index++] == GameAction{ActionKind::kFold});
    RC_ASSERT((menu.actions[index++] ==
               GameAction{ActionKind::kCall,
                          std::min(to_call,
                                   betting.stack[betting.player_to_act])}));
  } else {
    RC_ASSERT(menu.actions[index++] == GameAction{ActionKind::kCheck});
  }

  const ActionKind sized_kind =
      to_call > 0 ? ActionKind::kRaise : ActionKind::kBet;
  Chips previous_amount = 0;
  while (index < menu.count && menu.actions[index].kind == sized_kind) {
    RC_ASSERT(menu.actions[index].amount > previous_amount);
    previous_amount = menu.actions[index].amount;
    ++index;
  }
  if (index < menu.count) {
    RC_ASSERT((menu.actions[index] ==
               GameAction{ActionKind::kAllIn,
                          betting.stack[betting.player_to_act]}));
    ++index;
  }
  RC_ASSERT(index == menu.count);
  return menu;
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
  std::vector<double> bet_sizes;
  bet_sizes.reserve(scenario.bet_size_points.size());
  for (uint8_t points : scenario.bet_size_points) {
    bet_sizes.push_back((static_cast<double>(points) + 1.0) / 100.0);
  }

  ExactGameState state = InitialState(scenario.starting_stack);
  const int total_chips = TotalChips(state);
  size_t cursor = 0;

  while (true) {
    CheckState(state, total_chips);
    if (Terminal(state)) {
      return;
    }

    const int player = state.betting.player_to_act;
    if (IsPlayer(player)) {
      const ActionMenu menu = CheckLegalMenu(state, bet_sizes, total_chips);
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

TEST_CASE("reachable legal menus are canonical and applicable") {
  const bool passed = rc::check("reachable legal menu", [] {
    const ReachableScenario scenario = *ReachableScenarios();
    ReplayAndValidate(scenario);
  });
  CHECK(passed);
}

}  // namespace
}  // namespace poker
