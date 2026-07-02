#include "src/utility_calculator.h"

#include <cmath>
#include <stdexcept>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

poker::Hand MakeHand(int first_rank, poker::Suit first_suit, int second_rank,
                     poker::Suit second_suit) {
  poker::Hand hand;
  poker::Card* first = hand.add_cards();
  first->set_rank(first_rank);
  first->set_suit(first_suit);
  poker::Card* second = hand.add_cards();
  second->set_rank(second_rank);
  second->set_suit(second_suit);
  return hand;
}

void AddCard(poker::BoardState* state, int rank, poker::Suit suit) {
  poker::Card* card = state->add_cards();
  card->set_rank(rank);
  card->set_suit(suit);
}

poker::BoardState RiverState() {
  poker::BoardState state;
  state.set_pot(20);
  state.set_street(poker::Street::RIVER);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(10);
  state.add_player_contribution(10);
  AddCard(&state, 2, poker::Suit::CLUBS);
  AddCard(&state, 7, poker::Suit::DIAMONDS);
  AddCard(&state, 9, poker::Suit::CLUBS);
  AddCard(&state, 11, poker::Suit::SPADES);
  AddCard(&state, 3, poker::Suit::DIAMONDS);
  return state;
}

void CheckRangeEquitySkipsPlayerOverlap() {
  poker::UtilityCalculator calculator;
  poker::BoardState state = RiverState();
  poker::Hand aces =
      MakeHand(14, poker::Suit::HEARTS, 14, poker::Suit::SPADES);
  poker::Hand kings =
      MakeHand(13, poker::Suit::HEARTS, 13, poker::Suit::SPADES);

  poker::HandRange opponent_range;
  opponent_range.add_hand(aces, 1.0);
  opponent_range.add_hand(kings, 1.0);

  double equity = calculator.calculate_equity(aces, opponent_range, state);
  Expect(std::abs(equity - 1.0) < 0.000001,
         "range equity should skip hands overlapping the player hand");
}

void CheckRangeEquitySkipsBoardOverlap() {
  poker::UtilityCalculator calculator;
  poker::BoardState state = RiverState();
  state.mutable_cards(0)->set_rank(13);
  state.mutable_cards(0)->set_suit(poker::Suit::SPADES);

  poker::Hand aces =
      MakeHand(14, poker::Suit::HEARTS, 14, poker::Suit::DIAMONDS);
  poker::Hand blocked_kings =
      MakeHand(13, poker::Suit::SPADES, 13, poker::Suit::HEARTS);
  poker::Hand queens =
      MakeHand(12, poker::Suit::SPADES, 12, poker::Suit::HEARTS);

  poker::HandRange opponent_range;
  opponent_range.add_hand(blocked_kings, 1.0);
  opponent_range.add_hand(queens, 1.0);

  double equity = calculator.calculate_equity(aces, opponent_range, state);
  Expect(std::abs(equity - 1.0) < 0.000001,
         "range equity should skip hands overlapping the board");
}

void CheckRangeEvSkipsPlayerOverlap() {
  poker::UtilityCalculator calculator;
  poker::BoardState state = RiverState();
  poker::Hand aces =
      MakeHand(14, poker::Suit::HEARTS, 14, poker::Suit::SPADES);
  poker::Hand kings =
      MakeHand(13, poker::Suit::HEARTS, 13, poker::Suit::SPADES);

  poker::HandRange opponent_range;
  opponent_range.add_hand(aces, 1.0);
  opponent_range.add_hand(kings, 1.0);

  double ev =
      calculator.calculate_expected_value(state, aces, opponent_range, 0);
  Expect(std::abs(ev - 20.0) < 0.000001,
         "range EV should skip hands overlapping the player hand");
}

}  // namespace

int main() {
  CheckRangeEquitySkipsPlayerOverlap();
  CheckRangeEquitySkipsBoardOverlap();
  CheckRangeEvSkipsPlayerOverlap();
  return 0;
}
