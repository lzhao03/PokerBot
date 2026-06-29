#include "src/info_set.h"

#include <stdexcept>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Action MakeAction(int player, ActionType type, float amount = 0.0f) {
  Action action;
  action.set_player(player);
  action.set_action(type);
  action.set_amount(amount);
  return action;
}

void AddCard(BoardState* state, int rank, Suit suit) {
  Card* card = state->add_cards();
  card->set_rank(rank);
  card->set_suit(suit);
}

Hand MakeHand() {
  Hand hand;
  Card* first = hand.add_cards();
  first->set_rank(14);
  first->set_suit(Suit::SPADES);
  Card* second = hand.add_cards();
  second->set_rank(13);
  second->set_suit(Suit::SPADES);
  return hand;
}

BoardState MakeState() {
  BoardState state;
  state.set_street(Street::FLOP);
  AddCard(&state, 2, Suit::HEARTS);
  AddCard(&state, 7, Suit::DIAMONDS);
  AddCard(&state, 11, Suit::CLUBS);
  return state;
}

}  // namespace
}  // namespace poker

int main() {
  poker::InfoSetAbstraction abstraction;
  poker::Hand hand = poker::MakeHand();

  poker::BoardState checked = poker::MakeState();
  *checked.mutable_history()->add_actions() =
      poker::MakeAction(0, poker::ActionType::CHECK);

  poker::BoardState bet = poker::MakeState();
  *bet.mutable_history()->add_actions() =
      poker::MakeAction(1, poker::ActionType::BET, 5.0f);

  poker::BoardState player_zero_bet = poker::MakeState();
  *player_zero_bet.mutable_history()->add_actions() =
      poker::MakeAction(0, poker::ActionType::BET, 5.0f);

  std::string checked_key = abstraction.state_to_info_set(checked, 0, hand);
  std::string bet_key = abstraction.state_to_info_set(bet, 0, hand);
  std::string player_zero_bet_key =
      abstraction.state_to_info_set(player_zero_bet, 0, hand);

  poker::Expect(checked_key != bet_key, "different histories need different info sets");
  poker::Expect(player_zero_bet_key != bet_key,
                "same action by different players needs different info sets");
  poker::Expect(!abstraction.same_info_set(checked, bet, 0),
                "same_info_set should include action history");

  poker::BoardState small_pot = poker::MakeState();
  small_pot.set_pot(10);
  small_pot.set_stack_a(90);
  small_pot.set_stack_b(90);
  small_pot.add_player_contribution(5);
  small_pot.add_player_contribution(5);
  poker::BoardState large_pot = small_pot;
  large_pot.set_pot(20);
  std::string small_pot_key = abstraction.state_to_info_set(small_pot, 0, hand);
  std::string large_pot_key = abstraction.state_to_info_set(large_pot, 0, hand);
  poker::Expect(small_pot_key != large_pot_key,
                "different public pot sizes need different info sets");
  poker::Expect(!abstraction.same_info_set(small_pot, large_pot, 0),
                "same_info_set should include public pot size");

  poker::InfoSetAbstraction::InfoSetComponents components =
      abstraction.parse_info_set(bet_key);
  poker::Expect(components.betting_history.size() == 1,
                "info set parser should round-trip history");
  poker::Expect(components.betting_history[0].action() == poker::ActionType::BET,
                "parsed action type should match");
  poker::Expect(components.betting_history[0].amount() == 5.0f,
                "parsed amount should match");
  poker::Expect(components.betting_history[0].player() == 1,
                "parsed player should match");

  return 0;
}
