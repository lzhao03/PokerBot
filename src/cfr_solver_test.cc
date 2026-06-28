#include "src/cfr_solver.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace poker;

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Action MakeAction(ActionType type, int amount = 0) {
  Action action;
  action.set_action(type);
  action.set_amount(amount);
  return action;
}

int TestActionKey(ActionType type, int amount = 0) {
  return static_cast<int>(type) * 1000000 + amount;
}

void AddCard(BoardState* state, int rank, Suit suit) {
  Card* card = state->add_cards();
  card->set_rank(rank);
  card->set_suit(suit);
}

void CheckCfrUsesLegalActions() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_stack_a(10);
  node.state.set_stack_b(10);
  node.state.set_pot(0);
  node.state.set_street(Street::FLOP);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(0);
  node.state.add_player_contribution(0);
  node.state.set_player_to_act(0);
  AddCard(&node.state, 2, Suit::HEARTS);
  AddCard(&node.state, 3, Suit::DIAMONDS);
  AddCard(&node.state, 4, Suit::CLUBS);
  node.player_to_act = 0;
  node.legal_actions.push_back(MakeAction(ActionType::CHECK));

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  const Strategy strategy = solver.get_equilibrium_strategy();
  Expect(strategy.get_info_sets().size() == 1, "CFR should visit one info set");
  const auto action_probs = strategy.get_strategy(strategy.get_info_sets()[0]);
  Expect(action_probs.size() == 1, "strategy should only include legal actions");
  Expect(action_probs.count(TestActionKey(ActionType::CHECK)) == 1,
         "strategy should include check");
  Expect(action_probs.at(TestActionKey(ActionType::CHECK)) == 1.0,
         "single legal action should get probability 1");
}

void CheckCfrDistinguishesActionAmounts() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_stack_a(19);
  node.state.set_stack_b(18);
  node.state.set_pot(3);
  node.state.set_street(Street::PREFLOP);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(1);
  node.state.add_player_contribution(2);
  node.state.set_player_to_act(0);
  node.player_to_act = 0;
  node.legal_actions.push_back(MakeAction(ActionType::FOLD));
  node.legal_actions.push_back(MakeAction(ActionType::CALL, 1));
  node.legal_actions.push_back(MakeAction(ActionType::RAISE, 3));
  node.legal_actions.push_back(MakeAction(ActionType::RAISE, 5));
  node.legal_actions.push_back(MakeAction(ActionType::RAISE, 7));

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  const Strategy strategy = solver.get_equilibrium_strategy();
  Expect(strategy.get_info_sets().size() == 1, "CFR should visit one info set");
  const auto action_probs = strategy.get_strategy(strategy.get_info_sets()[0]);
  Expect(action_probs.size() == 5, "same action type with different amounts should not collide");
  for (const auto& action_prob : action_probs) {
    Expect(std::abs(action_prob.second - 0.2) < 0.000001,
           "initial strategy should be uniform over all legal actions");
  }
}

}  // namespace

int main() {
  CheckCfrUsesLegalActions();
  CheckCfrDistinguishesActionAmounts();

  PokerConfig config;
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(10);
  config.set_max_depth(4);

  CFRSolver solver(config);
  solver.run(1);

  if (solver.get_equilibrium_strategy().get_info_sets().empty()) {
    throw std::runtime_error("CFR did not visit any information sets");
  }

  return 0;
}
