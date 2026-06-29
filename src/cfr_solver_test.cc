#include "src/cfr_solver.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
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

Hand MakeHand(int first_rank, Suit first_suit, int second_rank, Suit second_suit) {
  Hand hand;
  Card* first = hand.add_cards();
  first->set_rank(first_rank);
  first->set_suit(first_suit);
  Card* second = hand.add_cards();
  second->set_rank(second_rank);
  second->set_suit(second_suit);
  return hand;
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

void CheckSaveStrategyUsesReadableActions() {
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
  node.legal_actions.push_back(MakeAction(ActionType::RAISE, 5));

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path = std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/strategy.txt";
  solver.save_strategy(path);
  solver.load_strategy(path);

  std::ifstream file(path);
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  Expect(contents.find("raise 5 ") != std::string::npos,
         "saved strategy should use readable action names");
  Expect(contents.find("4000005") == std::string::npos,
         "saved strategy should not expose encoded action keys");
}

void CheckLoadStrategyPopulatesEquilibriumStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(config);
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/loaded_strategy.txt";
  {
    std::ofstream file(path);
    file << "loaded_info_set\n";
    file << "fold 0.25\n";
    file << "call 3 0.75\n";
    file << "END_INFO_SET\n";
  }

  solver.load_strategy(path);

  const Strategy strategy = solver.get_equilibrium_strategy();
  const auto action_probs = strategy.get_strategy("loaded_info_set");
  Expect(action_probs.size() == 2, "loaded strategy should expose actions");
  Expect(std::abs(action_probs.at(TestActionKey(ActionType::FOLD)) - 0.25) <
             0.000001,
         "loaded strategy should keep fold probability");
  Expect(std::abs(action_probs.at(TestActionKey(ActionType::CALL, 3)) - 0.75) <
             0.000001,
         "loaded strategy should keep call probability");
}

void CheckRunUsesConfiguredBlinds() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_small_blind(2);
  config.set_big_blind(5);
  config.set_max_depth(1);

  CFRSolver solver(config);
  solver.run(1);

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/configured_blinds_strategy.txt";
  solver.save_strategy(path);

  std::ifstream file(path);
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  Expect(contents.find("call 3 ") != std::string::npos,
         "configured blinds should set the root call amount");
}

void CheckRunUpdatesExpectedValue() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  CFRSolver solver(config);
  solver.run(1);

  double player_a_ev = solver.get_expected_value(0);
  double player_b_ev = solver.get_expected_value(1);
  Expect(std::abs(player_a_ev + (1.0 / 3.0)) < 0.000001,
         "root EV should average completed CFR traversals");
  Expect(std::abs(player_a_ev + player_b_ev) < 0.000001,
         "heads-up EV should be zero-sum");
}

BoardState FoldedState(int folded_player) {
  BoardState state;
  state.set_pot(10);
  state.set_folded_player(folded_player);
  state.add_player_contribution(5);
  state.add_player_contribution(5);
  return state;
}

void CheckTerminalUtilityBeatsDepthLimit() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state = FoldedState(1);
  node.is_terminal = true;

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  double value =
      solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 5.0, "terminal utility should be returned at the depth limit");
}

void CheckDepthLimitUsesShowdownUtility() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_pot(20);
  node.state.set_street(Street::RIVER);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  node.state.set_player_to_act(1);
  *node.state.mutable_history()->add_actions() = MakeAction(ActionType::CHECK);
  *node.state.mutable_history()->add_actions() = MakeAction(ActionType::CHECK);
  AddCard(&node.state, 2, Suit::HEARTS);
  AddCard(&node.state, 7, Suit::DIAMONDS);
  AddCard(&node.state, 9, Suit::CLUBS);
  AddCard(&node.state, 11, Suit::SPADES);
  AddCard(&node.state, 12, Suit::DIAMONDS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::vector<double> reach_probabilities = {1.0, 1.0};
  double value =
      solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 10.0, "depth cutoff should use showdown utility when available");
}

void CheckDepthLimitDoesNotScoreUncalledBet() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_pot(15);
  node.state.set_street(Street::RIVER);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(5);
  node.state.set_player_to_act(1);
  AddCard(&node.state, 2, Suit::HEARTS);
  AddCard(&node.state, 7, Suit::DIAMONDS);
  AddCard(&node.state, 9, Suit::CLUBS);
  AddCard(&node.state, 11, Suit::SPADES);
  AddCard(&node.state, 12, Suit::DIAMONDS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::vector<double> reach_probabilities = {1.0, 1.0};
  double value =
      solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 0.0, "depth cutoff should not score unresolved bets");
}

void CheckZeroMaxDepthDoesNotCutOff() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node root;
  root.state.set_player_to_act(0);
  root.player_to_act = 0;
  root.legal_actions.push_back(MakeAction(ActionType::CHECK));

  GameTree::Node* first_child = new GameTree::Node();
  first_child->state.set_player_to_act(1);
  first_child->player_to_act = 1;
  first_child->legal_actions.push_back(MakeAction(ActionType::CHECK));
  root.children[TestActionKey(ActionType::CHECK)] = first_child;

  GameTree::Node* second_child = new GameTree::Node();
  second_child->state.set_player_to_act(0);
  second_child->player_to_act = 0;
  second_child->legal_actions.push_back(MakeAction(ActionType::CHECK));
  first_child->children[TestActionKey(ActionType::CHECK)] = second_child;

  GameTree::Node* terminal_child = new GameTree::Node();
  terminal_child->state = FoldedState(1);
  terminal_child->is_terminal = true;
  second_child->children[TestActionKey(ActionType::CHECK)] = terminal_child;

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  double value =
      solver.cfr(&root, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 0);

  Expect(value == 5.0, "zero max depth should not cut off CFR traversal");
}

void CheckChanceDoesNotConsumeDepth() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.is_chance_node = true;
  node.state.set_pot(20);
  node.state.set_street(Street::TURN);
  node.state.set_all_in(true);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  AddCard(&node.state, 14, Suit::HEARTS);
  AddCard(&node.state, 14, Suit::DIAMONDS);
  AddCard(&node.state, 14, Suit::CLUBS);
  AddCard(&node.state, 2, Suit::CLUBS);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 3, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::vector<double> reach_probabilities = {1.0, 1.0};
  double value =
      solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 10.0, "chance nodes should not consume CFR depth");
}

void CheckPlayerBRegretsUsePlayerBUtility() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_player_to_act(1);
  node.player_to_act = 1;

  Action player_b_loses = MakeAction(ActionType::FOLD);
  Action player_b_wins = MakeAction(ActionType::CALL, 1);
  node.legal_actions.push_back(player_b_loses);
  node.legal_actions.push_back(player_b_wins);

  GameTree::Node* player_b_loses_child = new GameTree::Node();
  player_b_loses_child->state = FoldedState(1);
  player_b_loses_child->is_terminal = true;
  node.children[TestActionKey(ActionType::FOLD)] = player_b_loses_child;

  GameTree::Node* player_b_wins_child = new GameTree::Node();
  player_b_wins_child->state = FoldedState(0);
  player_b_wins_child->is_terminal = true;
  node.children[TestActionKey(ActionType::CALL, 1)] = player_b_wins_child;

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 2);
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 1, 0, 2);

  const Strategy strategy = solver.get_equilibrium_strategy();
  const auto action_probs = strategy.get_strategy(strategy.get_info_sets()[0]);
  Expect(action_probs.at(TestActionKey(ActionType::CALL, 1)) >
             action_probs.at(TestActionKey(ActionType::FOLD)),
         "player B should prefer the action that lowers player A utility");
}

}  // namespace

int main() {
  CheckCfrUsesLegalActions();
  CheckCfrDistinguishesActionAmounts();
  CheckSaveStrategyUsesReadableActions();
  CheckLoadStrategyPopulatesEquilibriumStrategy();
  CheckRunUsesConfiguredBlinds();
  CheckRunUpdatesExpectedValue();
  CheckTerminalUtilityBeatsDepthLimit();
  CheckDepthLimitUsesShowdownUtility();
  CheckDepthLimitDoesNotScoreUncalledBet();
  CheckZeroMaxDepthDoesNotCutOff();
  CheckChanceDoesNotConsumeDepth();
  CheckPlayerBRegretsUsePlayerBUtility();

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
