#include "src/cfr_solver.h"
#include "src/hand_range.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poker;

namespace poker {

class CFRSolverRegretTestPeer {
 public:
  static double Regret(const CFRSolver& solver, const std::string& info_set_key,
                       int action_id) {
    auto info_set_it = solver.cumulative_regrets_.find(info_set_key);
    if (info_set_it == solver.cumulative_regrets_.end()) {
      return 0.0;
    }
    auto action_it = info_set_it->second.find(action_id);
    return action_it == info_set_it->second.end() ? 0.0 : action_it->second;
  }

  static double EvaluateNode(CFRSolver& solver, GameTree::Node* node,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand) {
    Strategy strategy;
    return solver.evaluate_strategy_node(node, player_a_hand, player_b_hand,
                                         strategy);
  }

  static double BestResponseNode(CFRSolver& solver, GameTree::Node* node,
                                 const Hand& player_a_hand,
                                 const Hand& player_b_hand,
                                 int best_response_player) {
    Strategy strategy;
    return solver.best_response_value(node, player_a_hand, player_b_hand,
                                      strategy, best_response_player);
  }

  static double BestResponseRangeNode(
      CFRSolver& solver, GameTree::Node* node,
      const Hand& best_response_hand,
      const std::vector<std::pair<Hand, double>>& opponent_hands,
      int best_response_player) {
    Strategy strategy;
    return solver.best_response_value_against_range(
        node, best_response_hand, opponent_hands, strategy, best_response_player);
  }

  static std::vector<double> CompatibleDealWeights(
      const HandRange& player_a_range,
      const HandRange& player_b_range) {
    std::vector<CFRSolver::RangeDeal> deals =
        CFRSolver::build_compatible_range_deals(
            player_a_range.get_all_weighted_combos(),
            player_b_range.get_all_weighted_combos());
    std::vector<double> weights;
    weights.reserve(deals.size());
    for (const CFRSolver::RangeDeal& deal : deals) {
      weights.push_back(deal.weight);
    }
    return weights;
  }
};

}  // namespace poker

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

Card MakeTestCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
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

bool TestHandsOverlap(const Hand& left, const Hand& right) {
  for (const Card& left_card : left.cards()) {
    for (const Card& right_card : right.cards()) {
      if (left_card.rank() == right_card.rank() &&
          left_card.suit() == right_card.suit()) {
        return true;
      }
    }
  }
  return false;
}

Hand MakeHandFromCards(const std::vector<Card>& cards) {
  Hand hand;
  for (const Card& card : cards) {
    *hand.add_cards() = card;
  }
  return hand;
}

BoardState InitialRootState(const PokerConfig& config) {
  int small_blind = config.small_blind() > 0 ? config.small_blind() : 1;
  int big_blind = config.big_blind() > 0 ? config.big_blind() : 2;
  int starting_stack = config.starting_stack_size();

  BoardState state;
  state.set_stack_a(starting_stack > small_blind ? starting_stack - small_blind : 0);
  state.set_stack_b(starting_stack > big_blind ? starting_stack - big_blind : 0);
  state.set_pot(small_blind + big_blind);
  state.set_folded_player(-1);
  state.set_street(Street::PREFLOP);
  state.set_all_in(false);
  state.set_player_to_act(0);
  state.add_player_contribution(small_blind);
  state.add_player_contribution(big_blind);
  return state;
}

void AddStrategyInfoSet(StrategySnapshot* snapshot,
                        const std::string& info_set_key,
                        const std::vector<std::pair<Action, double>>& actions) {
  StrategyInfoSetSnapshot* info_set = snapshot->add_info_sets();
  info_set->set_info_set_key(info_set_key);
  for (const auto& action_prob : actions) {
    StrategyActionSnapshot* action = info_set->add_actions();
    *action->mutable_action() = action_prob.first;
    action->set_probability(action_prob.second);
  }
}

void WriteStrategySnapshot(const std::string& path,
                           const std::string& info_set_key,
                           const std::vector<std::pair<Action, double>>& actions) {
  StrategySnapshot snapshot;
  AddStrategyInfoSet(&snapshot, info_set_key, actions);
  std::ofstream file(path, std::ios::binary);
  Expect(snapshot.SerializeToOstream(&file), "strategy snapshot should write");
}

StrategySnapshot ReadStrategySnapshot(const std::string& path) {
  StrategySnapshot snapshot;
  std::ifstream file(path, std::ios::binary);
  Expect(snapshot.ParseFromIstream(&file), "strategy snapshot should parse");
  return snapshot;
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

void CheckSaveStrategyUsesProtobufSnapshot() {
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
  std::string path = std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/strategy.pb";
  solver.save_strategy(path);
  StrategySnapshot snapshot = ReadStrategySnapshot(path);

  Expect(snapshot.info_sets_size() == 1, "saved snapshot should include one info set");
  Expect(snapshot.info_sets(0).actions_size() == 1,
         "saved snapshot should include one action");
  Expect(snapshot.info_sets(0).actions(0).action().action() ==
             ActionType::RAISE,
         "saved snapshot should store action type");
  Expect(snapshot.info_sets(0).actions(0).action().amount() == 5,
         "saved snapshot should store action amount");
  Expect(std::abs(snapshot.info_sets(0).actions(0).probability() - 1.0) <
             0.000001,
         "saved snapshot should store action probability");
  Expect(snapshot.config().starting_stack_size() == 20,
         "saved snapshot should include solver config");
  Expect(snapshot.abstraction_version() == "exact_cards_v1",
         "saved snapshot should include abstraction version");
}

void CheckLoadStrategyPopulatesEquilibriumStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(config);
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/loaded_strategy.pb";
  WriteStrategySnapshot(path, "loaded_info_set",
                        {{MakeAction(ActionType::FOLD), 0.25},
                         {MakeAction(ActionType::CALL, 3), 0.75}});

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

void CheckEvaluateLoadedStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);
  InfoSetAbstraction abstraction;
  std::string root_info_set =
      abstraction.state_to_info_set(InitialRootState(config), 0, player_a_hand);

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/loaded_eval_strategy.pb";
  WriteStrategySnapshot(path, root_info_set,
                        {{MakeAction(ActionType::FOLD), 1.0}});

  CFRSolver solver(config);
  solver.load_strategy(path);
  double value = solver.evaluate_strategy(player_a_hand, player_b_hand);
  Expect(std::abs(value + 1.0) < 0.000001,
         "evaluated fold strategy should lose the small blind");

  std::string saved_path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/saved_eval_strategy.pb";
  solver.save_strategy(saved_path);

  CFRSolver loaded(config);
  loaded.load_strategy(saved_path);
  double loaded_value = loaded.evaluate_strategy(player_a_hand, player_b_hand);
  Expect(std::abs(value - loaded_value) < 0.000001,
         "saved and loaded strategies should evaluate the same");
}

void CheckEvaluateRangeStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  HandRange player_a_range;
  player_a_range.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);
  HandRange player_b_range;
  player_b_range.add_hand_by_index(HandRange::string_to_index("KK"), 1.0);

  InfoSetAbstraction abstraction;
  BoardState root_state = InitialRootState(config);
  StrategySnapshot snapshot;
  for (const auto& combo : player_a_range.get_all_weighted_combos()) {
    AddStrategyInfoSet(
        &snapshot, abstraction.state_to_info_set(root_state, 0, combo.first),
        {{MakeAction(ActionType::FOLD), 1.0}});
  }

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/range_eval_strategy.pb";
  {
    std::ofstream file(path, std::ios::binary);
    Expect(snapshot.SerializeToOstream(&file),
           "range strategy snapshot should write");
  }

  CFRSolver solver(config);
  solver.load_strategy(path);

  std::vector<std::pair<Hand, double>> player_a_combos =
      player_a_range.get_all_weighted_combos();
  std::vector<std::pair<Hand, double>> player_b_combos =
      player_b_range.get_all_weighted_combos();
  double exact_value =
      solver.evaluate_strategy(player_a_combos[0].first, player_b_combos[0].first);
  double range_value = solver.evaluate_strategy(3, player_a_range, player_b_range);
  Expect(std::abs(range_value - exact_value) < 0.000001,
         "range evaluation should match exact fold strategy value");
  Expect(solver.calculate_exploitability(0, player_a_range, player_b_range) == 0.0,
         "zero range exploitability samples should return zero");
  Expect(solver.calculate_exploitability(1, player_a_range, player_b_range) > 0.0,
         "range fold strategy should be exploitable");
}

void CheckSingletonRangeMatchesExactEvaluationAndBestResponse() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);
  HandRange player_a_range;
  player_a_range.add_hand(player_a_hand, 1.0);
  HandRange player_b_range;
  player_b_range.add_hand(player_b_hand, 1.0);

  CFRSolver solver(config);
  InfoSetAbstraction abstraction;
  std::string root_info_set =
      abstraction.state_to_info_set(InitialRootState(config), 0, player_a_hand);
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/singleton_strategy.pb";
  WriteStrategySnapshot(path, root_info_set,
                        {{MakeAction(ActionType::FOLD), 1.0}});
  solver.load_strategy(path);

  double exact_value = solver.evaluate_strategy(player_a_hand, player_b_hand);
  double range_value = solver.evaluate_strategy(5, player_a_range, player_b_range);
  Expect(std::abs(exact_value - range_value) < 0.000001,
         "singleton range evaluation should match exact-hand evaluation");

  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;
  node.legal_actions.push_back(MakeAction(ActionType::FOLD));
  node.legal_actions.push_back(MakeAction(ActionType::CALL, 1));

  auto folded_state = [](int folded_player) {
    BoardState state;
    state.set_pot(10);
    state.set_folded_player(folded_player);
    state.add_player_contribution(5);
    state.add_player_contribution(5);
    return state;
  };

  GameTree::Node* player_a_loses_child = new GameTree::Node();
  player_a_loses_child->state = folded_state(0);
  player_a_loses_child->is_terminal = true;
  node.children[TestActionKey(ActionType::FOLD)] = player_a_loses_child;

  GameTree::Node* player_a_wins_child = new GameTree::Node();
  player_a_wins_child->state = folded_state(1);
  player_a_wins_child->is_terminal = true;
  node.children[TestActionKey(ActionType::CALL, 1)] = player_a_wins_child;

  double exact_best_response = CFRSolverRegretTestPeer::BestResponseNode(
      solver, &node, player_a_hand, player_b_hand, 0);
  double range_best_response = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, &node, player_a_hand, {{player_b_hand, 1.0}}, 0);
  Expect(std::abs(exact_best_response - range_best_response) < 0.000001,
         "singleton range best response should match exact-hand best response");
}

void CheckExploitabilityDetectsFoldStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);
  InfoSetAbstraction abstraction;
  std::string root_info_set =
      abstraction.state_to_info_set(InitialRootState(config), 0, player_a_hand);

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/exploitability_strategy.pb";
  WriteStrategySnapshot(path, root_info_set,
                        {{MakeAction(ActionType::FOLD), 1.0}});

  CFRSolver solver(config);
  solver.load_strategy(path);
  double exploitability =
      solver.calculate_exploitability(player_a_hand, player_b_hand);
  Expect(exploitability > 0.0, "fold-only strategy should be exploitable");
}

void CheckExploitabilityZeroSamples() {
  PokerConfig config;
  CFRSolver solver(config);
  Expect(solver.calculate_exploitability(0) == 0.0,
         "zero exploitability samples should return zero");
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
      std::string(test_tmpdir ? test_tmpdir : "/tmp") + "/configured_blinds_strategy.pb";
  solver.save_strategy(path);

  StrategySnapshot snapshot = ReadStrategySnapshot(path);
  Expect(snapshot.iterations_run() == 1,
         "saved snapshot should include completed iteration count");
  Expect(snapshot.config().small_blind() == 2,
         "saved snapshot should include configured small blind");
  Expect(snapshot.config().big_blind() == 5,
         "saved snapshot should include configured big blind");
  bool has_call_three = false;
  for (const StrategyInfoSetSnapshot& info_set : snapshot.info_sets()) {
    for (const StrategyActionSnapshot& action : info_set.actions()) {
      if (action.action().action() == ActionType::CALL &&
          action.action().amount() == 3) {
        has_call_three = true;
      }
    }
  }
  Expect(has_call_three, "configured blinds should set the root call amount");
}

void CheckRunUpdatesExpectedValue() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  CFRSolver solver(config);
  Expect(solver.get_iterations_run() == 0, "new solver should have no completed iterations");
  solver.run(1);
  Expect(solver.get_iterations_run() == 1, "run should record completed iterations");

  double player_a_ev = solver.get_expected_value(0);
  double player_b_ev = solver.get_expected_value(1);
  Expect(std::abs(player_a_ev + (1.0 / 3.0)) < 0.000001,
         "root EV should average completed CFR traversals");
  Expect(std::abs(player_a_ev + player_b_ev) < 0.000001,
         "heads-up EV should be zero-sum");

  solver.run(1);
  Expect(solver.get_iterations_run() == 2, "repeated run should accumulate iterations");
  player_a_ev = solver.get_expected_value(0);
  player_b_ev = solver.get_expected_value(1);
  Expect(std::abs(player_a_ev + player_b_ev) < 0.000001,
         "continued EV should stay zero-sum");
}

void CheckRunTrainsSwappedPrivateHands() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  CFRSolver solver(config);
  solver.run(1);

  const Strategy strategy = solver.get_equilibrium_strategy();
  Expect(strategy.get_info_sets().size() == 2,
         "run should train both dealt private-hand perspectives");
}

void CheckRunUsesProvidedPrivateRanges() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  player_a_range.add_hand(player_a_hand, 1.0);
  HandRange player_b_range;
  player_b_range.add_hand(player_b_hand, 1.0);

  CFRSolver solver(config);
  solver.run(1, player_a_range, player_b_range);

  InfoSetAbstraction abstraction;
  const Strategy strategy = solver.get_equilibrium_strategy();
  Expect(strategy.get_info_sets().size() == 1,
         "range run should not swap asymmetric player ranges");
  InfoSetAbstraction::InfoSetComponents components =
      abstraction.parse_info_set(strategy.get_info_sets()[0]);
  Hand trained_hand = MakeHandFromCards(components.player_cards);
  Expect(HandRange::hand_to_index(trained_hand) ==
             HandRange::hand_to_index(player_a_hand),
         "range run should train the supplied player A hand class");
}

void CheckRangeExpansionUsesExactCombos() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange aces;
  aces.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);
  std::vector<std::pair<Hand, double>> combos = aces.get_all_weighted_combos();
  Expect(combos.size() == 6, "AA should expand to six exact combos");

  HandRange ace_king_suited;
  ace_king_suited.add_hand_by_index(HandRange::string_to_index("AKs"), 1.0);
  Expect(ace_king_suited.get_all_weighted_combos().size() == 4,
         "AKs should expand to four exact combos");

  HandRange ace_king_offsuit;
  ace_king_offsuit.add_hand_by_index(HandRange::string_to_index("AKo"), 1.0);
  Expect(ace_king_offsuit.get_all_weighted_combos().size() == 12,
         "AKo should expand to twelve exact combos");

  bool has_disjoint_aces = false;
  for (const auto& left : combos) {
    for (const auto& right : combos) {
      if (!TestHandsOverlap(left.first, right.first)) {
        has_disjoint_aces = true;
      }
    }
  }
  Expect(has_disjoint_aces, "AA range should contain disjoint exact combos");

  CFRSolver solver(config);
  solver.run(1, aces, aces);
  Expect(!solver.get_equilibrium_strategy().get_info_sets().empty(),
         "same hand class ranges should train from disjoint exact combos");
}

void CheckRangeSamplingRejectsEmptyRange() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  HandRange player_b_range;
  player_b_range.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);

  CFRSolver solver(config);
  bool threw = false;
  try {
    solver.run(1, player_a_range, player_b_range);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "range sampling should reject empty ranges");
}

void CheckCompatibleDealWeightsUseProductWeights() {
  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_a_compatible = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  Hand player_b_compatible = MakeHand(12, Suit::SPADES, 12, Suit::HEARTS);

  HandRange player_a_range;
  player_a_range.add_hand(blocked, 2.0);
  player_a_range.add_hand(player_a_compatible, 3.0);
  HandRange player_b_range;
  player_b_range.add_hand(blocked, 5.0);
  player_b_range.add_hand(player_b_compatible, 7.0);

  std::vector<double> weights =
      CFRSolverRegretTestPeer::CompatibleDealWeights(player_a_range,
                                                     player_b_range);
  std::sort(weights.begin(), weights.end());

  Expect(weights.size() == 3, "compatible deal builder should skip overlaps");
  Expect(std::abs(weights[0] - 14.0) < 0.000001,
         "deal weight should be player hand weight product");
  Expect(std::abs(weights[1] - 15.0) < 0.000001,
         "deal weight should include compatible player B hands");
  Expect(std::abs(weights[2] - 21.0) < 0.000001,
         "deal weight should include compatible player A and B hands");
}

void CheckRangeSamplingRejectsOnlyOverlappingHands() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  HandRange player_a_range;
  player_a_range.add_hand(blocked, 1.0);
  HandRange player_b_range;
  player_b_range.add_hand(blocked, 1.0);

  CFRSolver solver(config);
  bool threw = false;
  try {
    solver.run(1, player_a_range, player_b_range);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "range sampling should reject fully overlapping exact ranges");
}

void CheckRangeSamplingSkipsOverlappingDeals() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand compatible = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  player_a_range.add_hand(blocked, 1000.0);
  player_a_range.add_hand(compatible, 1.0);
  HandRange player_b_range;
  player_b_range.add_hand(blocked, 1.0);

  CFRSolver solver(config);
  solver.run(3, player_a_range, player_b_range);

  InfoSetAbstraction abstraction;
  const Strategy strategy = solver.get_equilibrium_strategy();
  Expect(strategy.get_info_sets().size() == 1,
         "range sampling should only train compatible private deals");
  InfoSetAbstraction::InfoSetComponents components =
      abstraction.parse_info_set(strategy.get_info_sets()[0]);
  Hand trained_hand = MakeHandFromCards(components.player_cards);
  Expect(HandRange::hand_to_index(trained_hand) ==
             HandRange::hand_to_index(compatible),
         "range sampling should skip overlapping private-card deals");
}

void CheckRunLoggingUsesConfig() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  std::ostringstream quiet_output;
  std::streambuf* original = std::cout.rdbuf(quiet_output.rdbuf());
  CFRSolver quiet_solver(config);
  quiet_solver.run(1);
  std::cout.rdbuf(original);
  Expect(quiet_output.str().empty(), "run should be quiet by default");

  config.set_enable_logging(true);
  std::ostringstream logged_output;
  original = std::cout.rdbuf(logged_output.rdbuf());
  CFRSolver logged_solver(config);
  logged_solver.run(1);
  std::cout.rdbuf(original);
  Expect(logged_output.str().find("Starting CFR iterations") != std::string::npos,
         "run should log when enable_logging is set");
  Expect(logged_output.str().find("Iterations run: 1") != std::string::npos,
         "run should log completed iteration count");
  Expect(logged_output.str().find("Information sets: 2") != std::string::npos,
         "run should log trained info set count");
  Expect(logged_output.str().find("Player A average EV:") != std::string::npos,
         "run should log average root EV");
}

void CheckRunProducesDeterministicStrategyShape() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  CFRSolver first_solver(config);
  CFRSolver second_solver(config);
  first_solver.run(2);
  second_solver.run(2);

  const Strategy first = first_solver.get_equilibrium_strategy();
  const Strategy second = second_solver.get_equilibrium_strategy();
  const auto& first_strategy = first.get_full_strategy();
  const auto& second_strategy = second.get_full_strategy();
  Expect(first_strategy.size() == second_strategy.size(),
         "same config should produce same number of info sets");

  for (const auto& info_set_strategy : first_strategy) {
    auto second_info_set = second_strategy.find(info_set_strategy.first);
    Expect(second_info_set != second_strategy.end(),
           "same config should produce same info set keys");
    Expect(info_set_strategy.second.size() == second_info_set->second.size(),
           "same config should produce same action count");
    for (const auto& action_prob : info_set_strategy.second) {
      auto second_action = second_info_set->second.find(action_prob.first);
      Expect(second_action != second_info_set->second.end(),
             "same config should produce same action keys");
      Expect(std::abs(action_prob.second - second_action->second) < 0.000001,
             "same config should produce same action probabilities");
    }
  }
}

void CheckRepeatedRunMatchesSingleRun() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  player_a_range.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);
  HandRange player_b_range;
  player_b_range.add_hand_by_index(HandRange::string_to_index("KK"), 1.0);

  CFRSolver continued(config);
  continued.run(10, player_a_range, player_b_range);
  continued.run(10, player_a_range, player_b_range);

  CFRSolver single(config);
  single.run(20, player_a_range, player_b_range);

  const Strategy continued_equilibrium = continued.get_equilibrium_strategy();
  const Strategy single_equilibrium = single.get_equilibrium_strategy();
  const auto& continued_strategy = continued_equilibrium.get_full_strategy();
  const auto& single_strategy = single_equilibrium.get_full_strategy();
  Expect(continued_strategy.size() == single_strategy.size(),
         "continued run should visit the same info sets as one longer run");

  for (const auto& info_set_strategy : single_strategy) {
    auto continued_info_set = continued_strategy.find(info_set_strategy.first);
    Expect(continued_info_set != continued_strategy.end(),
           "continued run should include each info set from the longer run");
    Expect(continued_info_set->second.size() == info_set_strategy.second.size(),
           "continued run should have the same action count");
    for (const auto& action_prob : info_set_strategy.second) {
      auto continued_action = continued_info_set->second.find(action_prob.first);
      Expect(continued_action != continued_info_set->second.end(),
             "continued run should include each action");
      Expect(std::abs(continued_action->second - action_prob.second) < 0.000001,
             "continued run should keep CFR+ iteration weighting");
    }
  }
}

BoardState FoldedState(int folded_player) {
  BoardState state;
  state.set_pot(10);
  state.set_folded_player(folded_player);
  state.add_player_contribution(5);
  state.add_player_contribution(5);
  return state;
}

BoardState ShowdownState(const std::vector<Card>& board_cards) {
  BoardState state;
  state.set_pot(20);
  state.set_street(Street::RIVER);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(10);
  state.add_player_contribution(10);
  for (const Card& card : board_cards) {
    *state.add_cards() = card;
  }
  return state;
}

GameTree::Node* TerminalShowdownNode(const std::vector<Card>& board_cards) {
  GameTree::Node* node = new GameTree::Node();
  node->state = ShowdownState(board_cards);
  node->is_terminal = true;
  return node;
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

void CheckChanceSamplesVisitMultipleBoards() {
  BoardState state;
  state.set_stack_a(10);
  state.set_stack_b(10);
  state.set_pot(0);
  state.set_street(Street::PREFLOP);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.set_player_to_act(-1);
  state.add_player_contribution(0);
  state.add_player_contribution(0);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);

  PokerConfig one_sample_config;
  one_sample_config.set_starting_stack_size(10);
  CFRSolver one_sample_solver(one_sample_config);
  GameTree::Node one_sample_node;
  one_sample_node.is_chance_node = true;
  one_sample_node.state = state;
  std::vector<double> one_sample_reach = {1.0, 1.0};
  one_sample_solver.cfr(&one_sample_node, player_a_hand, player_b_hand,
                        one_sample_reach, 0, 0, 1);

  PokerConfig three_sample_config;
  three_sample_config.set_starting_stack_size(10);
  three_sample_config.set_chance_samples(3);
  CFRSolver three_sample_solver(three_sample_config);
  GameTree::Node three_sample_node;
  three_sample_node.is_chance_node = true;
  three_sample_node.state = state;
  std::vector<double> three_sample_reach = {1.0, 1.0};
  three_sample_solver.cfr(&three_sample_node, player_a_hand, player_b_hand,
                          three_sample_reach, 0, 0, 1);

  Expect(one_sample_solver.get_equilibrium_strategy().get_info_sets().size() == 1,
         "default chance sampling should visit one sampled board");
  Expect(three_sample_solver.get_equilibrium_strategy().get_info_sets().size() >
             one_sample_solver.get_equilibrium_strategy().get_info_sets().size(),
         "configured chance samples should visit more sampled boards");
}

void CheckEvaluationUsesChanceSamples() {
  GameTree::Node node;
  node.is_chance_node = true;
  node.state.set_pot(20);
  node.state.set_street(Street::TURN);
  node.state.set_all_in(true);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  AddCard(&node.state, 2, Suit::HEARTS);
  AddCard(&node.state, 7, Suit::DIAMONDS);
  AddCard(&node.state, 9, Suit::CLUBS);
  AddCard(&node.state, 11, Suit::SPADES);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(10, Suit::HEARTS, 10, Suit::CLUBS);

  PokerConfig one_sample_config;
  one_sample_config.set_starting_stack_size(10);
  CFRSolver one_sample_solver(one_sample_config);
  double first = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, &node, player_a_hand, player_b_hand);
  double second = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, &node, player_a_hand, player_b_hand);
  double third = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, &node, player_a_hand, player_b_hand);
  double expected_average = (first + second + third) / 3.0;
  Expect(std::abs(first - expected_average) > 0.000001,
         "chance sample fixture should have varied outcomes");

  PokerConfig three_sample_config;
  three_sample_config.set_starting_stack_size(10);
  three_sample_config.set_chance_samples(3);
  CFRSolver three_sample_solver(three_sample_config);
  double sampled_average = CFRSolverRegretTestPeer::EvaluateNode(
      three_sample_solver, &node, player_a_hand, player_b_hand);
  Expect(std::abs(sampled_average - expected_average) < 0.000001,
         "strategy evaluation should average configured chance samples");

  CFRSolver one_sample_best_response(one_sample_config);
  double first_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, &node, player_a_hand, player_b_hand, 0);
  double second_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, &node, player_a_hand, player_b_hand, 0);
  double third_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, &node, player_a_hand, player_b_hand, 0);
  double expected_response_average =
      (first_response + second_response + third_response) / 3.0;

  CFRSolver three_sample_best_response(three_sample_config);
  double sampled_response_average = CFRSolverRegretTestPeer::BestResponseNode(
      three_sample_best_response, &node, player_a_hand, player_b_hand, 0);
  Expect(std::abs(sampled_response_average - expected_response_average) <
             0.000001,
         "best response should average configured chance samples");
}

void CheckBestResponseActionSelectsBestLegalAction() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  Action player_a_loses = MakeAction(ActionType::FOLD);
  Action player_a_wins = MakeAction(ActionType::CALL, 1);
  node.legal_actions.push_back(player_a_loses);
  node.legal_actions.push_back(player_a_wins);

  GameTree::Node* player_a_loses_child = new GameTree::Node();
  player_a_loses_child->state = FoldedState(0);
  player_a_loses_child->is_terminal = true;
  node.children[TestActionKey(ActionType::FOLD)] = player_a_loses_child;

  GameTree::Node* player_a_wins_child = new GameTree::Node();
  player_a_wins_child->state = FoldedState(1);
  player_a_wins_child->is_terminal = true;
  node.children[TestActionKey(ActionType::CALL, 1)] = player_a_wins_child;

  Hand player_a_hand;
  Hand player_b_hand;
  Action best_action =
      solver.get_best_response_action(&node, player_a_hand, player_b_hand, 0);
  Expect(best_action.action() == ActionType::CALL,
         "best response should select the highest-value legal action");
  Expect(best_action.amount() == 1,
         "best response should preserve selected action amount");

  Action opponent_turn_action =
      solver.get_best_response_action(&node, player_a_hand, player_b_hand, 1);
  Expect(opponent_turn_action.action() == ActionType::NO_ACTION,
         "best response should return no action away from its turn");
}

void CheckRangeBestResponseDoesNotKnowOpponentHand() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  Action first_board = MakeAction(ActionType::CHECK);
  Action second_board = MakeAction(ActionType::CALL, 1);
  node.legal_actions.push_back(first_board);
  node.legal_actions.push_back(second_board);

  node.children[TestActionKey(ActionType::CHECK)] = TerminalShowdownNode(
      {MakeTestCard(13, Suit::SPADES), MakeTestCard(13, Suit::HEARTS),
       MakeTestCard(3, Suit::CLUBS), MakeTestCard(4, Suit::DIAMONDS),
       MakeTestCard(5, Suit::SPADES)});
  node.children[TestActionKey(ActionType::CALL, 1)] = TerminalShowdownNode(
      {MakeTestCard(2, Suit::SPADES), MakeTestCard(2, Suit::HEARTS),
       MakeTestCard(3, Suit::CLUBS), MakeTestCard(4, Suit::DIAMONDS),
       MakeTestCard(5, Suit::SPADES)});

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand kings = MakeHand(13, Suit::CLUBS, 13, Suit::DIAMONDS);
  Hand twos = MakeHand(2, Suit::CLUBS, 2, Suit::DIAMONDS);
  std::vector<std::pair<Hand, double>> opponent_hands = {{kings, 1.0},
                                                         {twos, 1.0}};

  double clairvoyant_average =
      (CFRSolverRegretTestPeer::BestResponseNode(solver, &node, player_a_hand,
                                                 kings, 0) +
       CFRSolverRegretTestPeer::BestResponseNode(solver, &node, player_a_hand,
                                                 twos, 0)) /
      2.0;
  double range_value = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, &node, player_a_hand, opponent_hands, 0);

  Expect(clairvoyant_average > 0.0,
         "exact-hand best response can overuse hidden opponent cards");
  Expect(std::abs(range_value) < 0.000001,
         "range best response should choose one action for the opponent range");
}

void CheckRangeBestResponseChanceUsesSampledOpponent() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_chance_samples(1);

  CFRSolver solver(config);
  GameTree::Node node;
  node.is_chance_node = true;
  node.state.set_pot(20);
  node.state.set_street(Street::TURN);
  node.state.set_all_in(true);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  AddCard(&node.state, 2, Suit::CLUBS);
  AddCard(&node.state, 2, Suit::DIAMONDS);
  AddCard(&node.state, 2, Suit::HEARTS);
  AddCard(&node.state, 3, Suit::CLUBS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand quads = MakeHand(2, Suit::SPADES, 7, Suit::SPADES);
  Hand air = MakeHand(13, Suit::CLUBS, 12, Suit::DIAMONDS);
  std::vector<std::pair<Hand, double>> opponent_hands = {{quads, 1.0},
                                                         {air, 1.0}};

  double value = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, &node, player_a_hand, opponent_hands, 0);

  Expect(std::abs(std::abs(value) - 10.0) < 0.000001,
         "chance range best response should score the sampled opponent only");
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

void CheckCfrPlusClipsNegativeRegrets() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  Action player_a_loses = MakeAction(ActionType::FOLD);
  Action player_a_wins = MakeAction(ActionType::CALL, 1);
  node.legal_actions.push_back(player_a_loses);
  node.legal_actions.push_back(player_a_wins);

  GameTree::Node* player_a_loses_child = new GameTree::Node();
  player_a_loses_child->state = FoldedState(0);
  player_a_loses_child->is_terminal = true;
  node.children[TestActionKey(ActionType::FOLD)] = player_a_loses_child;

  GameTree::Node* player_a_wins_child = new GameTree::Node();
  player_a_wins_child->state = FoldedState(1);
  player_a_wins_child->is_terminal = true;
  node.children[TestActionKey(ActionType::CALL, 1)] = player_a_wins_child;

  Hand player_a_hand;
  Hand player_b_hand;
  InfoSetAbstraction abstraction;
  std::string info_set_key =
      abstraction.state_to_info_set(node.state, 0, player_a_hand);
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  Expect(CFRSolverRegretTestPeer::Regret(
             solver, info_set_key, TestActionKey(ActionType::FOLD)) == 0.0,
         "CFR+ should clip negative cumulative regrets");
  Expect(CFRSolverRegretTestPeer::Regret(
             solver, info_set_key, TestActionKey(ActionType::CALL, 1)) > 0.0,
         "positive cumulative regret should remain positive");
}

void CheckCfrPlusWeightsLaterStrategies() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(config);
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  Action player_a_loses = MakeAction(ActionType::FOLD);
  Action player_a_wins = MakeAction(ActionType::CALL, 1);
  node.legal_actions.push_back(player_a_loses);
  node.legal_actions.push_back(player_a_wins);

  GameTree::Node* player_a_loses_child = new GameTree::Node();
  player_a_loses_child->state = FoldedState(0);
  player_a_loses_child->is_terminal = true;
  node.children[TestActionKey(ActionType::FOLD)] = player_a_loses_child;

  GameTree::Node* player_a_wins_child = new GameTree::Node();
  player_a_wins_child->state = FoldedState(1);
  player_a_wins_child->is_terminal = true;
  node.children[TestActionKey(ActionType::CALL, 1)] = player_a_wins_child;

  Hand player_a_hand;
  Hand player_b_hand;
  std::vector<double> reach_probabilities = {1.0, 1.0};
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);
  solver.cfr(&node, player_a_hand, player_b_hand, reach_probabilities, 1, 0, 1);

  const Strategy strategy = solver.get_equilibrium_strategy();
  const auto action_probs = strategy.get_strategy(strategy.get_info_sets()[0]);
  Expect(std::abs(action_probs.at(TestActionKey(ActionType::CALL, 1)) -
                  (5.0 / 6.0)) < 0.000001,
         "CFR+ average strategy should weight later iterations linearly");
}

}  // namespace

int main() {
  CheckCfrUsesLegalActions();
  CheckCfrDistinguishesActionAmounts();
  CheckSaveStrategyUsesProtobufSnapshot();
  CheckLoadStrategyPopulatesEquilibriumStrategy();
  CheckEvaluateLoadedStrategy();
  CheckEvaluateRangeStrategy();
  CheckSingletonRangeMatchesExactEvaluationAndBestResponse();
  CheckExploitabilityDetectsFoldStrategy();
  CheckExploitabilityZeroSamples();
  CheckRunUsesConfiguredBlinds();
  CheckRunUpdatesExpectedValue();
  CheckRunTrainsSwappedPrivateHands();
  CheckRunUsesProvidedPrivateRanges();
  CheckRangeExpansionUsesExactCombos();
  CheckRangeSamplingRejectsEmptyRange();
  CheckCompatibleDealWeightsUseProductWeights();
  CheckRangeSamplingRejectsOnlyOverlappingHands();
  CheckRangeSamplingSkipsOverlappingDeals();
  CheckRunLoggingUsesConfig();
  CheckRunProducesDeterministicStrategyShape();
  CheckRepeatedRunMatchesSingleRun();
  CheckTerminalUtilityBeatsDepthLimit();
  CheckDepthLimitUsesShowdownUtility();
  CheckDepthLimitDoesNotScoreUncalledBet();
  CheckZeroMaxDepthDoesNotCutOff();
  CheckChanceDoesNotConsumeDepth();
  CheckChanceSamplesVisitMultipleBoards();
  CheckEvaluationUsesChanceSamples();
  CheckBestResponseActionSelectsBestLegalAction();
  CheckRangeBestResponseDoesNotKnowOpponentHand();
  CheckRangeBestResponseChanceUsesSampledOpponent();
  CheckPlayerBRegretsUsePlayerBUtility();
  CheckCfrPlusClipsNegativeRegrets();
  CheckCfrPlusWeightsLaterStrategies();

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
