#include "src/cfr_solver.h"
#include "src/card_utils.h"
#include "src/continuation_value.h"
#include "src/hand_range.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kActionKeyMultiplier = 1000000;
constexpr char kInfoSetAbstractionVersion[] = "exact_cards_v1";
constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kParallelBestResponseSampleThreshold = 32;

int ActionKey(const Action& action) {
  // ponytail: amounts are whole chips today; use a structured key if fractional chips matter.
  return static_cast<int>(action.action()) * kActionKeyMultiplier +
         static_cast<int>(std::lround(action.amount()));
}

Action ActionFromKey(int action_key) {
  Action action;
  action.set_action(static_cast<ActionType>(action_key / kActionKeyMultiplier));
  action.set_amount(action_key % kActionKeyMultiplier);
  return action;
}

bool HandsOverlap(const Hand& left, const Hand& right) {
  for (const Card& left_card : left.cards()) {
    for (const Card& right_card : right.cards()) {
      if (SameCard(left_card, right_card)) {
        return true;
      }
    }
  }
  return false;
}

bool HandOverlapsBoard(const Hand& hand, const BoardState& state) {
  for (const Card& hand_card : hand.cards()) {
    for (const Card& board_card : state.cards()) {
      if (SameCard(hand_card, board_card)) {
        return true;
      }
    }
  }
  return false;
}

double TotalWeight(const std::vector<std::pair<Hand, double>>& hands) {
  double total = 0.0;
  for (const auto& hand : hands) {
    total += hand.second;
  }
  return total;
}

std::vector<std::pair<Hand, double>> CompatibleHands(
    const std::vector<std::pair<Hand, double>>& hands,
    const Hand& known_hand,
    const BoardState& state) {
  std::vector<std::pair<Hand, double>> compatible_hands;
  for (const auto& hand : hands) {
    if (!HandsOverlap(hand.first, known_hand) &&
        !HandOverlapsBoard(hand.first, state)) {
      compatible_hands.push_back(hand);
    }
  }
  return compatible_hands;
}

std::vector<double> WeightsFor(
    const std::vector<std::pair<Hand, double>>& hands) {
  std::vector<double> weights;
  weights.reserve(hands.size());
  for (const auto& hand : hands) {
    weights.push_back(hand.second);
  }
  return weights;
}

std::string RoundClosureKey(const ActionHistory& history) {
  if (history.actions_size() == 0) {
    return "H0";
  }

  const Action& last = history.actions(history.actions_size() - 1);
  std::ostringstream oss;
  oss << "H" << (history.actions_size() > 1 ? "2+" : "1") << ":L"
      << last.player() << ":" << static_cast<int>(last.action()) << ":"
      << last.amount();
  return oss.str();
}

std::string CanonicalPublicStateKey(const BoardState& state) {
  std::ostringstream oss;
  oss << "S" << static_cast<int>(state.street()) << ":P" << state.pot()
      << ":SA" << state.stack_a() << ":SB" << state.stack_b() << ":AI"
      << state.all_in() << ":F" << state.folded_player() << ":T"
      << state.player_to_act() << ":C[";
  for (int i = 0; i < state.player_contribution_size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << state.player_contribution(i);
  }
  oss << "]:B[";
  for (int i = 0; i < state.cards_size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    const Card& card = state.cards(i);
    oss << card.rank() << ":" << static_cast<int>(card.suit());
  }
  oss << "]:" << RoundClosureKey(state.history());
  return oss.str();
}

struct WeightedHands {
  std::vector<std::pair<Hand, double>> hands;
  std::vector<double> weights;
  std::discrete_distribution<size_t> distribution;

  void reset(std::vector<std::pair<Hand, double>> new_hands) {
    hands = std::move(new_hands);
    weights = WeightsFor(hands);
    distribution =
        std::discrete_distribution<size_t>(weights.begin(), weights.end());
  }
};

Hand SampleWeightedHand(WeightedHands* hands, std::mt19937* rng) {
  return hands->hands[hands->distribution(*rng)].first;
}

double StrategyActionProbability(const Strategy& strategy,
                                 const std::string& info_set_key,
                                 const std::vector<Action>& legal_actions,
                                 int action_id) {
  double probability_sum = 0.0;
  for (const Action& legal_action : legal_actions) {
    probability_sum +=
        strategy.get_action_probability(info_set_key, ActionKey(legal_action));
  }
  if (probability_sum > 0.0) {
    return strategy.get_action_probability(info_set_key, action_id) /
           probability_sum;
  }
  return legal_actions.empty() ? 0.0 : 1.0 / legal_actions.size();
}

int ChanceSamples(const PokerConfig& config) {
  return std::max(1, config.chance_samples());
}

int WorkerCountForSamples(int samples) {
  int worker_count = static_cast<int>(std::thread::hardware_concurrency());
  if (worker_count <= 0) {
    worker_count = 1;
  }
  return std::min(worker_count, samples);
}

template <typename EvaluateChild>
double SampleChanceValue(GameTree* game_tree,
                         GameTree::Node* node,
                         const Hand& player_a_hand,
                         const Hand& player_b_hand,
                         int samples,
                         std::mt19937* rng,
                         EvaluateChild evaluate_child) {
  double value = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<Card> cards =
        SampleStreetCards(node->state, player_a_hand, player_b_hand, rng);
    GameTree::Node* child_node =
        game_tree->create_chance_child_node(node, cards);
    value += evaluate_child(child_node);
    delete child_node;
  }
  return value / samples;
}

BoardState DefaultInitialState(const PokerConfig& config) {
  const int small_blind = config.small_blind() > 0 ? config.small_blind() : 1;
  const int big_blind = config.big_blind() > 0 ? config.big_blind() : 2;
  const int starting_stack = config.starting_stack_size();

  BoardState initial_state;
  initial_state.set_stack_a(std::max(0, starting_stack - small_blind));
  initial_state.set_stack_b(std::max(0, starting_stack - big_blind));
  initial_state.set_pot(small_blind + big_blind);
  initial_state.set_folded_player(-1);
  initial_state.set_street(Street::PREFLOP);
  initial_state.set_all_in(false);
  initial_state.set_player_to_act(0);
  initial_state.add_player_contribution(small_blind);
  initial_state.add_player_contribution(big_blind);
  return initial_state;
}

}  // namespace

CFRSolver::CFRSolver(const PokerConfig& config)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>()) {
}

CFRSolver::CFRSolver(const PokerConfig& config,
                     const BoardState& initial_state)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>(),
                std::make_shared<BettingRoundTerminalValueProvider>(),
                initial_state) {
}

CFRSolver::CFRSolver(const PokerConfig& config,
                     std::shared_ptr<TerminalUtilityCache> utility_cache)
    : CFRSolver(config, std::move(utility_cache),
                std::make_shared<BettingRoundTerminalValueProvider>()) {
}

CFRSolver::CFRSolver(
    const PokerConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider)
    : CFRSolver(config, std::move(utility_cache),
                std::move(continuation_value_provider),
                DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(
    const PokerConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider,
    BoardState initial_state)
  : config_(config),
    initial_state_(std::move(initial_state)),
    game_tree_(new GameTree(config)),
    hand_evaluator_(new HandEvaluator()),
    info_set_abstraction_(new InfoSetAbstraction()),
    rng_(12345),
    cumulative_root_utility_(0.0),
    iterations_run_(0),
    cfr_update_count_(0),
    utility_cache_(std::move(utility_cache)),
    continuation_value_provider_(std::move(continuation_value_provider)) {
}

CFRSolver::~CFRSolver() {
  delete game_tree_;
  delete hand_evaluator_;
  delete info_set_abstraction_;
}

void CFRSolver::set_continuation_value_provider(
    std::shared_ptr<ContinuationValueProvider> provider) {
  if (provider == nullptr) {
    throw std::invalid_argument("Continuation value provider cannot be null");
  }
  continuation_value_provider_ = std::move(provider);
}

GameTree::Node* CFRSolver::get_or_build_root() {
  GameTree::Node* root = game_tree_->get_root();
  if (root == nullptr) {
    return game_tree_->build_tree(initial_state_);
  }
  return root;
}

std::vector<CFRSolver::RangeDeal> CFRSolver::build_compatible_range_deals(
    const std::vector<std::pair<Hand, double>>& player_a_hands,
    const std::vector<std::pair<Hand, double>>& player_b_hands) {
  std::vector<RangeDeal> deals;
  for (const auto& player_a_hand : player_a_hands) {
    if (player_a_hand.second <= 0.0) {
      continue;
    }
    for (const auto& player_b_hand : player_b_hands) {
      if (player_b_hand.second <= 0.0 ||
          HandsOverlap(player_a_hand.first, player_b_hand.first)) {
        continue;
      }
      deals.push_back({player_a_hand.first, player_b_hand.first,
                       player_a_hand.second * player_b_hand.second});
    }
  }
  return deals;
}

void CFRSolver::run(int iterations) {
  run_iterations(iterations, nullptr, nullptr, true);
}

void CFRSolver::run(int iterations, const Hand& player_a_hand,
                    const Hand& player_b_hand) {
  if (iterations <= 0) {
    return;
  }

  GameTree::Node* root = get_or_build_root();
  const int max_depth = config_.max_depth();
  for (int i = 0; i < iterations; ++i) {
    std::vector<double> reach_probabilities(2, 1.0);
    cumulative_root_utility_ += cfr(root, player_a_hand, player_b_hand,
                                    reach_probabilities, iterations_run_, 0,
                                    max_depth);
    ++iterations_run_;
  }
}

void CFRSolver::run(int iterations, const HandRange& player_a_range,
                    const HandRange& player_b_range) {
  run_iterations(iterations, &player_a_range, &player_b_range, false);
}

void CFRSolver::run_iterations(int iterations, const HandRange* player_a_range,
                               const HandRange* player_b_range,
                               bool train_swapped) {
  std::vector<RangeDeal> range_deals;
  std::vector<double> range_deal_weights;
  std::discrete_distribution<size_t> range_deal_distribution;
  if (iterations > 0 && player_a_range != nullptr &&
      player_b_range != nullptr) {
    range_deals = build_compatible_range_deals(
        player_a_range->get_all_weighted_combos(),
        player_b_range->get_all_weighted_combos());
    if (range_deals.empty()) {
      throw std::invalid_argument(
          "Could not sample non-overlapping hands from ranges");
    }
    range_deal_weights.reserve(range_deals.size());
    for (const RangeDeal& deal : range_deals) {
      range_deal_weights.push_back(deal.weight);
    }
    range_deal_distribution = std::discrete_distribution<size_t>(
        range_deal_weights.begin(), range_deal_weights.end());
  }

  const bool log = config_.enable_logging();
  if (log) {
    std::cout << "Preparing game tree..." << std::endl;
  }
  bool had_root = game_tree_->get_root() != nullptr;
  GameTree::Node* root = get_or_build_root();
  if (log) {
    if (!had_root) {
      std::cout << "Game tree built with " << root->legal_actions.size()
                << " legal actions at root" << std::endl;
    } else {
      std::cout << "Reusing game tree with " << root->legal_actions.size()
                << " legal actions at root" << std::endl;
    }
  }
  
  // Run iterations of CFR
  if (log) {
    std::cout << "Starting CFR iterations..." << std::endl;
  }
  for (int i = 0; i < iterations; ++i) {
    Hand player_a_hand;
    Hand player_b_hand;
    if (player_a_range == nullptr || player_b_range == nullptr) {
      std::vector<Card> deck = BuildDeck();
      std::shuffle(deck.begin(), deck.end(), rng_);
      player_a_hand = DealHand(&deck);
      player_b_hand = DealHand(&deck);
    } else {
      const RangeDeal& deal = range_deals[range_deal_distribution(rng_)];
      player_a_hand = deal.player_a_hand;
      player_b_hand = deal.player_b_hand;
    }
    
    const int max_depth = config_.max_depth();
    if (log) {
      std::cout << "Iteration " << i+1 << "/" << iterations << std::endl;
    }
    int cfr_iteration = iterations_run_;
    std::vector<double> reach_probabilities(2, 1.0);
    double dealt_value = cfr(root, player_a_hand, player_b_hand,
                             reach_probabilities, cfr_iteration, 0, max_depth);

    if (train_swapped) {
      // Train both private-card assignments for the sampled heads-up deal.
      std::vector<double> swapped_reach_probabilities(2, 1.0);
      double swapped_value = cfr(root, player_b_hand, player_a_hand,
                                 swapped_reach_probabilities, cfr_iteration, 0,
                                 max_depth);
      cumulative_root_utility_ += (dealt_value + swapped_value) / 2.0;
    } else {
      cumulative_root_utility_ += dealt_value;
    }
    ++iterations_run_;
  }
  
  if (log) {
    std::cout << "CFR iterations completed" << std::endl;
    std::cout << "Iterations run: " << iterations_run_ << std::endl;
    std::cout << "Information sets: "
              << get_equilibrium_strategy().get_info_sets().size() << std::endl;
    std::cout << "Player A average EV: " << get_expected_value(0) << std::endl;
  }
}

double CFRSolver::cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      int iteration,
                      int depth,
                      int max_depth) {
  // If the node is a terminal node, return the utility
  if (node->is_terminal) {
    if (max_depth > 0) {
      return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
    }
    return utility(node->state, player_a_hand, player_b_hand);
  }

  // Chance card deals are not player decisions, so they do not consume CFR depth.
  if (node->is_chance_node) {
    return chance_sampling_cfr(node, player_a_hand, player_b_hand,
                               reach_probabilities, iteration, depth, max_depth);
  }

  // Check depth limit to prevent infinite recursion
  if (max_depth > 0 && depth >= max_depth) {
    ContinuationContext context = ContinuationContext::ExactHands(
        node->state, player_a_hand, player_b_hand);
    return continuation_value_provider_->value(game_tree_, context);
  }
  
  // Get the player to act at this node
  int player = node->player_to_act;
  
  // Get the player's hand
  const Hand& player_hand = (player == 0) ? player_a_hand : player_b_hand;
  
  // Get the information set key for this player
  std::string info_set_key = info_set_abstraction_->state_to_info_set(node->state, player, player_hand);
  
  // Get the current strategy for this information set
  Strategy::ActionProbabilities strategy =
      get_strategy(info_set_key, node->legal_actions);
  
  // Initialize expected values for each action
  std::unordered_map<int, double> action_values;
  
  // Initialize the expected value for the player
  double node_value = 0.0;
  
  // For each action, recursively call CFR and compute the expected value
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    
    // Create child node for this action if it doesn't exist
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    
    // Get the child node for this action
    GameTree::Node* child_node = node->children[action_id];
    
    // Update reach probabilities for the recursive call
    std::vector<double> new_reach_probabilities = reach_probabilities;
    new_reach_probabilities[player] *= strategy[action_id];
    
    // Recursive call to get the expected value of this action
    double action_value = cfr(child_node, player_a_hand, player_b_hand, new_reach_probabilities, iteration, depth + 1, max_depth);
    
    // Store the action value
    action_values[action_id] = action_value;
    
    // Update the expected value of the node
    node_value += strategy[action_id] * action_value;
  }
  
  // Compute counterfactual regrets if this is not a chance player
  if (player == 0 || player == 1) {
    ++cfr_update_count_;
    ++traversal_stats_.cfr_updates;
    ++traversal_stats_.canonical_state_visits;
    if (visited_canonical_states_.insert(
            CanonicalPublicStateKey(node->state)).second) {
      ++traversal_stats_.unique_canonical_states;
    } else {
      ++traversal_stats_.duplicate_canonical_state_visits;
    }
    traversal_stats_.max_decision_depth =
        std::max(traversal_stats_.max_decision_depth, depth);
    switch (node->state.street()) {
      case Street::PREFLOP:
        ++traversal_stats_.preflop_updates;
        break;
      case Street::FLOP:
        ++traversal_stats_.flop_updates;
        break;
      case Street::TURN:
        ++traversal_stats_.turn_updates;
        break;
      case Street::RIVER:
        ++traversal_stats_.river_updates;
        break;
      default:
        break;
    }

    // Compute the counterfactual reach probability of the opponent
    double opponent_reach_prob = reach_probabilities[1 - player];
    
    // For each action, compute and accumulate the counterfactual regret
    for (const Action& action : node->legal_actions) {
      int action_id = ActionKey(action);
      
      // get_utility returns player A's utility; player B's regret uses the
      // opposite payoff in this zero-sum game.
      double utility_sign = player == 0 ? 1.0 : -1.0;
      double regret =
          opponent_reach_prob * utility_sign * (action_values[action_id] - node_value);
      
      // CFR+ clips cumulative regrets at zero.
      double& cumulative_regret = cumulative_regrets_[info_set_key][action_id];
      cumulative_regret = std::max(0.0, cumulative_regret + regret);
    }
    
    // CFR+ commonly weights later average-strategy samples more heavily.
    update_strategy(info_set_key, strategy,
                    reach_probabilities[player] * (iteration + 1));
  }
  
  return node_value;
}

double CFRSolver::chance_sampling_cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      int iteration,
                      int depth,
                      int max_depth) {
  return SampleChanceValue(
      game_tree_, node, player_a_hand, player_b_hand, ChanceSamples(config_),
      &rng_, [&](GameTree::Node* child_node) {
        return cfr(child_node, player_a_hand, player_b_hand,
                   reach_probabilities, iteration, depth, max_depth);
      });
}

Strategy CFRSolver::get_equilibrium_strategy() const {
  Strategy equilibrium_strategy;
  
  // For each information set
  for (const auto& info_set_pair : cumulative_strategy_) {
    const std::string& info_set_key = info_set_pair.first;
    const auto& cumulative_action_probs = info_set_pair.second;
    
    // Compute the sum of cumulative probabilities
    double sum = 0.0;
    for (const auto& action_prob : cumulative_action_probs) {
      sum += action_prob.second;
    }
    
    // Normalize the probabilities
    Strategy::ActionProbabilities normalized_strategy;
    if (sum > 0.0) {
      for (const auto& action_prob : cumulative_action_probs) {
        normalized_strategy[action_prob.first] = action_prob.second / sum;
      }
    } else {
      // If sum is 0, use uniform strategy
      double uniform_prob = 1.0 / cumulative_action_probs.size();
      for (const auto& action_prob : cumulative_action_probs) {
        normalized_strategy[action_prob.first] = uniform_prob;
      }
    }
    
    // Update the equilibrium strategy
    equilibrium_strategy.update(info_set_key, normalized_strategy);
  }
  
  return equilibrium_strategy;
}

double CFRSolver::evaluate_strategy(const Hand& player_a_hand,
                                    const Hand& player_b_hand) {
  Strategy strategy = get_equilibrium_strategy();
  return evaluate_strategy_node(get_or_build_root(), player_a_hand, player_b_hand,
                                strategy);
}

double CFRSolver::evaluate_strategy(int samples, const HandRange& player_a_range,
                                    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  std::vector<RangeDeal> range_deals = build_compatible_range_deals(
      player_a_range.get_all_weighted_combos(),
      player_b_range.get_all_weighted_combos());
  if (range_deals.empty()) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }

  std::vector<double> range_deal_weights;
  range_deal_weights.reserve(range_deals.size());
  for (const RangeDeal& deal : range_deals) {
    range_deal_weights.push_back(deal.weight);
  }

  Strategy strategy = get_equilibrium_strategy();
  if (samples < kParallelEvaluationSampleThreshold) {
    return evaluate_strategy_samples(samples, range_deals, range_deal_weights,
                                     strategy);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return evaluate_strategy_samples(samples, range_deals, range_deal_weights,
                                     strategy);
  }

  ThreadPoolExecutor executor(worker_count);
  std::uniform_int_distribution<unsigned int> seed_distribution;
  std::vector<unsigned int> worker_seeds;
  worker_seeds.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    worker_seeds.push_back(seed_distribution(rng_));
  }

  PokerConfig config = config_;
  std::shared_ptr<TerminalUtilityCache> utility_cache = utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
  std::vector<std::future<double>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, &range_deals,
                                       &range_deal_weights, &strategy,
                                       utility_cache, continuation_value_provider,
                                       shard_samples, seed]() {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.rng_.seed(seed);
      return worker.evaluate_strategy_samples(
                 shard_samples, range_deals, range_deal_weights, strategy) *
             shard_samples;
    }));
  }

  double total = 0.0;
  for (std::future<double>& future : futures) {
    total += future.get();
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_samples(
    int samples,
    const std::vector<RangeDeal>& range_deals,
    const std::vector<double>& range_deal_weights,
    const Strategy& strategy) {
  if (samples <= 0) {
    return 0.0;
  }

  std::discrete_distribution<size_t> range_deal_distribution(
      range_deal_weights.begin(), range_deal_weights.end());
  GameTree::Node* root = get_or_build_root();

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    const RangeDeal& deal = range_deals[range_deal_distribution(rng_)];
    total += evaluate_strategy_node(root, deal.player_a_hand, deal.player_b_hand,
                                    strategy);
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(GameTree::Node* node,
                                         const Hand& player_a_hand,
                                         const Hand& player_b_hand,
                                         const Strategy& strategy) {
  if (node->is_terminal) {
    return utility(node->state, player_a_hand, player_b_hand);
  }
  if (node->is_chance_node) {
    return SampleChanceValue(
        game_tree_, node, player_a_hand, player_b_hand, ChanceSamples(config_),
        &rng_, [&](GameTree::Node* child_node) {
          return evaluate_strategy_node(child_node, player_a_hand, player_b_hand,
                                        strategy);
        });
  }
  if (node->legal_actions.empty()) {
    return 0.0;
  }

  int player = node->player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const Hand& player_hand = player == 0 ? player_a_hand : player_b_hand;
  std::string info_set_key =
      info_set_abstraction_->state_to_info_set(node->state, player, player_hand);

  std::vector<int> action_ids;
  action_ids.reserve(node->legal_actions.size());
  double probability_sum = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    action_ids.push_back(action_id);
    probability_sum += strategy.get_action_probability(info_set_key, action_id);
  }

  double value = 0.0;
  for (size_t i = 0; i < node->legal_actions.size(); ++i) {
    const Action& action = node->legal_actions[i];
    int action_id = action_ids[i];
    double probability = probability_sum > 0.0
                             ? strategy.get_action_probability(info_set_key, action_id) /
                                   probability_sum
                             : 1.0 / node->legal_actions.size();
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    value += probability * evaluate_strategy_node(
                              node->children[action_id], player_a_hand,
                              player_b_hand, strategy);
  }
  return value;
}

double CFRSolver::best_response_value(GameTree::Node* node,
                                      const Hand& player_a_hand,
                                      const Hand& player_b_hand,
                                      const Strategy& strategy,
                                      int best_response_player) {
  if (node->is_terminal) {
    double player_a_value =
        utility(node->state, player_a_hand, player_b_hand);
    return best_response_player == 0 ? player_a_value : -player_a_value;
  }
  if (node->is_chance_node) {
    return SampleChanceValue(
        game_tree_, node, player_a_hand, player_b_hand, ChanceSamples(config_),
        &rng_, [&](GameTree::Node* child_node) {
          return best_response_value(child_node, player_a_hand, player_b_hand,
                                     strategy, best_response_player);
        });
  }
  if (node->legal_actions.empty()) {
    return 0.0;
  }

  int player = node->player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const Hand& player_hand = player == 0 ? player_a_hand : player_b_hand;
  std::string info_set_key =
      info_set_abstraction_->state_to_info_set(node->state, player, player_hand);

  std::vector<int> action_ids;
  action_ids.reserve(node->legal_actions.size());
  double probability_sum = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    action_ids.push_back(action_id);
    probability_sum += strategy.get_action_probability(info_set_key, action_id);
  }

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < node->legal_actions.size(); ++i) {
      const Action& action = node->legal_actions[i];
      int action_id = action_ids[i];
      if (node->children.find(action_id) == node->children.end()) {
        node->children[action_id] = game_tree_->create_child_node(node, action);
      }
      value = std::max(value, best_response_value(
                                  node->children[action_id], player_a_hand,
                                  player_b_hand, strategy, best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (size_t i = 0; i < node->legal_actions.size(); ++i) {
    const Action& action = node->legal_actions[i];
    int action_id = action_ids[i];
    double probability = probability_sum > 0.0
                             ? strategy.get_action_probability(info_set_key, action_id) /
                                   probability_sum
                             : 1.0 / node->legal_actions.size();
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    value += probability * best_response_value(
                               node->children[action_id], player_a_hand,
                               player_b_hand, strategy, best_response_player);
  }
  return value;
}

double CFRSolver::best_response_value_against_range(
    GameTree::Node* node,
    const Hand& best_response_hand,
    const std::vector<std::pair<Hand, double>>& opponent_hands,
    const Strategy& strategy,
    int best_response_player) {
  double total_weight = TotalWeight(opponent_hands);
  if (total_weight <= 0.0) {
    return 0.0;
  }

  if (node->is_terminal) {
    double value = 0.0;
    for (const auto& opponent_hand : opponent_hands) {
      const Hand& player_a_hand =
          best_response_player == 0 ? best_response_hand : opponent_hand.first;
      const Hand& player_b_hand =
          best_response_player == 0 ? opponent_hand.first : best_response_hand;
      double player_a_value =
          utility(node->state, player_a_hand, player_b_hand);
      value += opponent_hand.second *
               (best_response_player == 0 ? player_a_value : -player_a_value);
    }
    return value / total_weight;
  }

  if (node->is_chance_node) {
    double value = 0.0;
    int samples = ChanceSamples(config_);
    WeightedHands weighted_opponents;
    weighted_opponents.reset(opponent_hands);
    for (int i = 0; i < samples; ++i) {
      Hand sampled_opponent = SampleWeightedHand(&weighted_opponents, &rng_);
      const Hand& player_a_hand =
          best_response_player == 0 ? best_response_hand : sampled_opponent;
      const Hand& player_b_hand =
          best_response_player == 0 ? sampled_opponent : best_response_hand;
      std::vector<Card> cards =
          SampleStreetCards(node->state, player_a_hand, player_b_hand, &rng_);
      GameTree::Node* child_node =
          game_tree_->create_chance_child_node(node, cards);
      std::vector<std::pair<Hand, double>> child_opponents = {
          {sampled_opponent, 1.0}};
      value += best_response_value_against_range(
          child_node, best_response_hand, child_opponents, strategy,
          best_response_player);
      delete child_node;
    }
    return value / samples;
  }

  if (node->legal_actions.empty()) {
    return 0.0;
  }

  int player = node->player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (const Action& action : node->legal_actions) {
      int action_id = ActionKey(action);
      if (node->children.find(action_id) == node->children.end()) {
        node->children[action_id] = game_tree_->create_child_node(node, action);
      }
      value = std::max(value, best_response_value_against_range(
                                  node->children[action_id], best_response_hand,
                                  opponent_hands, strategy,
                                  best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }

    std::vector<std::pair<Hand, double>> child_opponents;
    for (const auto& opponent_hand : opponent_hands) {
      std::string info_set_key = info_set_abstraction_->state_to_info_set(
          node->state, player, opponent_hand.first);
      double probability = StrategyActionProbability(
          strategy, info_set_key, node->legal_actions, action_id);
      if (probability > 0.0) {
        child_opponents.emplace_back(opponent_hand.first,
                                     opponent_hand.second * probability);
      }
    }

    double child_weight = TotalWeight(child_opponents);
    if (child_weight > 0.0) {
      value += (child_weight / total_weight) *
               best_response_value_against_range(
                   node->children[action_id], best_response_hand,
                   child_opponents, strategy, best_response_player);
    }
  }
  return value;
}

double CFRSolver::sampled_range_best_response_value(
    int samples,
    const HandRange& best_response_range,
    const HandRange& opponent_range,
    const Strategy& strategy,
    int best_response_player) {
  if (samples <= 0) {
    return 0.0;
  }

  std::vector<std::pair<Hand, double>> best_response_hands =
      best_response_range.get_all_weighted_combos();
  std::vector<std::pair<Hand, double>> opponent_hands =
      opponent_range.get_all_weighted_combos();
  if (best_response_hands.empty() || opponent_hands.empty()) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }

  if (samples < kParallelBestResponseSampleThreshold) {
    return sampled_range_best_response_samples(
        samples, best_response_hands, opponent_hands, strategy,
        best_response_player);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return sampled_range_best_response_samples(
        samples, best_response_hands, opponent_hands, strategy,
        best_response_player);
  }

  ThreadPoolExecutor executor(worker_count);
  std::uniform_int_distribution<unsigned int> seed_distribution;
  std::vector<unsigned int> worker_seeds;
  worker_seeds.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    worker_seeds.push_back(seed_distribution(rng_));
  }

  PokerConfig config = config_;
  std::shared_ptr<TerminalUtilityCache> utility_cache = utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
  std::vector<std::future<double>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, &best_response_hands,
                                       &opponent_hands, &strategy,
                                       utility_cache, continuation_value_provider,
                                       shard_samples, seed, best_response_player]() {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.rng_.seed(seed);
      return worker.sampled_range_best_response_samples(
                 shard_samples, best_response_hands, opponent_hands, strategy,
                 best_response_player) *
             shard_samples;
    }));
  }

  double total = 0.0;
  for (std::future<double>& future : futures) {
    total += future.get();
  }
  return total / samples;
}

double CFRSolver::sampled_range_best_response_samples(
    int samples,
    const std::vector<std::pair<Hand, double>>& best_response_hands,
    const std::vector<std::pair<Hand, double>>& opponent_hands,
    const Strategy& strategy,
    int best_response_player) {
  if (samples <= 0) {
    return 0.0;
  }

  WeightedHands weighted_best_response_hands;
  weighted_best_response_hands.reset(best_response_hands);
  GameTree::Node* root = get_or_build_root();
  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    Hand best_response_hand =
        SampleWeightedHand(&weighted_best_response_hands, &rng_);
    std::vector<std::pair<Hand, double>> compatible_opponents =
        CompatibleHands(opponent_hands, best_response_hand, root->state);
    if (compatible_opponents.empty()) {
      throw std::invalid_argument(
          "Could not sample non-overlapping hands from ranges");
    }
    total += best_response_value_against_range(root, best_response_hand,
                                               compatible_opponents, strategy,
                                               best_response_player);
  }
  return total / samples;
}

double CFRSolver::calculate_exploitability() {
  return calculate_exploitability(1);
}

double CFRSolver::calculate_exploitability(int samples) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<Card> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), rng_);
    Hand player_a_hand = DealHand(&deck);
    Hand player_b_hand = DealHand(&deck);
    total += calculate_exploitability(player_a_hand, player_b_hand);
  }
  return total / samples;
}

double CFRSolver::calculate_exploitability(int samples,
                                           const HandRange& player_a_range,
                                           const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  Strategy strategy = get_equilibrium_strategy();
  double strategy_player_a_value =
      evaluate_strategy(samples, player_a_range, player_b_range);
  double player_a_gap =
      sampled_range_best_response_value(samples, player_a_range, player_b_range,
                                        strategy, 0) -
      strategy_player_a_value;
  double player_b_gap =
      sampled_range_best_response_value(samples, player_b_range, player_a_range,
                                        strategy, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

double CFRSolver::calculate_player_a_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  Strategy strategy = get_equilibrium_strategy();
  return sampled_range_best_response_value(samples, player_a_range,
                                           player_b_range, strategy, 0);
}

double CFRSolver::calculate_player_b_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  Strategy strategy = get_equilibrium_strategy();
  return sampled_range_best_response_value(samples, player_b_range,
                                           player_a_range, strategy, 1);
}

double CFRSolver::calculate_exploitability(const Hand& player_a_hand,
                                           const Hand& player_b_hand) {
  Strategy strategy = get_equilibrium_strategy();
  GameTree::Node* root = get_or_build_root();
  double strategy_player_a_value =
      evaluate_strategy_node(root, player_a_hand, player_b_hand, strategy);
  double player_a_gap =
      best_response_value(root, player_a_hand, player_b_hand, strategy, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_hand, player_b_hand, strategy, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

Action CFRSolver::get_best_response_action(GameTree::Node* node,
                                           const Hand& player_a_hand,
                                           const Hand& player_b_hand,
                                           int best_response_player) {
  Action no_action;
  no_action.set_action(ActionType::NO_ACTION);
  if (node == nullptr || node->is_terminal || node->is_chance_node ||
      node->legal_actions.empty() || node->player_to_act != best_response_player) {
    return no_action;
  }

  Strategy strategy = get_equilibrium_strategy();
  double best_value = -std::numeric_limits<double>::infinity();
  Action best_action = no_action;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    double value = best_response_value(node->children[action_id], player_a_hand,
                                       player_b_hand, strategy,
                                       best_response_player);
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  return best_action;
}

void CFRSolver::save_strategy(const std::string& filename) const {
  std::ofstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  Strategy equilibrium_strategy = get_equilibrium_strategy();
  StrategySnapshot snapshot;
  *snapshot.mutable_config() = config_;
  snapshot.set_iterations_run(iterations_run_);
  snapshot.set_abstraction_version(kInfoSetAbstractionVersion);

  for (const std::string& info_set_key : equilibrium_strategy.get_info_sets()) {
    StrategyInfoSetSnapshot* info_set = snapshot.add_info_sets();
    info_set->set_info_set_key(info_set_key);

    Strategy::ActionProbabilities action_probs =
        equilibrium_strategy.get_strategy(info_set_key);
    for (const auto& action_prob : action_probs) {
      StrategyActionSnapshot* action = info_set->add_actions();
      *action->mutable_action() = ActionFromKey(action_prob.first);
      action->set_probability(action_prob.second);
    }
  }

  if (!snapshot.SerializeToOstream(&file)) {
    std::cerr << "Failed to write strategy snapshot: " << filename << std::endl;
  }
}

void CFRSolver::load_strategy(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filename << std::endl;
    return;
  }

  StrategySnapshot snapshot;
  if (!snapshot.ParseFromIstream(&file)) {
    std::cerr << "Failed to parse strategy snapshot: " << filename << std::endl;
    return;
  }

  current_strategy_.clear();
  cumulative_strategy_.clear();

  for (const StrategyInfoSetSnapshot& info_set : snapshot.info_sets()) {
    Strategy::ActionProbabilities action_probs;
    for (const StrategyActionSnapshot& action : info_set.actions()) {
      if (action.has_action()) {
        action_probs[ActionKey(action.action())] = action.probability();
      }
    }
    current_strategy_.update(info_set.info_set_key(), action_probs);
    cumulative_strategy_[info_set.info_set_key()] = action_probs;
  }
}

double CFRSolver::get_expected_value(int player_id) const {
  if (iterations_run_ == 0) {
    return 0.0;
  }
  double player_a_ev = cumulative_root_utility_ / iterations_run_;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

CFRSolver::UtilityCacheStats CFRSolver::get_utility_cache_stats() const {
  TerminalUtilityCache::Stats stats = utility_cache_->stats();
  return {stats.hits, stats.misses, stats.entries};
}

double CFRSolver::utility(const BoardState& state,
                          const Hand& player_a_hand,
                          const Hand& player_b_hand) {
  if (state.folded_player() >= 0 ||
      player_a_hand.cards_size() + state.cards_size() < 5) {
    return game_tree_->get_utility(state, player_a_hand, player_b_hand);
  }

  return utility_cache_->get_or_compute(
      state, player_a_hand, player_b_hand, [&]() {
        return game_tree_->get_utility(state, player_a_hand, player_b_hand);
      });
}

Strategy::ActionProbabilities CFRSolver::get_strategy(
    const std::string& info_set_key,
    const std::vector<Action>& legal_actions) {
  Strategy::ActionProbabilities strategy;
  auto& regrets = cumulative_regrets_[info_set_key];
  if (legal_actions.empty()) {
    return strategy;
  }

  std::vector<int> action_ids;
  action_ids.reserve(legal_actions.size());
  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    if (regrets.find(action_id) == regrets.end()) {
      regrets[action_id] = 0.0;
    }
    if (std::find(action_ids.begin(), action_ids.end(), action_id) ==
        action_ids.end()) {
      action_ids.push_back(action_id);
    }
  }
  
  // Compute the sum of positive regrets
  double sum_positive_regrets = 0.0;
  for (int action_id : action_ids) {
    sum_positive_regrets += std::max(0.0, regrets[action_id]);
  }
  
  // If there are positive regrets, use regret matching
  if (sum_positive_regrets > 0.0) {
    for (int action_id : action_ids) {
      strategy[action_id] = std::max(0.0, regrets[action_id]) / sum_positive_regrets;
    }
  } else {
    // If all regrets are negative or zero, use a uniform strategy
    double uniform_prob = 1.0 / action_ids.size();
    for (int action_id : action_ids) {
      strategy[action_id] = uniform_prob;
    }
  }
  
  return strategy;
}

void CFRSolver::update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob) {
  // Accumulate the strategy weighted by the reach probability
  for (const auto& action_prob : strategy) {
    cumulative_strategy_[info_set_key][action_prob.first] += reach_prob * action_prob.second;
  }
}

} // namespace poker
