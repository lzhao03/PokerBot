#include "src/cfr_solver.h"
#include "src/card_utils.h"
#include "src/continuation_value.h"
#include "src/hand_range.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include <algorithm>
#include <cstdint>
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

double TotalWeight(const WeightedHandRange& hands) {
  return std::accumulate(hands.weights.begin(), hands.weights.end(), 0.0);
}

WeightedHandRange CompatibleHands(
    const WeightedHandRange& hands,
    const Hand& known_hand,
    const BoardState& state) {
  WeightedHandRange compatible_hands;
  compatible_hands.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    if (!HandsOverlap(hands.hands[i], known_hand) &&
        !HandOverlapsBoard(hands.hands[i], state)) {
      compatible_hands.add(hands.hands[i], hands.weights[i]);
    }
  }
  return compatible_hands;
}

WeightedHandRange PublicCompatibleRange(
    const WeightedHandRange& hands,
    const BoardState& state) {
  WeightedHandRange compatible_hands;
  compatible_hands.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    if (hands.weights[i] > 0.0 && !HandOverlapsBoard(hands.hands[i], state)) {
      compatible_hands.add(hands.hands[i], hands.weights[i]);
    }
  }
  return compatible_hands;
}

int EncodedCard(const Card& card) {
  return card.rank() * 8 + static_cast<int>(card.suit());
}

int ChanceCardsKey(const std::vector<Card>& cards) {
  int key = static_cast<int>(cards.size());
  for (const Card& card : cards) {
    key = key * 128 + EncodedCard(card);
  }
  return -1 - key;
}

GameTree::Node* CachedChanceChild(GameTree* game_tree,
                                  GameTree::Node* node,
                                  const std::vector<Card>& cards,
                                  int64_t* created_nodes) {
  const int child_key = ChanceCardsKey(cards);
  auto child = node->children.find(child_key);
  if (child != node->children.end()) {
    return child->second;
  }

  GameTree::Node* child_node = game_tree->create_chance_child_node(node, cards);
  node->children[child_key] = child_node;
  if (created_nodes != nullptr) {
    ++(*created_nodes);
  }
  return child_node;
}

GameTree::Node* CachedActionChild(GameTree* game_tree,
                                  GameTree::Node* node,
                                  const Action& action,
                                  int action_id,
                                  int64_t* created_nodes) {
  auto child = node->children.find(action_id);
  if (child != node->children.end()) {
    return child->second;
  }

  GameTree::Node* child_node = game_tree->create_child_node(node, action);
  node->children.emplace(action_id, child_node);
  if (created_nodes != nullptr) {
    ++(*created_nodes);
  }
  return child_node;
}

int RoundedContribution(const BoardState& state, int player) {
  return state.player_contribution_size() > player
             ? static_cast<int>(std::lround(state.player_contribution(player)))
             : 0;
}

CanonicalPublicStateKey MakeCanonicalPublicStateKey(const BoardState& state) {
  CanonicalPublicStateKey key;
  key.street = static_cast<int>(state.street());
  key.pot = state.pot();
  key.stack_a = state.stack_a();
  key.stack_b = state.stack_b();
  key.all_in = state.all_in() ? 1 : 0;
  key.folded_player = state.folded_player();
  key.player_to_act = state.player_to_act();
  key.player_contributions[0] = RoundedContribution(state, 0);
  key.player_contributions[1] = RoundedContribution(state, 1);
  key.board_size = state.cards_size();
  for (int i = 0; i < state.cards_size(); ++i) {
    key.board_cards[i] = EncodedCard(state.cards(i));
  }

  const int history_size = state.history().actions_size();
  if (history_size == 0) {
    return key;
  }

  key.history_bucket = history_size > 1 ? 2 : 1;
  const Action& last = state.history().actions(history_size - 1);
  key.last_player = last.player();
  key.last_action = static_cast<int>(last.action());
  key.last_amount = static_cast<int>(std::lround(last.amount()));
  return key;
}

void HashCombine(size_t* seed, int value) {
  *seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
}

template <size_t N>
void HashArray(size_t* seed, const std::array<int, N>& values) {
  for (int value : values) {
    HashCombine(seed, value);
  }
}

void AppendInt(std::string* out, int value) {
  out->append(std::to_string(value));
}

void AppendEncodedCard(std::string* out, int encoded_card) {
  AppendInt(out, encoded_card / 8);
  out->push_back(':');
  AppendInt(out, encoded_card % 8);
}

struct WeightedHands {
  WeightedHandRange hands;
  std::discrete_distribution<size_t> distribution;

  void reset(WeightedHandRange new_hands) {
    hands = std::move(new_hands);
    distribution = std::discrete_distribution<size_t>(
        hands.weights.begin(), hands.weights.end());
  }
};

Hand SampleWeightedHand(WeightedHands* hands, std::mt19937* rng) {
  return hands->hands.hands[hands->distribution(*rng)];
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
                         int64_t* created_nodes,
                         EvaluateChild evaluate_child) {
  double value = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<Card> cards =
        SampleStreetCards(node->state, player_a_hand, player_b_hand, rng);
    GameTree::Node* child_node =
        CachedChanceChild(game_tree, node, cards, created_nodes);
    value += evaluate_child(child_node);
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

bool CanonicalPublicStateKey::operator==(
    const CanonicalPublicStateKey& other) const {
  return street == other.street && pot == other.pot &&
         stack_a == other.stack_a && stack_b == other.stack_b &&
         all_in == other.all_in && folded_player == other.folded_player &&
         player_to_act == other.player_to_act &&
         player_contributions == other.player_contributions &&
         board_size == other.board_size && board_cards == other.board_cards &&
         history_bucket == other.history_bucket &&
         last_player == other.last_player && last_action == other.last_action &&
         last_amount == other.last_amount;
}

size_t CanonicalPublicStateKeyHash::operator()(
    const CanonicalPublicStateKey& key) const {
  size_t seed = 0;
  HashCombine(&seed, key.street);
  HashCombine(&seed, key.pot);
  HashCombine(&seed, key.stack_a);
  HashCombine(&seed, key.stack_b);
  HashCombine(&seed, key.all_in);
  HashCombine(&seed, key.folded_player);
  HashCombine(&seed, key.player_to_act);
  HashArray(&seed, key.player_contributions);
  HashCombine(&seed, key.board_size);
  HashArray(&seed, key.board_cards);
  HashCombine(&seed, key.history_bucket);
  HashCombine(&seed, key.last_player);
  HashCombine(&seed, key.last_action);
  HashCombine(&seed, key.last_amount);
  return seed;
}

bool CFRSolver::InfoSetKey::operator==(const InfoSetKey& other) const {
  return player == other.player && street == other.street &&
         pot == other.pot && stack_a == other.stack_a &&
         stack_b == other.stack_b && all_in == other.all_in &&
         folded_player == other.folded_player &&
         player_to_act == other.player_to_act &&
         player_contribution_size == other.player_contribution_size &&
         player_contributions == other.player_contributions &&
         hand_size == other.hand_size && hand_cards == other.hand_cards &&
         board_size == other.board_size && board_cards == other.board_cards &&
         history_size == other.history_size &&
         history_values == other.history_values &&
         history_overflow == other.history_overflow;
}

size_t CFRSolver::InfoSetKeyHash::operator()(const InfoSetKey& key) const {
  size_t seed = 0;
  HashCombine(&seed, key.player);
  HashCombine(&seed, key.street);
  HashCombine(&seed, key.pot);
  HashCombine(&seed, key.stack_a);
  HashCombine(&seed, key.stack_b);
  HashCombine(&seed, key.all_in);
  HashCombine(&seed, key.folded_player);
  HashCombine(&seed, key.player_to_act);
  HashCombine(&seed, key.player_contribution_size);
  HashArray(&seed, key.player_contributions);
  HashCombine(&seed, key.hand_size);
  HashArray(&seed, key.hand_cards);
  HashCombine(&seed, key.board_size);
  HashArray(&seed, key.board_cards);
  HashCombine(&seed, key.history_size);
  HashArray(&seed, key.history_values);
  for (int value : key.history_overflow) {
    HashCombine(&seed, value);
  }
  return seed;
}

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
    const WeightedHandRange& player_a_hands,
    const WeightedHandRange& player_b_hands) {
  std::vector<RangeDeal> deals;
  for (size_t a = 0; a < player_a_hands.size(); ++a) {
    if (player_a_hands.weights[a] <= 0.0) {
      continue;
    }
    for (size_t b = 0; b < player_b_hands.size(); ++b) {
      if (player_b_hands.weights[b] <= 0.0 ||
          HandsOverlap(player_a_hands.hands[a], player_b_hands.hands[b])) {
        continue;
      }
      deals.push_back({player_a_hands.hands[a], player_b_hands.hands[b],
                       player_a_hands.weights[a] * player_b_hands.weights[b]});
    }
  }
  return deals;
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const BoardState& state,
    int player,
    const Hand& hand) const {
  InfoSetKey key;
  key.player = player;
  key.street = static_cast<int>(state.street());
  key.pot = state.pot();
  key.stack_a = state.stack_a();
  key.stack_b = state.stack_b();
  key.all_in = state.all_in() ? 1 : 0;
  key.folded_player = state.folded_player();
  key.player_to_act = state.player_to_act();
  key.player_contribution_size =
      std::min(state.player_contribution_size(), 2);
  for (int i = 0; i < key.player_contribution_size; ++i) {
    key.player_contributions[i] = RoundedContribution(state, i);
  }

  key.hand_size = std::min(hand.cards_size(), 2);
  for (int i = 0; i < key.hand_size; ++i) {
    key.hand_cards[i] = EncodedCard(hand.cards(i));
  }

  key.board_size = std::min(state.cards_size(), InfoSetKey::kMaxCards);
  for (int i = 0; i < key.board_size; ++i) {
    key.board_cards[i] = EncodedCard(state.cards(i));
  }

  const int history_value_count = state.history().actions_size() * 3;
  if (history_value_count > InfoSetKey::kInlineHistoryValues) {
    key.history_overflow.reserve(history_value_count -
                                 InfoSetKey::kInlineHistoryValues);
  }
  auto add_history_value = [&key](int value) {
    if (key.history_size < InfoSetKey::kInlineHistoryValues) {
      key.history_values[key.history_size] = value;
    } else {
      key.history_overflow.push_back(value);
    }
    ++key.history_size;
  };
  for (const Action& action : state.history().actions()) {
    add_history_value(action.player());
    add_history_value(static_cast<int>(action.action()));
    add_history_value(static_cast<int>(std::lround(action.amount())));
  }

  return key;
}

void CFRSolver::ensure_info_set_actions(
    InfoSetData* info_set,
    const std::vector<Action>& legal_actions) {
  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    if (std::find(info_set->action_ids.begin(), info_set->action_ids.end(),
                  action_id) != info_set->action_ids.end()) {
      continue;
    }
    info_set->action_ids.push_back(action_id);
    info_set->cumulative_regrets.push_back(0.0);
    info_set->cumulative_strategy.push_back(0.0);
  }
}

int CFRSolver::get_or_create_info_set_id(
    const InfoSetKey& key,
    const std::vector<Action>& legal_actions) {
  auto existing = info_set_ids_.find(key);
  if (existing != info_set_ids_.end()) {
    ensure_info_set_actions(&info_sets_[existing->second], legal_actions);
    return existing->second;
  }

  const int id = static_cast<int>(info_sets_.size());
  InfoSetData data;
  data.key = key;
  ensure_info_set_actions(&data, legal_actions);
  info_sets_.push_back(std::move(data));
  info_set_ids_.emplace(info_sets_.back().key, id);
  return id;
}

const CFRSolver::InfoSetData* CFRSolver::find_info_set(
    const InfoSetKey& key) const {
  auto existing = info_set_ids_.find(key);
  if (existing == info_set_ids_.end()) {
    return nullptr;
  }
  return &info_sets_[existing->second];
}

int CFRSolver::get_or_create_legacy_info_set_id(
    const std::string& info_set_key) {
  auto existing = legacy_info_set_ids_.find(info_set_key);
  if (existing != legacy_info_set_ids_.end()) {
    return existing->second;
  }

  const int id = static_cast<int>(legacy_info_sets_.size());
  LegacyInfoSetData data;
  data.key = info_set_key;
  legacy_info_sets_.push_back(std::move(data));
  legacy_info_set_ids_.emplace(legacy_info_sets_.back().key, id);
  return id;
}

const CFRSolver::LegacyInfoSetData* CFRSolver::find_legacy_info_set(
    const std::string& info_set_key) const {
  auto existing = legacy_info_set_ids_.find(info_set_key);
  if (existing == legacy_info_set_ids_.end()) {
    return nullptr;
  }
  return &legacy_info_sets_[existing->second];
}

size_t CFRSolver::ensure_legacy_info_set_action(
    LegacyInfoSetData* info_set,
    int action_id) {
  auto existing = std::find(info_set->action_ids.begin(),
                            info_set->action_ids.end(), action_id);
  if (existing != info_set->action_ids.end()) {
    return static_cast<size_t>(existing - info_set->action_ids.begin());
  }

  info_set->action_ids.push_back(action_id);
  info_set->cumulative_regrets.push_back(0.0);
  info_set->cumulative_strategy.push_back(0.0);
  return info_set->action_ids.size() - 1;
}

void CFRSolver::ensure_legacy_info_set_actions(
    LegacyInfoSetData* info_set,
    const std::vector<Action>& legal_actions) {
  for (const Action& action : legal_actions) {
    ensure_legacy_info_set_action(info_set, ActionKey(action));
  }
}

std::string CFRSolver::info_set_key_to_string(const InfoSetKey& key) const {
  std::string text;
  text.reserve(128 + key.history_size * 4);
  text.push_back('P');
  AppendInt(&text, key.player);
  text.append(":H[");
  for (int i = 0; i < key.hand_size; ++i) {
    if (i > 0) {
      text.push_back(',');
    }
    AppendEncodedCard(&text, key.hand_cards[i]);
  }
  text.append("]:B[");
  for (int i = 0; i < key.board_size; ++i) {
    if (i > 0) {
      text.push_back(',');
    }
    AppendEncodedCard(&text, key.board_cards[i]);
  }
  text.append("]:S");
  AppendInt(&text, key.street);
  text.append(":POT");
  AppendInt(&text, key.pot);
  text.append(":ST");
  AppendInt(&text, key.stack_a);
  text.push_back(',');
  AppendInt(&text, key.stack_b);
  text.append(":AI");
  AppendInt(&text, key.all_in);
  text.append(":F");
  AppendInt(&text, key.folded_player);
  text.append(":T");
  AppendInt(&text, key.player_to_act);
  text.append(":C[");
  for (int i = 0; i < key.player_contribution_size; ++i) {
    if (i > 0) {
      text.push_back(',');
    }
    AppendInt(&text, key.player_contributions[i]);
  }
  text.append("]:A[");
  for (int i = 0; i < key.history_size; i += 3) {
    if (i > 0) {
      text.push_back(',');
    }
    auto history_value = [&](int index) {
      if (index < InfoSetKey::kInlineHistoryValues) {
        return key.history_values[index];
      }
      return key.history_overflow[index - InfoSetKey::kInlineHistoryValues];
    };
    AppendInt(&text, history_value(i));
    text.push_back(':');
    AppendInt(&text, history_value(i + 1));
    text.push_back(':');
    AppendInt(&text, history_value(i + 2));
  }
  text.push_back(']');
  return text;
}

double CFRSolver::regret_for_info_set(const std::string& info_set_key,
                                      int action_id) const {
  const LegacyInfoSetData* legacy_info_set =
      find_legacy_info_set(info_set_key);
  if (legacy_info_set != nullptr) {
    auto action = std::find(legacy_info_set->action_ids.begin(),
                            legacy_info_set->action_ids.end(), action_id);
    if (action == legacy_info_set->action_ids.end()) {
      return 0.0;
    }
    const size_t index =
        static_cast<size_t>(action - legacy_info_set->action_ids.begin());
    return legacy_info_set->cumulative_regrets[index];
  }

  for (const InfoSetData& info_set : info_sets_) {
    if (info_set_key_to_string(info_set.key) != info_set_key) {
      continue;
    }
    auto action = std::find(info_set.action_ids.begin(),
                            info_set.action_ids.end(), action_id);
    if (action == info_set.action_ids.end()) {
      return 0.0;
    }
    const size_t index =
        static_cast<size_t>(action - info_set.action_ids.begin());
    return info_set.cumulative_regrets[index];
  }
  return 0.0;
}

void CFRSolver::set_legacy_regret_for_info_set(
    const std::string& info_set_key,
    int action_id,
    double regret) {
  const int info_set_id = get_or_create_legacy_info_set_id(info_set_key);
  LegacyInfoSetData& info_set = legacy_info_sets_[info_set_id];
  const size_t action_index =
      ensure_legacy_info_set_action(&info_set, action_id);
  info_set.cumulative_regrets[action_index] = regret;
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
  WeightedHandRange player_a_hands;
  WeightedHandRange player_b_hands;
  std::vector<RangeDeal> range_deals;
  std::vector<double> range_deal_weights;
  std::discrete_distribution<size_t> range_deal_distribution;
  if (iterations > 0 && player_a_range != nullptr &&
      player_b_range != nullptr) {
    player_a_hands = player_a_range->get_all_weighted_combos();
    player_b_hands = player_b_range->get_all_weighted_combos();
    range_deals =
        build_compatible_range_deals(player_a_hands, player_b_hands);
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
    const WeightedHandRange* player_a_context_range =
        max_depth > 0 && player_a_range != nullptr ? &player_a_hands : nullptr;
    const WeightedHandRange* player_b_context_range =
        max_depth > 0 && player_b_range != nullptr ? &player_b_hands : nullptr;
    double dealt_value = cfr_with_ranges(
        root, player_a_hand, player_b_hand, reach_probabilities,
        cfr_iteration, 0, max_depth, player_a_context_range,
        player_b_context_range);

    if (train_swapped) {
      // Train both private-card assignments for the sampled heads-up deal.
      std::vector<double> swapped_reach_probabilities(2, 1.0);
      double swapped_value = cfr_with_ranges(
          root, player_b_hand, player_a_hand, swapped_reach_probabilities,
          cfr_iteration, 0, max_depth, nullptr, nullptr);
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
  return cfr_with_ranges(node, player_a_hand, player_b_hand,
                         reach_probabilities, iteration, depth, max_depth,
                         nullptr, nullptr);
}

double CFRSolver::cfr_with_ranges(
    GameTree::Node* node,
    const Hand& player_a_hand,
    const Hand& player_b_hand,
    std::vector<double>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    const WeightedHandRange* player_a_range,
    const WeightedHandRange* player_b_range) {
  // If the node is a terminal node, return the utility
  if (node->is_terminal) {
    ++traversal_stats_.terminal_utility_calls;
    if (node->state.folded_player() >= 0) {
      ++traversal_stats_.fold_utility_calls;
    } else {
      ++traversal_stats_.showdown_utility_calls;
    }
    if (max_depth > 0) {
      return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
    }
    return utility(node->state, player_a_hand, player_b_hand);
  }

  // Chance card deals are not player decisions, so they do not consume CFR depth.
  if (node->is_chance_node) {
    return chance_sampling_cfr(node, player_a_hand, player_b_hand,
                               reach_probabilities, iteration, depth, max_depth,
                               player_a_range, player_b_range);
  }

  // Check depth limit to prevent infinite recursion
  if (max_depth > 0 && depth >= max_depth) {
    ContinuationContext context = build_continuation_context(
        node->state, player_a_hand, player_b_hand, player_a_range,
        player_b_range);
    return continuation_value_provider_->value(game_tree_, context);
  }
  
  // Get the player to act at this node
  int player = node->player_to_act;
  
  // Get the player's hand
  const Hand& player_hand = (player == 0) ? player_a_hand : player_b_hand;
  
  InfoSetKey info_set_key = make_info_set_key(node->state, player, player_hand);
  const int info_set_id =
      get_or_create_info_set_id(info_set_key, node->legal_actions);
  std::vector<ActionChoice> action_choices =
      get_action_choices(info_set_id, node->legal_actions);
  
  // Initialize the expected value for the player
  double node_value = 0.0;
  WeightedHandRange conditioned_player_range;
  std::vector<int> legal_action_ids;
  legal_action_ids.reserve(action_choices.size());
  for (const ActionChoice& choice : action_choices) {
    legal_action_ids.push_back(choice.action_id);
  }
  // For each action, recursively call CFR and compute the expected value
  for (ActionChoice& choice : action_choices) {
    const Action& action = *choice.action;
    int action_id = choice.action_id;
    
    GameTree::Node* child_node = CachedActionChild(
        game_tree_, node, action, action_id,
        &traversal_stats_.child_nodes_created);
    
    // Update reach probabilities for the recursive call
    const double previous_reach_probability = reach_probabilities[player];
    reach_probabilities[player] =
        previous_reach_probability * choice.probability;

    const WeightedHandRange* child_player_a_range =
        player_a_range;
    const WeightedHandRange* child_player_b_range =
        player_b_range;
    if (player == 0 && player_a_range != nullptr) {
      condition_range_for_action(
          *player_a_range, node->state, player, legal_action_ids,
          action_id, &conditioned_player_range);
      child_player_a_range = &conditioned_player_range;
    } else if (player == 1 && player_b_range != nullptr) {
      condition_range_for_action(
          *player_b_range, node->state, player, legal_action_ids,
          action_id, &conditioned_player_range);
      child_player_b_range = &conditioned_player_range;
    }
    
    // Recursive call to get the expected value of this action
    choice.value = cfr_with_ranges(
        child_node, player_a_hand, player_b_hand, reach_probabilities,
        iteration, depth + 1, max_depth, child_player_a_range,
        child_player_b_range);
    reach_probabilities[player] = previous_reach_probability;
    
    // Update the expected value of the node
    node_value += choice.probability * choice.value;
  }
  
  // Compute counterfactual regrets if this is not a chance player
  if (player == 0 || player == 1) {
    ++cfr_update_count_;
    ++traversal_stats_.cfr_updates;
    ++traversal_stats_.canonical_state_visits;
    if (config_.enable_logging()) {
      if (visited_canonical_states_.insert(
              MakeCanonicalPublicStateKey(node->state)).second) {
        ++traversal_stats_.unique_canonical_states;
      } else {
        ++traversal_stats_.duplicate_canonical_state_visits;
      }
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
    InfoSetData& info_set = info_sets_[info_set_id];
    
    // For each action, compute and accumulate the counterfactual regret
    for (const ActionChoice& choice : action_choices) {
      // get_utility returns player A's utility; player B's regret uses the
      // opposite payoff in this zero-sum game.
      double utility_sign = player == 0 ? 1.0 : -1.0;
      double regret =
          opponent_reach_prob * utility_sign * (choice.value - node_value);
      
      // CFR+ clips cumulative regrets at zero.
      double& cumulative_regret =
          info_set.cumulative_regrets[choice.action_index];
      cumulative_regret = std::max(0.0, cumulative_regret + regret);
    }
    
    // CFR+ commonly weights later average-strategy samples more heavily.
    update_strategy(info_set_id, action_choices,
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
                      int max_depth,
                      const WeightedHandRange* player_a_range,
                      const WeightedHandRange* player_b_range) {
  int samples = ChanceSamples(config_);
  traversal_stats_.chance_samples += samples;
  return SampleChanceValue(
      game_tree_, node, player_a_hand, player_b_hand, samples,
      &rng_, &traversal_stats_.child_nodes_created,
      [&](GameTree::Node* child_node) {
        return cfr_with_ranges(child_node, player_a_hand, player_b_hand,
                               reach_probabilities, iteration, depth,
                               max_depth, player_a_range, player_b_range);
      });
}

double CFRSolver::action_probability_for_hand(
    const BoardState& state,
    int player,
    const Hand& hand,
    const std::vector<int>& legal_action_ids,
    int action_id) const {
  if (legal_action_ids.empty()) {
    return 0.0;
  }
  if (std::find(legal_action_ids.begin(), legal_action_ids.end(),
                action_id) == legal_action_ids.end()) {
    return 0.0;
  }

  if (!legacy_info_sets_.empty()) {
    std::string info_set_key = info_set_abstraction_->state_to_info_set(
        state, player, hand);
    const LegacyInfoSetData* legacy_info_set =
        find_legacy_info_set(info_set_key);
    if (legacy_info_set != nullptr) {
      double sum_positive_regrets = 0.0;
      for (int legal_action_id : legal_action_ids) {
        auto action = std::find(legacy_info_set->action_ids.begin(),
                                legacy_info_set->action_ids.end(),
                                legal_action_id);
        if (action != legacy_info_set->action_ids.end()) {
          const size_t index =
              static_cast<size_t>(action - legacy_info_set->action_ids.begin());
          sum_positive_regrets +=
              std::max(0.0, legacy_info_set->cumulative_regrets[index]);
        }
      }
      if (sum_positive_regrets <= 0.0) {
        return 1.0 / legal_action_ids.size();
      }

      auto action = std::find(legacy_info_set->action_ids.begin(),
                              legacy_info_set->action_ids.end(), action_id);
      if (action == legacy_info_set->action_ids.end()) {
        return 0.0;
      }
      const size_t index =
          static_cast<size_t>(action - legacy_info_set->action_ids.begin());
      return std::max(0.0, legacy_info_set->cumulative_regrets[index]) /
             sum_positive_regrets;
    }
  }

  InfoSetKey key = make_info_set_key(state, player, hand);
  const InfoSetData* info_set = find_info_set(key);
  if (info_set != nullptr) {
    double sum_positive_regrets = 0.0;
    for (int legal_action_id : legal_action_ids) {
      auto action = std::find(info_set->action_ids.begin(),
                              info_set->action_ids.end(), legal_action_id);
      if (action != info_set->action_ids.end()) {
        const size_t index =
            static_cast<size_t>(action - info_set->action_ids.begin());
        sum_positive_regrets += std::max(0.0,
                                         info_set->cumulative_regrets[index]);
      }
    }
    if (sum_positive_regrets <= 0.0) {
      return 1.0 / legal_action_ids.size();
    }

    auto action = std::find(info_set->action_ids.begin(),
                            info_set->action_ids.end(), action_id);
    if (action == info_set->action_ids.end()) {
      return 0.0;
    }
    const size_t index =
        static_cast<size_t>(action - info_set->action_ids.begin());
    return std::max(0.0, info_set->cumulative_regrets[index]) /
           sum_positive_regrets;
  }

  return 1.0 / legal_action_ids.size();
}

void CFRSolver::condition_range_for_action(
    const WeightedHandRange& range,
    const BoardState& state,
    int player,
    const std::vector<int>& legal_action_ids,
    int action_id,
    WeightedHandRange* conditioned_range) const {
  conditioned_range->clear();
  conditioned_range->reserve(range.size());
  for (size_t i = 0; i < range.size(); ++i) {
    if (range.weights[i] <= 0.0) {
      continue;
    }
    double probability = action_probability_for_hand(
        state, player, range.hands[i], legal_action_ids, action_id);
    double conditioned_weight = range.weights[i] * probability;
    if (conditioned_weight > 0.0) {
      conditioned_range->add(range.hands[i], conditioned_weight);
    }
  }
}

ContinuationContext CFRSolver::build_continuation_context(
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand,
    const WeightedHandRange* player_a_range,
    const WeightedHandRange* player_b_range) const {
  ContinuationContext context =
      ContinuationContext::ExactHands(state, player_a_hand, player_b_hand);
  if (player_a_range != nullptr && player_b_range != nullptr) {
    context.player_a_range = PublicCompatibleRange(*player_a_range, state);
    context.player_b_range = PublicCompatibleRange(*player_b_range, state);
  }
  return context;
}

Strategy CFRSolver::get_equilibrium_strategy() const {
  Strategy equilibrium_strategy;

  for (const InfoSetData& info_set : info_sets_) {
    double sum = 0.0;
    for (double probability : info_set.cumulative_strategy) {
      sum += probability;
    }

    Strategy::ActionProbabilities normalized_strategy;
    normalized_strategy.reserve(info_set.action_ids.size());
    if (sum > 0.0) {
      for (size_t i = 0; i < info_set.action_ids.size(); ++i) {
        normalized_strategy[info_set.action_ids[i]] =
            info_set.cumulative_strategy[i] / sum;
      }
    } else if (!info_set.action_ids.empty()) {
      double uniform_prob = 1.0 / info_set.action_ids.size();
      for (int action_id : info_set.action_ids) {
        normalized_strategy[action_id] = uniform_prob;
      }
    }

    equilibrium_strategy.update(info_set_key_to_string(info_set.key),
                                normalized_strategy);
  }
  
  for (const LegacyInfoSetData& info_set : legacy_info_sets_) {
    double sum = 0.0;
    for (double probability : info_set.cumulative_strategy) {
      sum += probability;
    }
    
    Strategy::ActionProbabilities normalized_strategy;
    normalized_strategy.reserve(info_set.action_ids.size());
    if (sum > 0.0) {
      for (size_t i = 0; i < info_set.action_ids.size(); ++i) {
        normalized_strategy[info_set.action_ids[i]] =
            info_set.cumulative_strategy[i] / sum;
      }
    } else if (!info_set.action_ids.empty()) {
      double uniform_prob = 1.0 / info_set.action_ids.size();
      for (int action_id : info_set.action_ids) {
        normalized_strategy[action_id] = uniform_prob;
      }
    }
    
    equilibrium_strategy.update(info_set.key, normalized_strategy);
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
        &rng_, nullptr, [&](GameTree::Node* child_node) {
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
    GameTree::Node* child_node =
        CachedActionChild(game_tree_, node, action, action_id, nullptr);
    value += probability * evaluate_strategy_node(
                              child_node, player_a_hand, player_b_hand,
                              strategy);
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
        &rng_, nullptr, [&](GameTree::Node* child_node) {
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
      GameTree::Node* child_node =
          CachedActionChild(game_tree_, node, action, action_id, nullptr);
      value = std::max(value, best_response_value(
                                  child_node, player_a_hand, player_b_hand,
                                  strategy, best_response_player));
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
    GameTree::Node* child_node =
        CachedActionChild(game_tree_, node, action, action_id, nullptr);
    value += probability * best_response_value(
                               child_node, player_a_hand, player_b_hand,
                               strategy, best_response_player);
  }
  return value;
}

double CFRSolver::best_response_value_against_range(
    GameTree::Node* node,
    const Hand& best_response_hand,
    const WeightedHandRange& opponent_hands,
    const Strategy& strategy,
    int best_response_player) {
  double total_weight = TotalWeight(opponent_hands);
  if (total_weight <= 0.0) {
    return 0.0;
  }

  if (node->is_terminal) {
    double value = 0.0;
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      const Hand& player_a_hand =
          best_response_player == 0 ? best_response_hand
                                    : opponent_hands.hands[i];
      const Hand& player_b_hand =
          best_response_player == 0 ? opponent_hands.hands[i]
                                    : best_response_hand;
      double player_a_value =
          utility(node->state, player_a_hand, player_b_hand);
      value += opponent_hands.weights[i] *
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
          CachedChanceChild(game_tree_, node, cards, nullptr);
      WeightedHandRange child_opponents;
      child_opponents.add(sampled_opponent, 1.0);
      value += best_response_value_against_range(
          child_node, best_response_hand, child_opponents, strategy,
          best_response_player);
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
      GameTree::Node* child_node =
          CachedActionChild(game_tree_, node, action, action_id, nullptr);
      value = std::max(value, best_response_value_against_range(
                                  child_node, best_response_hand,
                                  opponent_hands, strategy,
                                  best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    GameTree::Node* child_node =
        CachedActionChild(game_tree_, node, action, action_id, nullptr);

    WeightedHandRange child_opponents;
    child_opponents.reserve(opponent_hands.size());
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      std::string info_set_key = info_set_abstraction_->state_to_info_set(
          node->state, player, opponent_hands.hands[i]);
      double probability = StrategyActionProbability(
          strategy, info_set_key, node->legal_actions, action_id);
      if (probability > 0.0) {
        child_opponents.add(opponent_hands.hands[i],
                            opponent_hands.weights[i] * probability);
      }
    }

    double child_weight = TotalWeight(child_opponents);
    if (child_weight > 0.0) {
      value += (child_weight / total_weight) *
               best_response_value_against_range(
                   child_node, best_response_hand, child_opponents, strategy,
                   best_response_player);
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

  WeightedHandRange best_response_hands =
      best_response_range.get_all_weighted_combos();
  WeightedHandRange opponent_hands =
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
    const WeightedHandRange& best_response_hands,
    const WeightedHandRange& opponent_hands,
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
    WeightedHandRange compatible_opponents =
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
    GameTree::Node* child_node =
        CachedActionChild(game_tree_, node, action, action_id, nullptr);
    double value = best_response_value(child_node, player_a_hand,
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
  legacy_info_set_ids_.clear();
  legacy_info_sets_.clear();
  info_set_ids_.clear();
  info_sets_.clear();

  for (const StrategyInfoSetSnapshot& info_set : snapshot.info_sets()) {
    Strategy::ActionProbabilities action_probs;
    const int info_set_id =
        get_or_create_legacy_info_set_id(info_set.info_set_key());
    LegacyInfoSetData& legacy_info_set = legacy_info_sets_[info_set_id];
    for (const StrategyActionSnapshot& action : info_set.actions()) {
      if (action.has_action()) {
        const int action_id = ActionKey(action.action());
        action_probs[action_id] = action.probability();
        const size_t action_index =
            ensure_legacy_info_set_action(&legacy_info_set, action_id);
        legacy_info_set.cumulative_strategy[action_index] =
            action.probability();
      }
    }
    current_strategy_.update(info_set.info_set_key(), action_probs);
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
  if (legal_actions.empty()) {
    return strategy;
  }

  const int info_set_id = get_or_create_legacy_info_set_id(info_set_key);
  LegacyInfoSetData& info_set = legacy_info_sets_[info_set_id];
  ensure_legacy_info_set_actions(&info_set, legal_actions);

  std::vector<int> action_ids;
  action_ids.reserve(legal_actions.size());
  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    if (std::find(action_ids.begin(), action_ids.end(), action_id) ==
        action_ids.end()) {
      action_ids.push_back(action_id);
    }
  }
  strategy.reserve(action_ids.size());
  
  // Compute the sum of positive regrets
  double sum_positive_regrets = 0.0;
  for (int action_id : action_ids) {
    auto action = std::find(info_set.action_ids.begin(),
                            info_set.action_ids.end(), action_id);
    if (action == info_set.action_ids.end()) {
      continue;
    }
    const size_t action_index =
        static_cast<size_t>(action - info_set.action_ids.begin());
    sum_positive_regrets +=
        std::max(0.0, info_set.cumulative_regrets[action_index]);
  }
  
  // If there are positive regrets, use regret matching
  if (sum_positive_regrets > 0.0) {
    for (int action_id : action_ids) {
      auto action = std::find(info_set.action_ids.begin(),
                              info_set.action_ids.end(), action_id);
      if (action == info_set.action_ids.end()) {
        continue;
      }
      const size_t action_index =
          static_cast<size_t>(action - info_set.action_ids.begin());
      strategy[action_id] =
          std::max(0.0, info_set.cumulative_regrets[action_index]) /
          sum_positive_regrets;
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

std::vector<CFRSolver::ActionChoice> CFRSolver::get_action_choices(
    const std::string& info_set_key,
    const std::vector<Action>& legal_actions) {
  std::vector<ActionChoice> choices;
  choices.reserve(legal_actions.size());
  if (legal_actions.empty()) {
    return choices;
  }

  const int info_set_id = get_or_create_legacy_info_set_id(info_set_key);
  LegacyInfoSetData& info_set = legacy_info_sets_[info_set_id];
  ensure_legacy_info_set_actions(&info_set, legal_actions);

  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    auto existing_choice =
        std::find_if(choices.begin(), choices.end(),
                     [action_id](const ActionChoice& choice) {
                       return choice.action_id == action_id;
                     });
    if (existing_choice != choices.end()) {
      continue;
    }
    auto action_index = std::find(info_set.action_ids.begin(),
                                  info_set.action_ids.end(), action_id);
    if (action_index == info_set.action_ids.end()) {
      continue;
    }
    choices.push_back(
        {&action, action_id,
         static_cast<size_t>(action_index - info_set.action_ids.begin()), 0.0,
         0.0});
  }

  double sum_positive_regrets = 0.0;
  for (const ActionChoice& choice : choices) {
    sum_positive_regrets +=
        std::max(0.0, info_set.cumulative_regrets[choice.action_index]);
  }

  if (sum_positive_regrets > 0.0) {
    for (ActionChoice& choice : choices) {
      choice.probability =
          std::max(0.0, info_set.cumulative_regrets[choice.action_index]) /
          sum_positive_regrets;
    }
  } else if (!choices.empty()) {
    double uniform_prob = 1.0 / choices.size();
    for (ActionChoice& choice : choices) {
      choice.probability = uniform_prob;
    }
  }

  return choices;
}

std::vector<CFRSolver::ActionChoice> CFRSolver::get_action_choices(
    int info_set_id,
    const std::vector<Action>& legal_actions) {
  InfoSetData& info_set = info_sets_[info_set_id];
  ensure_info_set_actions(&info_set, legal_actions);

  std::vector<ActionChoice> choices;
  choices.reserve(legal_actions.size());
  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    auto existing_choice =
        std::find_if(choices.begin(), choices.end(),
                     [action_id](const ActionChoice& choice) {
                       return choice.action_id == action_id;
                     });
    if (existing_choice != choices.end()) {
      continue;
    }
    auto action_index = std::find(info_set.action_ids.begin(),
                                  info_set.action_ids.end(), action_id);
    if (action_index == info_set.action_ids.end()) {
      continue;
    }
    choices.push_back(
        {&action, action_id,
         static_cast<size_t>(action_index - info_set.action_ids.begin()), 0.0,
         0.0});
  }

  double sum_positive_regrets = 0.0;
  for (const ActionChoice& choice : choices) {
    sum_positive_regrets +=
        std::max(0.0, info_set.cumulative_regrets[choice.action_index]);
  }

  if (sum_positive_regrets > 0.0) {
    for (ActionChoice& choice : choices) {
      choice.probability =
          std::max(0.0, info_set.cumulative_regrets[choice.action_index]) /
          sum_positive_regrets;
    }
  } else if (!choices.empty()) {
    double uniform_prob = 1.0 / choices.size();
    for (ActionChoice& choice : choices) {
      choice.probability = uniform_prob;
    }
  }

  return choices;
}

void CFRSolver::update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob) {
  const int info_set_id = get_or_create_legacy_info_set_id(info_set_key);
  LegacyInfoSetData& info_set = legacy_info_sets_[info_set_id];
  for (const auto& action_prob : strategy) {
    const size_t action_index =
        ensure_legacy_info_set_action(&info_set, action_prob.first);
    info_set.cumulative_strategy[action_index] +=
        reach_prob * action_prob.second;
  }
}

void CFRSolver::update_strategy(const std::string& info_set_key,
                                const std::vector<ActionChoice>& choices,
                                double reach_prob) {
  const int info_set_id = get_or_create_legacy_info_set_id(info_set_key);
  LegacyInfoSetData& info_set = legacy_info_sets_[info_set_id];
  for (const ActionChoice& choice : choices) {
    const size_t action_index =
        ensure_legacy_info_set_action(&info_set, choice.action_id);
    info_set.cumulative_strategy[action_index] +=
        reach_prob * choice.probability;
  }
}

void CFRSolver::update_strategy(int info_set_id,
                                const std::vector<ActionChoice>& choices,
                                double reach_prob) {
  InfoSetData& info_set = info_sets_[info_set_id];
  for (const ActionChoice& choice : choices) {
    info_set.cumulative_strategy[choice.action_index] +=
        reach_prob * choice.probability;
  }
}

} // namespace poker
