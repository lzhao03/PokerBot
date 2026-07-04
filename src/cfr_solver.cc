#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "src/card_utils.h"
#include "src/continuation_value.h"
#include "src/hand_range.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include <algorithm>
#include <cstdint>
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
constexpr char kInfoSetKeyVersion[] = "exact_cards_v1";
constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kParallelBestResponseSampleThreshold = 32;

#ifndef POKER_ENABLE_TRAVERSAL_STATS
#define POKER_ENABLE_TRAVERSAL_STATS 1
#endif

#if POKER_ENABLE_TRAVERSAL_STATS
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
    statement;                                 \
  } while (false)
#define POKER_TRAVERSAL_STAT_PTR(member) (&(member))
#else
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
  } while (false)
#define POKER_TRAVERSAL_STAT_PTR(member) nullptr
#endif

int ActionKey(const Action& action) {
  return GameTree::action_key(action);
}

Action ActionFromKey(int action_key) {
  Action action;
  action.set_action(static_cast<ActionType>(action_key / kActionKeyMultiplier));
  action.set_amount(action_key % kActionKeyMultiplier);
  return action;
}

double TotalWeight(const WeightedHandRangeView& hands) {
  double total = 0.0;
  for (size_t i = 0; i < hands.size(); ++i) {
    total += hands.weight(i);
  }
  return total;
}

WeightedHandRangeView CompatibleHands(
    const WeightedHandRangeView& hands,
    CardMask known_hand_mask,
    const BoardState& state) {
  WeightedHandRangeView compatible_hands;
  if (!hands.has_source()) {
    return compatible_hands;
  }

  const CardMask blocked_cards = known_hand_mask | BoardMask(state);
  compatible_hands.reset_to_filtered(hands.source_range());
  compatible_hands.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    if ((hands.mask(i) & blocked_cards) == 0) {
      compatible_hands.add(hands.source_index(i), hands.weight(i));
    }
  }
  return compatible_hands;
}

void PublicCompatibleRangeInto(const TrainingRangeView& hands,
                               const BoardState& state,
                               TrainingRangeView& compatible_hands) {
  if (hands.empty()) {
    compatible_hands.clear();
    return;
  }

  const CardMask board_mask = BoardMask(state);
  compatible_hands.reset_to_filtered();
  for (size_t i = 0; i < hands.size(); ++i) {
    if (hands.weight(i) > 0.0 && (hands.mask(i) & board_mask) == 0) {
      compatible_hands.add(hands.combo(i), hands.weight(i));
    }
  }
}

int EncodedCard(const Card& card) {
  return card.rank() * 8 + static_cast<int>(card.suit());
}

int EncodedCardId(uint8_t card_id) {
  const int rank = 2 + card_id % 13;
  const int suit = 1 + card_id / 13;
  return rank * 8 + suit;
}

int ChanceCardsKey(const std::vector<Card>& cards) {
  int key = static_cast<int>(cards.size());
  for (const Card& card : cards) {
    key = key * 128 + EncodedCard(card);
  }
  return -1 - key;
}

GameTree::Node& CachedChanceChild(GameTree& game_tree,
                                  GameTree::Node& node,
                                  const std::vector<Card>& cards,
                                  int64_t* created_nodes) {
  const int child_key = ChanceCardsKey(cards);
  auto child = node.children.find(child_key);
  if (child != node.children.end()) {
    return game_tree.node(child->second);
  }

  if (created_nodes != nullptr) {
    ++*created_nodes;
  }
  return game_tree.create_chance_child_node(node, child_key, cards);
}

GameTree::Node& CachedActionChild(GameTree& game_tree,
                                  GameTree::Node& node,
                                  const Action& action,
                                  int action_id,
                                  int64_t* created_nodes) {
  auto child = node.children.find(action_id);
  if (child != node.children.end()) {
    return game_tree.node(child->second);
  }

  if (created_nodes != nullptr) {
    ++*created_nodes;
  }
  return game_tree.create_child_node(node, action_id, action);
}

void EnsureLegalActionIds(GameTree::Node& node) {
  if (node.legal_action_ids.size() == node.legal_actions.size()) {
    return;
  }

  node.legal_action_ids.clear();
  node.legal_action_ids.reserve(node.legal_actions.size());
  for (const Action& action : node.legal_actions) {
    node.legal_action_ids.push_back(ActionKey(action));
  }
}

int RoundedContribution(const BoardState& state, int player) {
  return state.player_contribution_size() > player
             ? static_cast<int>(std::lround(state.player_contribution(player)))
             : 0;
}

void HashCombine(size_t& seed, int value) {
  seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <size_t N>
void HashArray(size_t& seed, const std::array<int, N>& values) {
  for (int value : values) {
    HashCombine(seed, value);
  }
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
double SampleChanceValue(GameTree& game_tree,
                         GameTree::Node& node,
                         CardMask known_private_cards,
                         int samples,
                         std::mt19937& rng,
                         int64_t* created_nodes,
                         EvaluateChild evaluate_child) {
  double value = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<Card> cards =
        SampleStreetCards(node.state, known_private_cards, rng);
    GameTree::Node& child_node =
        CachedChanceChild(game_tree, node, cards, created_nodes);
    value += evaluate_child(child_node);
  }
  return value / samples;
}

template <typename EvaluateChild>
double SampleChanceValue(GameTree& game_tree,
                         GameTree::Node& node,
                         const Hand& player_a_hand,
                         const Hand& player_b_hand,
                         int samples,
                         std::mt19937& rng,
                         int64_t* created_nodes,
                         EvaluateChild evaluate_child) {
  return SampleChanceValue(
      game_tree, node, HandMask(player_a_hand) | HandMask(player_b_hand),
      samples, rng, created_nodes, evaluate_child);
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
  HashCombine(seed, key.player);
  HashCombine(seed, key.street);
  HashCombine(seed, key.pot);
  HashCombine(seed, key.stack_a);
  HashCombine(seed, key.stack_b);
  HashCombine(seed, key.all_in);
  HashCombine(seed, key.folded_player);
  HashCombine(seed, key.player_to_act);
  HashCombine(seed, key.player_contribution_size);
  HashArray(seed, key.player_contributions);
  HashCombine(seed, key.hand_size);
  HashArray(seed, key.hand_cards);
  HashCombine(seed, key.board_size);
  HashArray(seed, key.board_cards);
  HashCombine(seed, key.history_size);
  HashArray(seed, key.history_values);
  for (int value : key.history_overflow) {
    HashCombine(seed, value);
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
    game_tree_(std::make_unique<GameTree>(config)),
    rng_(12345),
    cumulative_root_utility_(0.0),
    iterations_run_(0),
    cfr_update_count_(0),
    utility_cache_(std::move(utility_cache)),
    continuation_value_provider_(std::move(continuation_value_provider)) {
}

void CFRSolver::set_continuation_value_provider(
    std::shared_ptr<ContinuationValueProvider> provider) {
  if (provider == nullptr) {
    throw std::invalid_argument("Continuation value provider cannot be null");
  }
  continuation_value_provider_ = std::move(provider);
}

GameTree::Node& CFRSolver::get_or_build_root() {
  if (!game_tree_->has_root()) {
    return game_tree_->build_tree(initial_state_);
  }
  return game_tree_->root();
}

CFRSolver::PrivateCards CFRSolver::PrivateCards::FromHand(const Hand& hand) {
  std::optional<ComboId> combo_id = MaybeHandToComboId(hand);
  if (combo_id.has_value()) {
    return FromCombo(*combo_id);
  }

  PrivateCards private_cards;
  private_cards.hand = hand;
  return private_cards;
}

CFRSolver::PrivateCards CFRSolver::PrivateCards::FromCombo(
    ComboId combo_id) {
  PrivateCards private_cards;
  private_cards.has_combo = true;
  private_cards.combo = combo_id;
  return private_cards;
}

CardMask CFRSolver::PrivateCards::mask() const {
  return has_combo ? ComboMask(combo) : HandMask(hand);
}

Hand CFRSolver::PrivateCards::to_hand() const {
  return has_combo ? ComboIdToHand(combo) : hand;
}

CFRSolver::RangeSampler::RangeSampler(const TrainingRange& player_a_range,
                                       const TrainingRange& player_b_range)
    : player_a_range(player_a_range),
      player_b_range(player_b_range),
      compatible_player_b_weight(kComboCount, 0.0f) {
  float total_weight = 0.0f;
  player_a_sample_weights.reserve(player_a_range.active_count);
  for (uint16_t a = 0; a < player_a_range.active_count; ++a) {
    const ComboId player_a_combo = player_a_range.active[a];
    const float player_a_weight = player_a_range.weights[player_a_combo];
    if (player_a_weight <= 0.0f) {
      player_a_sample_weights.push_back(0.0f);
      continue;
    }
    for (uint16_t b = 0; b < player_b_range.active_count; ++b) {
      const ComboId player_b_combo = player_b_range.active[b];
      const float player_b_weight = player_b_range.weights[player_b_combo];
      if (player_b_weight <= 0.0f ||
          (ComboMask(player_a_combo) & ComboMask(player_b_combo)) != 0) {
        continue;
      }
      compatible_player_b_weight[player_a_combo] += player_b_weight;
    }
    const float sample_weight =
        player_a_weight * compatible_player_b_weight[player_a_combo];
    player_a_sample_weights.push_back(sample_weight);
    total_weight += sample_weight;
  }

  if (total_weight <= 0.0f) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }

  player_a_distribution = std::discrete_distribution<size_t>(
      player_a_sample_weights.begin(), player_a_sample_weights.end());
}

CFRSolver::RangeDeal CFRSolver::RangeSampler::sample(std::mt19937& rng) {
  const size_t player_a_active_index = player_a_distribution(rng);
  const ComboId player_a_combo = player_a_range.active[player_a_active_index];
  const CardMask blocked_cards = ComboMask(player_a_combo);
  const float total_player_b_weight =
      compatible_player_b_weight[player_a_combo];
  std::uniform_real_distribution<float> distribution(
      0.0f, total_player_b_weight);
  float remaining = distribution(rng);
  std::optional<ComboId> fallback_player_b_combo;

  for (uint16_t b = 0; b < player_b_range.active_count; ++b) {
    const ComboId player_b_combo = player_b_range.active[b];
    const float player_b_weight = player_b_range.weights[player_b_combo];
    if (player_b_weight <= 0.0f ||
        (ComboMask(player_b_combo) & blocked_cards) != 0) {
      continue;
    }
    fallback_player_b_combo = player_b_combo;
    remaining -= player_b_weight;
    if (remaining <= 0.0f) {
      return RangeDeal(player_a_combo, player_b_combo);
    }
  }

  if (!fallback_player_b_combo.has_value()) {
    throw std::logic_error("Range sampler selected an incompatible hand");
  }

  return RangeDeal(player_a_combo, *fallback_player_b_combo);
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const BoardState& state,
    int player,
    const Hand& hand) const {
  InfoSetKey key = make_public_info_set_key(state, player);
  key.hand_size = std::min(hand.cards_size(), 2);
  for (int i = 0; i < key.hand_size; ++i) {
    key.hand_cards[i] = EncodedCard(hand.cards(i));
  }
  std::sort(key.hand_cards.begin(),
            key.hand_cards.begin() + key.hand_size);
  return key;
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const BoardState& state,
    int player,
    ComboId combo_id) const {
  InfoSetKey key = make_public_info_set_key(state, player);
  const ComboInfo& combo = GetComboInfo(combo_id);
  key.hand_size = 2;
  key.hand_cards[0] = EncodedCardId(combo.card0);
  key.hand_cards[1] = EncodedCardId(combo.card1);
  std::sort(key.hand_cards.begin(),
            key.hand_cards.begin() + key.hand_size);
  return key;
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const BoardState& state,
    int player,
    const PrivateCards& private_cards) const {
  if (private_cards.has_combo) {
    return make_info_set_key(state, player, private_cards.combo);
  }
  return make_info_set_key(state, player, private_cards.hand);
}

CFRSolver::InfoSetKey CFRSolver::make_public_info_set_key(
    const BoardState& state,
    int player) const {
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

  key.board_size = std::min(state.cards_size(), InfoSetKey::kMaxCards);
  for (int i = 0; i < key.board_size; ++i) {
    key.board_cards[i] = EncodedCard(state.cards(i));
  }
  std::sort(key.board_cards.begin(),
            key.board_cards.begin() + key.board_size);

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

void CFRSolver::initialize_info_set_actions(
    InfoSetData& info_set,
    const std::vector<int>& legal_action_ids) {
  info_set.actions.reserve(legal_action_ids.size());
  for (int action_id : legal_action_ids) {
    info_set.actions.push_back({action_id, 0.0, 0.0});
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
  }
}

int CFRSolver::get_or_create_info_set_id(
    const InfoSetKey& key,
    const std::vector<int>& legal_action_ids) {
  auto existing = info_set_ids_.find(key);
  if (existing != info_set_ids_.end()) {
    return existing->second;
  }

  if (info_sets_.size() == info_sets_.capacity()) {
    const size_t new_capacity =
        info_sets_.empty() ? 1024 : info_sets_.capacity() * 2;
    info_sets_.reserve(new_capacity);
    info_set_ids_.reserve(new_capacity);
  }

  const int id = static_cast<int>(info_sets_.size());
  InfoSetData data;
  data.key = key;
  initialize_info_set_actions(data, legal_action_ids);
  info_sets_.push_back(std::move(data));
  info_set_ids_.emplace(info_sets_.back().key, id);
  return id;
}

std::string CFRSolver::info_set_key_to_string(const InfoSetKey& key) const {
  std::ostringstream text;
  text << 'P' << key.player << ":H[";
  for (int i = 0; i < key.hand_size; ++i) {
    if (i > 0) {
      text << ',';
    }
    text << key.hand_cards[i] / 8 << ':' << key.hand_cards[i] % 8;
  }
  text << "]:B[";
  for (int i = 0; i < key.board_size; ++i) {
    if (i > 0) {
      text << ',';
    }
    text << key.board_cards[i] / 8 << ':' << key.board_cards[i] % 8;
  }
  text << "]:S" << key.street << ":POT" << key.pot << ":ST"
       << key.stack_a << ',' << key.stack_b << ":AI" << key.all_in << ":F"
       << key.folded_player << ":T" << key.player_to_act << ":C[";
  for (int i = 0; i < key.player_contribution_size; ++i) {
    if (i > 0) {
      text << ',';
    }
    text << key.player_contributions[i];
  }
  text << "]:A[";
  for (int i = 0; i < key.history_size; i += 3) {
    if (i > 0) {
      text << ',';
    }
    auto history_value = [&](int index) {
      if (index < InfoSetKey::kInlineHistoryValues) {
        return key.history_values[index];
      }
      return key.history_overflow[index - InfoSetKey::kInlineHistoryValues];
    };
    text << history_value(i) << ':' << history_value(i + 1) << ':'
         << history_value(i + 2);
  }
  text << ']';
  return text.str();
}

double CFRSolver::regret_for_info_set(const std::string& info_set_key,
                                      int action_id) const {
  for (const InfoSetData& info_set : info_sets_) {
    if (info_set_key_to_string(info_set.key) != info_set_key) {
      continue;
    }
    auto action = std::find_if(
        info_set.actions.begin(), info_set.actions.end(),
        [action_id](const ActionState& action_state) {
          return action_state.action_id == action_id;
        });
    if (action == info_set.actions.end()) {
      return 0.0;
    }
    return action->cumulative_regret;
  }
  return 0.0;
}

void CFRSolver::run(int iterations, const Hand& player_a_hand,
                    const Hand& player_b_hand) {
  if (iterations <= 0) {
    return;
  }

  GameTree::Node& root = get_or_build_root();
  const int max_depth = config_.max_depth();
  TraversalScratch scratch;
  const PrivateCards player_a_cards = PrivateCards::FromHand(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromHand(player_b_hand);
  for (int i = 0; i < iterations; ++i) {
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    cumulative_root_utility_ += cfr_with_ranges(
        root, player_a_cards, player_b_cards, reach_probabilities,
        iterations_run_, 0, max_depth, scratch, std::nullopt, std::nullopt);
    ++iterations_run_;
  }
}

void CFRSolver::run(int iterations, const HandRange& player_a_range,
                    const HandRange& player_b_range) {
  run_iterations(iterations, player_a_range, player_b_range);
}

void CFRSolver::run_iterations(int iterations,
                               const HandRange& player_a_range,
                               const HandRange& player_b_range) {
  if (iterations <= 0) {
    return;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  TrainingRangeView player_a_hands_view;
  TrainingRangeView player_b_hands_view;
  player_a_hands_view.reset_to_all(player_a_training_range);
  player_b_hands_view.reset_to_all(player_b_training_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);

  VLOG(1) << "Preparing game tree...";
  bool had_root = game_tree_->has_root();
  GameTree::Node& root = get_or_build_root();
  if (!had_root) {
    VLOG(1) << "Game tree built with " << root.legal_actions.size()
            << " legal actions at root";
  } else {
    VLOG(1) << "Reusing game tree with " << root.legal_actions.size()
            << " legal actions at root";
  }
  
  // Run iterations of CFR
  LOG(INFO) << "Starting CFR iterations...";
  TraversalScratch scratch;
  for (int i = 0; i < iterations; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    PrivateCards player_a_cards = PrivateCards::FromCombo(deal.player_a_combo);
    PrivateCards player_b_cards = PrivateCards::FromCombo(deal.player_b_combo);

    const int max_depth = config_.max_depth();
    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_;
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    double dealt_value = cfr_with_ranges(
        root, player_a_cards, player_b_cards, reach_probabilities,
        cfr_iteration, 0, max_depth, scratch, player_a_context_range,
        player_b_context_range);

    cumulative_root_utility_ += dealt_value;
    ++iterations_run_;
  }
  
  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_;
  LOG(INFO) << "Information sets: " << info_sets_.size();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

double CFRSolver::cfr(GameTree::Node& node,
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::array<double, 2>& reach_probabilities,
                      int iteration,
                      int depth,
                      int max_depth) {
  TraversalScratch scratch;
  return cfr_with_ranges(node, player_a_hand, player_b_hand,
                         reach_probabilities, iteration, depth, max_depth,
                         scratch, std::nullopt, std::nullopt);
}

double CFRSolver::cfr_with_ranges(
    GameTree::Node& node,
    const Hand& player_a_hand,
    const Hand& player_b_hand,
    std::array<double, 2>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  return cfr_with_ranges(
      node, PrivateCards::FromHand(player_a_hand),
      PrivateCards::FromHand(player_b_hand), reach_probabilities, iteration,
      depth, max_depth, scratch, player_a_range, player_b_range);
}

double CFRSolver::cfr_with_ranges(
    GameTree::Node& node,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  // If the node is a terminal node, return the utility
  if (node.is_terminal) {
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.terminal_utility_calls);
    if (node.state.folded_player() >= 0) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.fold_utility_calls);
    } else {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.showdown_utility_calls);
    }
    if (max_depth > 0) {
      return uncached_utility(node.state, player_a_cards, player_b_cards);
    }
    return utility(node.state, player_a_cards, player_b_cards);
  }

  // Chance card deals are not player decisions, so they do not consume CFR depth.
  if (node.is_chance_node) {
    return chance_sampling_cfr(node, player_a_cards, player_b_cards,
                               reach_probabilities, iteration, depth, max_depth,
                               scratch, player_a_range, player_b_range);
  }

  // Check depth limit to prevent infinite recursion
  if (max_depth > 0 && depth >= max_depth) {
    ContinuationContext context = build_continuation_context(
        node.state, player_a_cards.to_hand(), player_b_cards.to_hand(),
        player_a_range,
        player_b_range);
    return continuation_value_provider_->value(*game_tree_, context);
  }
  
  // Get the player to act at this node
  int player = node.player_to_act;
  
  // Get the player's hand
  const PrivateCards& player_cards =
      (player == 0) ? player_a_cards : player_b_cards;
  
  InfoSetKey info_set_key =
      make_info_set_key(node.state, player, player_cards);
  EnsureLegalActionIds(node);
  const int info_set_id =
      get_or_create_info_set_id(info_set_key, node.legal_action_ids);
  ActionChoices action_choices;
  action_choices.reserve(node.legal_actions.size());
  {
    InfoSetData& info_set = info_sets_[info_set_id];
    for (size_t i = 0; i < node.legal_actions.size(); ++i) {
      action_choices.push_back(
          {std::cref(node.legal_actions[i]), node.legal_action_ids[i], 0.0,
           0.0});
    }

    double sum_positive_regrets = 0.0;
    for (size_t action_index = 0; action_index < action_choices.size();
         ++action_index) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      sum_positive_regrets +=
          std::max(0.0, info_set.actions[action_index].cumulative_regret);
    }

    if (sum_positive_regrets > 0.0) {
      for (size_t action_index = 0; action_index < action_choices.size();
           ++action_index) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        ActionChoice& choice = action_choices[action_index];
        choice.probability =
            std::max(
                0.0,
                info_set.actions[action_index].cumulative_regret) /
            sum_positive_regrets;
      }
    } else if (!action_choices.empty()) {
      double uniform_prob = 1.0 / action_choices.size();
      for (ActionChoice& choice : action_choices) {
        choice.probability = uniform_prob;
      }
    }
  }
  
  // Initialize the expected value for the player
  double node_value = 0.0;
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  ConditionedRanges& conditioned_player_ranges =
      scratch_frame.conditioned_ranges;
  const bool condition_player_a_range =
      player == 0 && player_a_range.has_value();
  const bool condition_player_b_range =
      player == 1 && player_b_range.has_value();
  if (condition_player_a_range) {
    condition_ranges_for_actions(player_a_range->get(), node.state, player,
                                 action_choices, conditioned_player_ranges);
  } else if (condition_player_b_range) {
    condition_ranges_for_actions(player_b_range->get(), node.state, player,
                                 action_choices, conditioned_player_ranges);
  }

  // For each action, recursively call CFR and compute the expected value
  for (size_t choice_index = 0; choice_index < action_choices.size();
       ++choice_index) {
    ActionChoice& choice = action_choices[choice_index];
    const Action& action = choice.action.get();
    int action_id = choice.action_id;
    
    GameTree::Node& child_node = CachedActionChild(
        *game_tree_, node, action, action_id,
        POKER_TRAVERSAL_STAT_PTR(traversal_stats_.child_nodes_created));
    
    // Update reach probabilities for the recursive call
    const double previous_reach_probability = reach_probabilities[player];
    reach_probabilities[player] =
        previous_reach_probability * choice.probability;

    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (condition_player_a_range) {
      child_player_a_range = std::cref(conditioned_player_ranges[choice_index]);
    } else if (condition_player_b_range) {
      child_player_b_range = std::cref(conditioned_player_ranges[choice_index]);
    }
    
    // Recursive call to get the expected value of this action
    choice.value = cfr_with_ranges(
        child_node, player_a_cards, player_b_cards, reach_probabilities,
        iteration, depth + 1, max_depth, scratch, child_player_a_range,
        child_player_b_range);
    reach_probabilities[player] = previous_reach_probability;
    
    // Update the expected value of the node
    node_value += choice.probability * choice.value;
  }
  
  // Compute counterfactual regrets if this is not a chance player
  if (player == 0 || player == 1) {
    ++cfr_update_count_;
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.cfr_updates);
    POKER_RECORD_TRAVERSAL_STAT(
        traversal_stats_.max_decision_depth =
            std::max(traversal_stats_.max_decision_depth, depth));
    switch (node.state.street()) {
      case Street::PREFLOP:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.preflop_updates);
        break;
      case Street::FLOP:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.flop_updates);
        break;
      case Street::TURN:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.turn_updates);
        break;
      case Street::RIVER:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.river_updates);
        break;
      default:
        break;
    }

    // Compute the counterfactual reach probability of the opponent
    double opponent_reach_prob = reach_probabilities[1 - player];
    InfoSetData& info_set = info_sets_[info_set_id];
    // For each action, compute and accumulate the counterfactual regret
    for (size_t action_index = 0; action_index < action_choices.size();
         ++action_index) {
      const ActionChoice& choice = action_choices[action_index];
      // get_utility returns player A's utility; player B's regret uses the
      // opposite payoff in this zero-sum game.
      double utility_sign = player == 0 ? 1.0 : -1.0;
      double regret =
          opponent_reach_prob * utility_sign * (choice.value - node_value);
      
      // CFR+ clips cumulative regrets at zero.
      double& cumulative_regret =
          info_set.actions[action_index].cumulative_regret;
      POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
      cumulative_regret = std::max(0.0, cumulative_regret + regret);
    }
    
    // CFR+ commonly weights later average-strategy samples more heavily.
    update_strategy(info_set_id, action_choices,
                    reach_probabilities[player] * (iteration + 1));
  }
  
  return node_value;
}

double CFRSolver::chance_sampling_cfr(GameTree::Node& node,
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::array<double, 2>& reach_probabilities,
                      int iteration,
                      int depth,
                      int max_depth,
                      TraversalScratch& scratch,
                      OptionalTrainingRange player_a_range,
                      OptionalTrainingRange player_b_range) {
  return chance_sampling_cfr(
      node, PrivateCards::FromHand(player_a_hand),
      PrivateCards::FromHand(player_b_hand), reach_probabilities, iteration,
      depth, max_depth, scratch, player_a_range, player_b_range);
}

double CFRSolver::chance_sampling_cfr(GameTree::Node& node,
                      const PrivateCards& player_a_cards,
                      const PrivateCards& player_b_cards,
                      std::array<double, 2>& reach_probabilities,
                      int iteration,
                      int depth,
                      int max_depth,
                      TraversalScratch& scratch,
                      OptionalTrainingRange player_a_range,
                      OptionalTrainingRange player_b_range) {
  int samples = ChanceSamples(config_);
  POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.chance_samples += samples);
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  TrainingRangeView& public_player_a_range =
      scratch_frame.public_player_a_range;
  TrainingRangeView& public_player_b_range =
      scratch_frame.public_player_b_range;
  return SampleChanceValue(
      *game_tree_, node, player_a_cards.mask() | player_b_cards.mask(), samples,
      rng_, POKER_TRAVERSAL_STAT_PTR(traversal_stats_.child_nodes_created),
      [&](GameTree::Node& child_node) {
        OptionalTrainingRange child_player_a_range = player_a_range;
        OptionalTrainingRange child_player_b_range = player_b_range;
        if (player_a_range.has_value()) {
          PublicCompatibleRangeInto(
              player_a_range->get(), child_node.state, public_player_a_range);
          child_player_a_range = std::cref(public_player_a_range);
        }
        if (player_b_range.has_value()) {
          PublicCompatibleRangeInto(
              player_b_range->get(), child_node.state, public_player_b_range);
          child_player_b_range = std::cref(public_player_b_range);
        }
        return cfr_with_ranges(child_node, player_a_cards, player_b_cards,
                               reach_probabilities, iteration, depth,
                               max_depth, scratch, child_player_a_range,
                               child_player_b_range);
      });
}

double CFRSolver::average_strategy_action_probability(
    const BoardState& state,
    int player,
    const Hand& hand,
    const std::vector<Action>& legal_actions,
    int action_id) {
  return average_strategy_action_probability(
      state, player, PrivateCards::FromHand(hand), legal_actions, action_id);
}

double CFRSolver::average_strategy_action_probability(
    const BoardState& state,
    int player,
    const PrivateCards& private_cards,
    const std::vector<Action>& legal_actions,
    int action_id) {
  if (legal_actions.empty()) {
    return 0.0;
  }
  const double uniform_probability = 1.0 / legal_actions.size();

  InfoSetKey key = make_info_set_key(state, player, private_cards);
  auto existing_info_set = info_set_ids_.find(key);
  if (existing_info_set != info_set_ids_.end()) {
    return average_strategy_action_probability(
        info_sets_[existing_info_set->second], legal_actions, action_id,
        uniform_probability);
  }

  if (!loaded_strategy_.empty()) {
    return StrategyActionProbability(
        loaded_strategy_, info_set_key_to_string(key), legal_actions,
        action_id);
  }
  return uniform_probability;
}

double CFRSolver::average_strategy_action_probability(
    const InfoSetData& info_set,
    const std::vector<Action>& legal_actions,
    int action_id,
    double fallback_probability) {
  StrategyProbabilities probabilities;
  average_strategy_probabilities(
      info_set, legal_actions, fallback_probability, probabilities);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    if (ActionKey(legal_actions[i]) == action_id) {
      return probabilities[i];
    }
  }
  return fallback_probability;
}

void CFRSolver::average_strategy_probabilities(
    const BoardState& state,
    int player,
    const PrivateCards& private_cards,
    const std::vector<Action>& legal_actions,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(legal_actions.size(), 0.0);
  if (legal_actions.empty()) {
    return;
  }

  const double uniform_probability = 1.0 / legal_actions.size();
  InfoSetKey key = make_info_set_key(state, player, private_cards);
  auto existing_info_set = info_set_ids_.find(key);
  if (existing_info_set != info_set_ids_.end()) {
    average_strategy_probabilities(
        info_sets_[existing_info_set->second], legal_actions,
        uniform_probability, probabilities);
    return;
  }

  if (loaded_strategy_.empty()) {
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }

  double probability_sum = 0.0;
  const std::string info_set_key = info_set_key_to_string(key);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    probabilities[i] = loaded_strategy_.get_action_probability(
        info_set_key, ActionKey(legal_actions[i]));
    probability_sum += probabilities[i];
  }
  if (probability_sum <= 0.0) {
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }
  for (double& probability : probabilities) {
    probability /= probability_sum;
  }
}

void CFRSolver::average_strategy_probabilities(
    const InfoSetData& info_set,
    const std::vector<Action>& legal_actions,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(legal_actions.size(), 0.0);
  double probability_sum = 0.0;

  const bool aligned_action_ids =
      legal_actions.size() == info_set.actions.size() &&
      std::equal(legal_actions.begin(), legal_actions.end(),
                 info_set.actions.begin(),
                 [](const Action& legal_action,
                    const ActionState& action_state) {
                   return ActionKey(legal_action) == action_state.action_id;
                 });
  if (aligned_action_ids) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      probabilities[i] = info_set.actions[i].cumulative_strategy;
      probability_sum += probabilities[i];
    }
  } else {
    for (size_t legal_action_index = 0;
         legal_action_index < legal_actions.size(); ++legal_action_index) {
      const int legal_action_id = ActionKey(legal_actions[legal_action_index]);
      for (const ActionState& action_state : info_set.actions) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        if (action_state.action_id == legal_action_id) {
          probabilities[legal_action_index] = action_state.cumulative_strategy;
          probability_sum += probabilities[legal_action_index];
          break;
        }
      }
    }
  }

  if (probability_sum <= 0.0) {
    std::fill(probabilities.begin(), probabilities.end(), fallback_probability);
    return;
  }
  for (double& probability : probabilities) {
    probability /= probability_sum;
  }
}

void CFRSolver::condition_ranges_for_actions(
    const TrainingRangeView& range,
    const BoardState& state,
    int player,
    const ActionChoices& action_choices,
    ConditionedRanges& conditioned_ranges) {
  while (conditioned_ranges.size() < action_choices.size()) {
    conditioned_ranges.emplace_back();
  }
  if (action_choices.empty()) {
    return;
  }

  for (size_t i = 0; i < action_choices.size(); ++i) {
    conditioned_ranges[i].reset_to_filtered();
  }
  if (range.empty()) {
    return;
  }

  const double fallback_probability = 1.0 / action_choices.size();
  const CardMask board_mask = BoardMask(state);
  absl::InlinedVector<double, 8> positive_regrets(
      action_choices.size(), 0.0);
  for (size_t i = 0; i < range.size(); ++i) {
    if (range.weight(i) <= 0.0 || (range.mask(i) & board_mask) != 0) {
      continue;
    }

    double positive_regret_sum = 0.0;
    InfoSetKey key = make_info_set_key(state, player, range.combo(i));
    auto existing_info_set = info_set_ids_.find(key);
    if (existing_info_set != info_set_ids_.end()) {
      const InfoSetData& info_set = info_sets_[existing_info_set->second];
      std::fill(positive_regrets.begin(), positive_regrets.end(), 0.0);
      const size_t action_count =
          std::min(action_choices.size(), info_set.actions.size());
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        if (info_set.actions[action_index].action_id !=
            action_choices[action_index].action_id) {
          continue;
        }

        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const double positive_regret =
            std::max(0.0,
                     info_set.actions[action_index].cumulative_regret);
        positive_regrets[action_index] = positive_regret;
        positive_regret_sum += positive_regret;
      }
    }

    for (size_t action_index = 0; action_index < action_choices.size();
         ++action_index) {
      const double probability =
          positive_regret_sum > 0.0
              ? positive_regrets[action_index] / positive_regret_sum
              : fallback_probability;
      const double conditioned_weight = range.weight(i) * probability;
      if (conditioned_weight > 0.0) {
        conditioned_ranges[action_index].add(
            range.combo(i), static_cast<float>(conditioned_weight));
      }
    }
  }
}

ContinuationContext CFRSolver::build_continuation_context(
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) const {
  ContinuationContext context =
      ContinuationContext::ExactHands(state, player_a_hand, player_b_hand);
  if (player_a_range.has_value() && player_b_range.has_value()) {
    PublicCompatibleRangeInto(
        player_a_range->get(), state, context.player_a_range);
    PublicCompatibleRangeInto(
        player_b_range->get(), state, context.player_b_range);
  }
  return context;
}

Strategy CFRSolver::get_equilibrium_strategy() const {
  Strategy equilibrium_strategy = loaded_strategy_;

  for (const InfoSetData& info_set : info_sets_) {
    double sum = 0.0;
    for (const ActionState& action : info_set.actions) {
      sum += action.cumulative_strategy;
    }

    Strategy::ActionProbabilities normalized_strategy;
    normalized_strategy.reserve(info_set.actions.size());
    if (sum > 0.0) {
      for (const ActionState& action : info_set.actions) {
        normalized_strategy[action.action_id] =
            action.cumulative_strategy / sum;
      }
    } else if (!info_set.actions.empty()) {
      double uniform_prob = 1.0 / info_set.actions.size();
      for (const ActionState& action : info_set.actions) {
        normalized_strategy[action.action_id] = uniform_prob;
      }
    }

    equilibrium_strategy.update(info_set_key_to_string(info_set.key),
                                normalized_strategy);
  }

  return equilibrium_strategy;
}

double CFRSolver::evaluate_strategy(const Hand& player_a_hand,
                                    const Hand& player_b_hand) {
  return evaluate_strategy_node(get_or_build_root(), player_a_hand,
                                player_b_hand);
}

double CFRSolver::evaluate_strategy(int samples, const HandRange& player_a_range,
                                    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);

  if (samples < kParallelEvaluationSampleThreshold) {
    return evaluate_strategy_samples(samples, range_sampler);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return evaluate_strategy_samples(samples, range_sampler);
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
  auto strategy_info_set_ids = info_set_ids_;
  auto strategy_info_sets = info_sets_;
  Strategy loaded_strategy_copy = loaded_strategy_;
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, range_sampler,
                                       &loaded_strategy_copy,
                                       &strategy_info_set_ids,
                                       &strategy_info_sets,
                                       utility_cache, continuation_value_provider,
                                       shard_samples, seed]() mutable {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.info_set_ids_ = strategy_info_set_ids;
      worker.info_sets_ = strategy_info_sets;
      worker.loaded_strategy_ = loaded_strategy_copy;
      worker.rng_.seed(seed);
      const double value = worker.evaluate_strategy_samples(
          shard_samples, range_sampler);
      return std::make_pair(
          value * shard_samples,
          worker.get_traversal_stats().action_entry_touches);
    }));
  }

  double total = 0.0;
  for (std::future<std::pair<double, int64_t>>& future : futures) {
    const std::pair<double, int64_t> result = future.get();
    total += result.first;
    POKER_RECORD_TRAVERSAL_STAT(
        traversal_stats_.action_entry_touches += result.second);
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_samples(
    int samples,
    RangeSampler range_sampler) {
  if (samples <= 0) {
    return 0.0;
  }

  GameTree::Node& root = get_or_build_root();

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    total += evaluate_strategy_node(
        root, PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo));
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(GameTree::Node& node,
                                         const Hand& player_a_hand,
                                         const Hand& player_b_hand) {
  return evaluate_strategy_node(node, PrivateCards::FromHand(player_a_hand),
                                PrivateCards::FromHand(player_b_hand));
}

double CFRSolver::evaluate_strategy_node(GameTree::Node& node,
                                         const PrivateCards& player_a_cards,
                                         const PrivateCards& player_b_cards) {
  if (node.is_terminal) {
    return utility(node.state, player_a_cards, player_b_cards);
  }
  if (node.is_chance_node) {
    return SampleChanceValue(
        *game_tree_, node, player_a_cards.mask() | player_b_cards.mask(),
        ChanceSamples(config_),
        rng_, nullptr, [&](GameTree::Node& child_node) {
          return evaluate_strategy_node(child_node, player_a_cards,
                                        player_b_cards);
        });
  }
  if (node.legal_actions.empty()) {
    return 0.0;
  }

  int player = node.player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;
  StrategyProbabilities probabilities;
  average_strategy_probabilities(
      node.state, player, player_cards, node.legal_actions, probabilities);

  double value = 0.0;
  for (size_t action_index = 0; action_index < node.legal_actions.size();
       ++action_index) {
    const Action& action = node.legal_actions[action_index];
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);
    value += probabilities[action_index] *
             evaluate_strategy_node(
                 child_node, player_a_cards, player_b_cards);
  }
  return value;
}

double CFRSolver::best_response_value(GameTree::Node& node,
                                      const Hand& player_a_hand,
                                      const Hand& player_b_hand,
                                      int best_response_player) {
  if (node.is_terminal) {
    double player_a_value =
        utility(node.state, player_a_hand, player_b_hand);
    return best_response_player == 0 ? player_a_value : -player_a_value;
  }
  if (node.is_chance_node) {
    return SampleChanceValue(
        *game_tree_, node, player_a_hand, player_b_hand, ChanceSamples(config_),
        rng_, nullptr, [&](GameTree::Node& child_node) {
          return best_response_value(child_node, player_a_hand, player_b_hand,
                                     best_response_player);
        });
  }
  if (node.legal_actions.empty()) {
    return 0.0;
  }

  int player = node.player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const Hand& player_hand = player == 0 ? player_a_hand : player_b_hand;

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (const Action& action : node.legal_actions) {
      int action_id = ActionKey(action);
      GameTree::Node& child_node =
          CachedActionChild(*game_tree_, node, action, action_id,
                            nullptr);
      value = std::max(value, best_response_value(
                                  child_node, player_a_hand, player_b_hand,
                                  best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (const Action& action : node.legal_actions) {
    int action_id = ActionKey(action);
    double probability = average_strategy_action_probability(
        node.state, player, player_hand, node.legal_actions, action_id);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);
    value += probability * best_response_value(
                               child_node, player_a_hand, player_b_hand,
                               best_response_player);
  }
  return value;
}

double CFRSolver::best_response_value_against_range(
    GameTree::Node& node,
    const Hand& best_response_hand,
    const WeightedHandRangeView& opponent_hands,
    int best_response_player) {
  return best_response_value_against_range(
      node, PrivateCards::FromHand(best_response_hand), opponent_hands,
      best_response_player);
}

double CFRSolver::best_response_value_against_range(
    GameTree::Node& node,
    const PrivateCards& best_response_cards,
    const WeightedHandRangeView& opponent_hands,
    int best_response_player) {
  double total_weight = TotalWeight(opponent_hands);
  if (total_weight <= 0.0) {
    return 0.0;
  }

  if (node.is_terminal) {
    double value = 0.0;
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      const PrivateCards opponent_cards =
          PrivateCards::FromHand(opponent_hands.hand(i));
      const PrivateCards& player_a_cards =
          best_response_player == 0 ? best_response_cards : opponent_cards;
      const PrivateCards& player_b_cards =
          best_response_player == 0 ? opponent_cards : best_response_cards;
      double player_a_value =
          utility(node.state, player_a_cards, player_b_cards);
      value += opponent_hands.weight(i) *
               (best_response_player == 0 ? player_a_value : -player_a_value);
    }
    return value / total_weight;
  }

  if (node.is_chance_node) {
    double value = 0.0;
    int samples = ChanceSamples(config_);
    std::vector<double> opponent_weights;
    opponent_weights.reserve(opponent_hands.size());
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      opponent_weights.push_back(opponent_hands.weight(i));
    }
    std::discrete_distribution<size_t> opponent_distribution(
        opponent_weights.begin(), opponent_weights.end());
    for (int i = 0; i < samples; ++i) {
      size_t sampled_opponent_view_index = opponent_distribution(rng_);
      size_t sampled_opponent_index =
          opponent_hands.source_index(sampled_opponent_view_index);
      const PrivateCards sampled_opponent = PrivateCards::FromHand(
          opponent_hands.source_range().hands[sampled_opponent_index]);
      std::vector<Card> cards =
          SampleStreetCards(
              node.state, best_response_cards.mask() | sampled_opponent.mask(),
              rng_);
      GameTree::Node& child_node =
          CachedChanceChild(*game_tree_, node, cards, nullptr);
      WeightedHandRangeView child_opponents;
      child_opponents.reset_to_filtered(opponent_hands.source_range());
      child_opponents.add(sampled_opponent_index, 1.0);
      value += best_response_value_against_range(
          child_node, best_response_cards, child_opponents,
          best_response_player);
    }
    return value / samples;
  }

  if (node.legal_actions.empty()) {
    return 0.0;
  }

  int player = node.player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (const Action& action : node.legal_actions) {
      int action_id = ActionKey(action);
      GameTree::Node& child_node =
          CachedActionChild(*game_tree_, node, action, action_id,
                            nullptr);
      value = std::max(value, best_response_value_against_range(
                                  child_node, best_response_cards,
                                  opponent_hands, best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (const Action& action : node.legal_actions) {
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);

    WeightedHandRangeView child_opponents;
    child_opponents.reset_to_filtered(opponent_hands.source_range());
    child_opponents.reserve(opponent_hands.size());
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      double probability = average_strategy_action_probability(
          node.state, player, PrivateCards::FromHand(opponent_hands.hand(i)),
          node.legal_actions, action_id);
      if (probability > 0.0) {
        child_opponents.add(opponent_hands.source_index(i),
                            opponent_hands.weight(i) * probability);
      }
    }

    double child_weight = TotalWeight(child_opponents);
    if (child_weight > 0.0) {
      value += (child_weight / total_weight) *
               best_response_value_against_range(
                   child_node, best_response_cards, child_opponents,
                   best_response_player);
    }
  }
  return value;
}

double CFRSolver::sampled_range_best_response_value(
    int samples,
    const HandRange& best_response_range,
    const HandRange& opponent_range,
    int best_response_player) {
  if (samples <= 0) {
    return 0.0;
  }

  const WeightedHandRange& best_response_hands =
      best_response_range.get_all_weighted_combos();
  const WeightedHandRange& opponent_hands =
      opponent_range.get_all_weighted_combos();
  if (best_response_hands.empty() || opponent_hands.empty()) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }

  if (samples < kParallelBestResponseSampleThreshold) {
    return sampled_range_best_response_samples(
        samples, best_response_hands, opponent_hands, best_response_player);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return sampled_range_best_response_samples(
        samples, best_response_hands, opponent_hands, best_response_player);
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
  auto strategy_info_set_ids = info_set_ids_;
  auto strategy_info_sets = info_sets_;
  Strategy loaded_strategy_copy = loaded_strategy_;
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, &best_response_hands,
                                       &opponent_hands, &loaded_strategy_copy,
                                       &strategy_info_set_ids,
                                       &strategy_info_sets,
                                       utility_cache, continuation_value_provider,
                                       shard_samples, seed, best_response_player]() {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.info_set_ids_ = strategy_info_set_ids;
      worker.info_sets_ = strategy_info_sets;
      worker.loaded_strategy_ = loaded_strategy_copy;
      worker.rng_.seed(seed);
      const double value = worker.sampled_range_best_response_samples(
          shard_samples, best_response_hands, opponent_hands,
          best_response_player);
      return std::make_pair(
          value * shard_samples,
          worker.get_traversal_stats().action_entry_touches);
    }));
  }

  double total = 0.0;
  for (std::future<std::pair<double, int64_t>>& future : futures) {
    const std::pair<double, int64_t> result = future.get();
    total += result.first;
    POKER_RECORD_TRAVERSAL_STAT(
        traversal_stats_.action_entry_touches += result.second);
  }
  return total / samples;
}

double CFRSolver::sampled_range_best_response_samples(
    int samples,
    const WeightedHandRange& best_response_hands,
    const WeightedHandRange& opponent_hands,
    int best_response_player) {
  if (samples <= 0) {
    return 0.0;
  }

  std::discrete_distribution<size_t> best_response_hand_distribution(
      best_response_hands.weights.begin(), best_response_hands.weights.end());
  GameTree::Node& root = get_or_build_root();
  WeightedHandRangeView opponent_view(opponent_hands);
  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    const PrivateCards best_response_cards = PrivateCards::FromHand(
        best_response_hands.hands[best_response_hand_distribution(rng_)]);
    WeightedHandRangeView compatible_opponents =
        CompatibleHands(opponent_view, best_response_cards.mask(), root.state);
    if (compatible_opponents.empty()) {
      throw std::invalid_argument(
          "Could not sample non-overlapping hands from ranges");
    }
    total += best_response_value_against_range(root, best_response_cards,
                                               compatible_opponents,
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
    Hand player_a_hand = DealHand(deck);
    Hand player_b_hand = DealHand(deck);
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

  double strategy_player_a_value =
      evaluate_strategy(samples, player_a_range, player_b_range);
  double player_a_gap =
      sampled_range_best_response_value(samples, player_a_range, player_b_range,
                                        0) -
      strategy_player_a_value;
  double player_b_gap =
      sampled_range_best_response_value(samples, player_b_range, player_a_range,
                                        1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

double CFRSolver::calculate_player_a_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return sampled_range_best_response_value(samples, player_a_range,
                                           player_b_range, 0);
}

double CFRSolver::calculate_player_b_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return sampled_range_best_response_value(samples, player_b_range,
                                           player_a_range, 1);
}

double CFRSolver::calculate_exploitability(const Hand& player_a_hand,
                                           const Hand& player_b_hand) {
  GameTree::Node& root = get_or_build_root();
  double strategy_player_a_value =
      evaluate_strategy_node(root, player_a_hand, player_b_hand);
  double player_a_gap =
      best_response_value(root, player_a_hand, player_b_hand, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_hand, player_b_hand, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

Action CFRSolver::get_best_response_action(GameTree::Node& node,
                                           const Hand& player_a_hand,
                                           const Hand& player_b_hand,
                                           int best_response_player) {
  Action no_action;
  no_action.set_action(ActionType::NO_ACTION);
  if (node.is_terminal || node.is_chance_node ||
      node.legal_actions.empty() || node.player_to_act != best_response_player) {
    return no_action;
  }

  double best_value = -std::numeric_limits<double>::infinity();
  Action best_action = no_action;
  for (const Action& action : node.legal_actions) {
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);
    double value = best_response_value(child_node, player_a_hand,
                                       player_b_hand, best_response_player);
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
  snapshot.set_abstraction_version(kInfoSetKeyVersion);

  for (const std::string& info_set_key : equilibrium_strategy.get_info_sets()) {
    StrategyInfoSetSnapshot& info_set = *snapshot.add_info_sets();
    info_set.set_info_set_key(info_set_key);

    Strategy::ActionProbabilities action_probs =
        equilibrium_strategy.get_strategy(info_set_key);
    for (const auto& action_prob : action_probs) {
      StrategyActionSnapshot& action = *info_set.add_actions();
      *action.mutable_action() = ActionFromKey(action_prob.first);
      action.set_probability(action_prob.second);
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

  loaded_strategy_.clear();
  info_set_ids_.clear();
  info_sets_.clear();

  for (const StrategyInfoSetSnapshot& info_set : snapshot.info_sets()) {
    Strategy::ActionProbabilities action_probs;
    for (const StrategyActionSnapshot& action : info_set.actions()) {
      if (action.has_action()) {
        const int action_id = ActionKey(action.action());
        action_probs[action_id] = action.probability();
      }
    }
    loaded_strategy_.update(info_set.info_set_key(), action_probs);
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

double CFRSolver::utility(const BoardState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  const double player_a_contribution =
      state.player_contribution_size() > 0 ? state.player_contribution(0) : 0.0;
  if (state.folded_player() == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player() == 1) {
    return state.pot() - player_a_contribution;
  }

  const int player_a_card_count =
      player_a_cards.has_combo ? 2 : player_a_cards.hand.cards_size();
  if (player_a_card_count + state.cards_size() < 5) {
    return 0.0;
  }

  const Hand player_a_hand = player_a_cards.to_hand();
  const Hand player_b_hand = player_b_cards.to_hand();
  return utility(state, player_a_hand, player_b_hand);
}

double CFRSolver::uncached_utility(const BoardState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  const double player_a_contribution =
      state.player_contribution_size() > 0 ? state.player_contribution(0) : 0.0;
  if (state.folded_player() == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player() == 1) {
    return state.pot() - player_a_contribution;
  }

  const int player_a_card_count =
      player_a_cards.has_combo ? 2 : player_a_cards.hand.cards_size();
  if (player_a_card_count + state.cards_size() < 5) {
    return 0.0;
  }

  const Hand player_a_hand = player_a_cards.to_hand();
  const Hand player_b_hand = player_b_cards.to_hand();
  return game_tree_->get_utility(state, player_a_hand, player_b_hand);
}

void CFRSolver::update_strategy(int info_set_id,
                                const ActionChoices& choices,
                                double reach_prob) {
  InfoSetData& info_set = info_sets_[info_set_id];
  for (size_t action_index = 0; action_index < choices.size();
       ++action_index) {
    POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
    info_set.actions[action_index].cumulative_strategy +=
        reach_prob * choices[action_index].probability;
  }
}

} // namespace poker
