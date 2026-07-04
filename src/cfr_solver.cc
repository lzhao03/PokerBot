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

int ActionKey(const GameAction& action) {
  return GameTree::action_key(action);
}

Action ProtoActionFromKey(int action_key) {
  Action action;
  action.set_action(static_cast<ActionType>(action_key / kActionKeyMultiplier));
  action.set_amount(action_key % kActionKeyMultiplier);
  return action;
}

int ProtoActionKey(const Action& action) {
  const double amount = action.amount();
  const int rounded_amount = static_cast<int>(std::lround(amount));
  if (amount < 0.0 || rounded_amount < 0 ||
      rounded_amount >= kActionKeyMultiplier) {
    throw std::invalid_argument("Action amount is outside action-key range");
  }
  return static_cast<int>(action.action()) * kActionKeyMultiplier +
         rounded_amount;
}

ActionKind ToActionKind(ActionType action_type) {
  switch (action_type) {
    case ActionType::BET:
      return ActionKind::kBet;
    case ActionType::FOLD:
      return ActionKind::kFold;
    case ActionType::CALL:
      return ActionKind::kCall;
    case ActionType::RAISE:
      return ActionKind::kRaise;
    case ActionType::CHECK:
      return ActionKind::kCheck;
    case ActionType::ALL_IN:
      return ActionKind::kAllIn;
    case ActionType::NO_ACTION:
    default:
      return ActionKind::kNoAction;
  }
}

StreetKind ToStreetKind(Street street) {
  switch (street) {
    case Street::FLOP:
      return StreetKind::kFlop;
    case Street::TURN:
      return StreetKind::kTurn;
    case Street::RIVER:
      return StreetKind::kRiver;
    case Street::PREFLOP:
    default:
      return StreetKind::kPreflop;
  }
}

SuitKind ToSuitKind(Suit suit) {
  switch (suit) {
    case Suit::DIAMONDS:
      return SuitKind::kDiamonds;
    case Suit::CLUBS:
      return SuitKind::kClubs;
    case Suit::SPADES:
      return SuitKind::kSpades;
    case Suit::HEARTS:
    default:
      return SuitKind::kHearts;
  }
}

CardId ProtoCardToId(const Card& card) {
  return MakeCardId(card.rank(), ToSuitKind(card.suit()));
}

ComboId ProtoHandToComboId(const Hand& hand) {
  if (hand.cards_size() != 2) {
    return 0;
  }
  return CardsToComboId(ProtoCardToId(hand.cards(0)),
                        ProtoCardToId(hand.cards(1)));
}

GameAction GameActionFromProto(const Action& action) {
  return {ToActionKind(action.action()),
          static_cast<int>(std::lround(action.amount())),
          action.player()};
}

SolverConfig SolverConfigFromProto(const PokerConfig& config) {
  SolverConfig native;
  native.bet_sizes.assign(config.bet_sizes().begin(), config.bet_sizes().end());
  native.starting_stack_size = config.starting_stack_size();
  native.max_depth = config.max_depth();
  native.enable_logging = config.enable_logging();
  native.small_blind = config.small_blind();
  native.big_blind = config.big_blind();
  native.chance_samples = config.chance_samples();
  native.preflop_bet_sizes.assign(config.preflop_bet_sizes().begin(),
                                  config.preflop_bet_sizes().end());
  native.flop_bet_sizes.assign(config.flop_bet_sizes().begin(),
                               config.flop_bet_sizes().end());
  native.turn_bet_sizes.assign(config.turn_bet_sizes().begin(),
                               config.turn_bet_sizes().end());
  native.river_bet_sizes.assign(config.river_bet_sizes().begin(),
                                config.river_bet_sizes().end());
  native.regret_only_training = config.regret_only_training();
  return native;
}

PokerConfig SolverConfigToProto(const SolverConfig& config) {
  PokerConfig proto;
  for (double bet_size : config.bet_sizes) {
    proto.add_bet_sizes(bet_size);
  }
  proto.set_starting_stack_size(config.starting_stack_size);
  proto.set_max_depth(config.max_depth);
  proto.set_enable_logging(config.enable_logging);
  proto.set_small_blind(config.small_blind);
  proto.set_big_blind(config.big_blind);
  proto.set_chance_samples(config.chance_samples);
  for (double bet_size : config.preflop_bet_sizes) {
    proto.add_preflop_bet_sizes(bet_size);
  }
  for (double bet_size : config.flop_bet_sizes) {
    proto.add_flop_bet_sizes(bet_size);
  }
  for (double bet_size : config.turn_bet_sizes) {
    proto.add_turn_bet_sizes(bet_size);
  }
  for (double bet_size : config.river_bet_sizes) {
    proto.add_river_bet_sizes(bet_size);
  }
  proto.set_regret_only_training(config.regret_only_training);
  return proto;
}

GameState GameStateFromProto(const BoardState& state) {
  GameState native;
  native.stack[0] = state.stack_a();
  native.stack[1] = state.stack_b();
  native.pot = state.pot();
  native.street = ToStreetKind(state.street());
  native.all_in = state.all_in();
  native.folded_player = state.folded_player();
  native.player_to_act = state.player_to_act();
  native.player_contribution[0] =
      state.player_contribution_size() > 0
          ? static_cast<int>(std::lround(state.player_contribution(0)))
          : 0;
  native.player_contribution[1] =
      state.player_contribution_size() > 1
          ? static_cast<int>(std::lround(state.player_contribution(1)))
          : 0;
  for (const Card& card : state.cards()) {
    AddBoardCard(native, ProtoCardToId(card));
  }
  for (const Action& action : state.history().actions()) {
    native.history.push_back(GameActionFromProto(action));
  }
  return native;
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
    const GameState& state) {
  WeightedHandRangeView compatible_hands;
  if (!hands.has_source()) {
    return compatible_hands;
  }

  const CardMask blocked_cards = known_hand_mask | state.board_mask;
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
                               const GameState& state,
                               TrainingRangeView& compatible_hands) {
  if (hands.empty()) {
    compatible_hands.clear();
    return;
  }

  const CardMask board_mask = state.board_mask;
  compatible_hands.reset_to_filtered();
  for (size_t i = 0; i < hands.size(); ++i) {
    if (hands.weight(i) > 0.0 && (hands.mask(i) & board_mask) == 0) {
      compatible_hands.add(hands.combo(i), hands.weight(i));
    }
  }
}

int ChanceCardsKey(const std::vector<CardId>& cards) {
  absl::InlinedVector<int, 5> encoded_cards;
  encoded_cards.reserve(cards.size());
  for (CardId card : cards) {
    encoded_cards.push_back(EncodedCard(card));
  }
  std::sort(encoded_cards.begin(), encoded_cards.end());

  int key = static_cast<int>(cards.size());
  for (int encoded_card : encoded_cards) {
    key = key * 128 + encoded_card;
  }
  return -1 - key;
}

GameTree::Node& CachedChanceChild(GameTree& game_tree,
                                  GameTree::Node& node,
                                  const std::vector<CardId>& cards,
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
                                  const GameAction& action,
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
  for (const GameAction& action : node.legal_actions) {
    node.legal_action_ids.push_back(ActionKey(action));
  }
}

int RoundedContribution(const GameState& state, int player) {
  return state.player_contribution[player];
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
                                 const std::vector<GameAction>& legal_actions,
                                 int action_id) {
  double probability_sum = 0.0;
  for (const GameAction& legal_action : legal_actions) {
    probability_sum +=
        strategy.get_action_probability(info_set_key, ActionKey(legal_action));
  }
  if (probability_sum > 0.0) {
    return strategy.get_action_probability(info_set_key, action_id) /
           probability_sum;
  }
  return legal_actions.empty() ? 0.0 : 1.0 / legal_actions.size();
}

int ChanceSamples(const SolverConfig& config) {
  return std::max(1, config.chance_samples);
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
    std::vector<CardId> cards =
        SampleStreetCards(node.state, known_private_cards, rng);
    GameTree::Node& child_node =
        CachedChanceChild(game_tree, node, cards, created_nodes);
    value += evaluate_child(child_node);
  }
  return value / samples;
}

GameState DefaultInitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const int starting_stack = config.starting_stack_size;

  GameState initial_state;
  initial_state.stack[0] = std::max(0, starting_stack - small_blind);
  initial_state.stack[1] = std::max(0, starting_stack - big_blind);
  initial_state.pot = small_blind + big_blind;
  initial_state.folded_player = -1;
  initial_state.street = StreetKind::kPreflop;
  initial_state.all_in = false;
  initial_state.player_to_act = 0;
  initial_state.player_contribution = {small_blind, big_blind};
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

bool CFRSolver::PublicStateKey::operator==(
    const PublicStateKey& other) const {
  return street == other.street && pot == other.pot &&
         stack_a == other.stack_a && stack_b == other.stack_b &&
         all_in == other.all_in && folded_player == other.folded_player &&
         player_to_act == other.player_to_act &&
         player_contribution_size == other.player_contribution_size &&
         player_contributions == other.player_contributions &&
         board_size == other.board_size && board_cards == other.board_cards &&
         history_size == other.history_size &&
         history_values == other.history_values &&
         history_overflow == other.history_overflow;
}

bool CFRSolver::CompactInfoSetKey::operator==(
    const CompactInfoSetKey& other) const {
  return public_state_id == other.public_state_id &&
         private_combo == other.private_combo && player == other.player;
}

size_t CFRSolver::CompactInfoSetKeyHash::operator()(
    const CompactInfoSetKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<int>(key.public_state_id));
  HashCombine(seed, static_cast<int>(key.private_combo));
  HashCombine(seed, static_cast<int>(key.player));
  return seed;
}

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>()) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const GameState& initial_state)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>(),
                std::make_shared<BettingRoundTerminalValueProvider>(),
                initial_state) {
}

CFRSolver::CFRSolver(const PokerConfig& config)
    : CFRSolver(SolverConfigFromProto(config)) {
}

CFRSolver::CFRSolver(const PokerConfig& config,
                     const BoardState& initial_state)
    : CFRSolver(SolverConfigFromProto(config), GameStateFromProto(initial_state)) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     std::shared_ptr<TerminalUtilityCache> utility_cache)
    : CFRSolver(config, std::move(utility_cache),
                std::make_shared<BettingRoundTerminalValueProvider>()) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider)
    : CFRSolver(config, std::move(utility_cache),
                std::move(continuation_value_provider),
                DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    std::shared_ptr<ContinuationValueProvider> continuation_value_provider,
    GameState initial_state)
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

CFRSolver::PrivateCards CFRSolver::PrivateCards::FromCombo(
    ComboId combo_id) {
  PrivateCards private_cards;
  private_cards.combo = combo_id;
  return private_cards;
}

CardMask CFRSolver::PrivateCards::mask() const {
  return ComboMask(combo);
}

CFRSolver::RangeSampler::RangeSampler(const TrainingRange& player_a_range,
                                       const TrainingRange& player_b_range)
    : player_a_range(player_a_range),
      player_b_range(player_b_range),
      compatible_player_b_weight(kComboCount, 0.0f),
      compatible_player_b_offsets(kComboCount, 0),
      compatible_player_b_counts(kComboCount, 0) {
  float total_weight = 0.0f;
  player_a_sample_weights.reserve(player_a_range.active_count);
  const size_t max_compatible_pairs =
      static_cast<size_t>(player_a_range.active_count) *
      static_cast<size_t>(player_b_range.active_count);
  compatible_player_b_combos.reserve(max_compatible_pairs);
  compatible_player_b_cumulative_weights.reserve(max_compatible_pairs);
  for (uint16_t a = 0; a < player_a_range.active_count; ++a) {
    const ComboId player_a_combo = player_a_range.active[a];
    const float player_a_weight = player_a_range.weights[player_a_combo];
    if (player_a_weight <= 0.0f) {
      player_a_sample_weights.push_back(0.0f);
      continue;
    }
    const size_t offset = compatible_player_b_combos.size();
    compatible_player_b_offsets[player_a_combo] =
        static_cast<uint32_t>(offset);
    float cumulative_player_b_weight = 0.0f;
    for (uint16_t b = 0; b < player_b_range.active_count; ++b) {
      const ComboId player_b_combo = player_b_range.active[b];
      const float player_b_weight = player_b_range.weights[player_b_combo];
      if (player_b_weight <= 0.0f ||
          (ComboMask(player_a_combo) & ComboMask(player_b_combo)) != 0) {
        continue;
      }
      cumulative_player_b_weight += player_b_weight;
      compatible_player_b_combos.push_back(player_b_combo);
      compatible_player_b_cumulative_weights.push_back(
          cumulative_player_b_weight);
    }
    compatible_player_b_weight[player_a_combo] = cumulative_player_b_weight;
    compatible_player_b_counts[player_a_combo] =
        static_cast<uint16_t>(compatible_player_b_combos.size() - offset);
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
  const float total_player_b_weight =
      compatible_player_b_weight[player_a_combo];
  const uint16_t compatible_count =
      compatible_player_b_counts[player_a_combo];
  if (total_player_b_weight <= 0.0f || compatible_count == 0) {
    throw std::logic_error("Range sampler selected an incompatible hand");
  }

  std::uniform_real_distribution<float> distribution(
      0.0f, total_player_b_weight);
  const float sample = distribution(rng);
  const size_t offset = compatible_player_b_offsets[player_a_combo];
  const auto first = compatible_player_b_cumulative_weights.begin() + offset;
  const auto last = first + compatible_count;
  auto sampled = std::upper_bound(first, last, sample);
  if (sampled == last) {
    sampled = last - 1;
  }

  const size_t sampled_index =
      offset + static_cast<size_t>(sampled - first);
  return RangeDeal(player_a_combo,
                   compatible_player_b_combos[sampled_index]);
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const GameState& state,
    int player,
    ComboId combo_id) const {
  InfoSetKey key = make_public_info_set_key(state, player);
  const ComboInfo& combo = GetComboInfo(combo_id);
  key.hand_size = 2;
  key.hand_cards[0] = EncodedCard(combo.card0);
  key.hand_cards[1] = EncodedCard(combo.card1);
  std::sort(key.hand_cards.begin(),
            key.hand_cards.begin() + key.hand_size);
  return key;
}

CFRSolver::InfoSetKey CFRSolver::make_info_set_key(
    const GameState& state,
    int player,
    const PrivateCards& private_cards) const {
  return make_info_set_key(state, player, private_cards.combo);
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    const GameState& state) const {
  PublicStateKey key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  for (int i = 0; i < key.player_contribution_size; ++i) {
    key.player_contributions[i] = RoundedContribution(state, i);
  }

  key.board_size = std::min(static_cast<int>(state.board_cards.size()),
                            InfoSetKey::kMaxCards);
  for (int i = 0; i < key.board_size; ++i) {
    key.board_cards[i] = EncodedCard(state.board_cards[i]);
  }
  std::sort(key.board_cards.begin(),
            key.board_cards.begin() + key.board_size);

  const int history_value_count = state.history.size() * 3;
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
  for (const GameAction& action : state.history) {
    add_history_value(action.player);
    add_history_value(static_cast<int>(action.kind));
    add_history_value(action.amount);
  }

  return key;
}

CFRSolver::InfoSetKey CFRSolver::make_public_info_set_key(
    const GameState& state,
    int player) const {
  InfoSetKey key;
  key.player = player;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  for (int i = 0; i < key.player_contribution_size; ++i) {
    key.player_contributions[i] = RoundedContribution(state, i);
  }

  key.board_size = std::min(static_cast<int>(state.board_cards.size()),
                            InfoSetKey::kMaxCards);
  for (int i = 0; i < key.board_size; ++i) {
    key.board_cards[i] = EncodedCard(state.board_cards[i]);
  }
  std::sort(key.board_cards.begin(),
            key.board_cards.begin() + key.board_size);

  const int history_value_count = state.history.size() * 3;
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
  for (const GameAction& action : state.history) {
    add_history_value(action.player);
    add_history_value(static_cast<int>(action.kind));
    add_history_value(action.amount);
  }

  return key;
}

void CFRSolver::initialize_info_set_actions(
    InfoSetData& info_set,
    const std::vector<int>& legal_action_ids) {
  info_set.action_offset = static_cast<uint32_t>(action_ids_.size());
  info_set.action_count = static_cast<uint16_t>(legal_action_ids.size());
  const size_t required_action_capacity =
      action_ids_.size() + legal_action_ids.size();
  if (required_action_capacity > action_ids_.capacity()) {
    const size_t new_capacity =
        std::max(required_action_capacity,
                 action_ids_.empty() ? size_t{4096}
                                     : action_ids_.capacity() * 2);
    action_ids_.reserve(new_capacity);
    cumulative_regrets_.reserve(new_capacity);
    cumulative_strategies_.reserve(new_capacity);
  }
  for (int action_id : legal_action_ids) {
    action_ids_.push_back(action_id);
    cumulative_regrets_.push_back(0.0f);
    cumulative_strategies_.push_back(0.0f);
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
  }
}

int CFRSolver::get_or_create_compact_info_set_id(
    uint32_t public_state_id,
    const GameState& state,
    int player,
    ComboId combo_id,
    const std::vector<int>& legal_action_ids) {
  CompactInfoSetKey compact_key;
  compact_key.public_state_id = public_state_id;
  compact_key.private_combo = combo_id;
  compact_key.player = static_cast<uint8_t>(player);
  auto existing_compact = compact_info_set_ids_.find(compact_key);
  if (existing_compact != compact_info_set_ids_.end()) {
    return existing_compact->second;
  }

  InfoSetKey full_key = make_info_set_key(state, player, combo_id);
  auto existing_legacy = info_set_ids_.find(full_key);
  if (existing_legacy != info_set_ids_.end()) {
    compact_info_set_ids_.emplace(compact_key, existing_legacy->second);
    return existing_legacy->second;
  }

  if (info_sets_.size() == info_sets_.capacity()) {
    const size_t new_capacity =
        info_sets_.empty() ? 1024 : info_sets_.capacity() * 2;
    info_sets_.reserve(new_capacity);
    compact_info_set_ids_.reserve(new_capacity);
  }

  const int id = static_cast<int>(info_sets_.size());
  InfoSetData data;
  data.key = std::move(full_key);
  initialize_info_set_actions(data, legal_action_ids);
  info_sets_.push_back(std::move(data));
  compact_info_set_ids_.emplace(compact_key, id);
  legacy_info_set_index_valid_ = false;
  return id;
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
    compact_info_set_ids_.reserve(new_capacity);
  }

  const int id = static_cast<int>(info_sets_.size());
  InfoSetData data;
  data.key = key;
  initialize_info_set_actions(data, legal_action_ids);
  info_sets_.push_back(std::move(data));
  info_set_ids_.emplace(info_sets_.back().key, id);
  return id;
}

void CFRSolver::ensure_legacy_info_set_index() {
  if (!legacy_info_set_index_valid_ ||
      info_set_ids_.size() != info_sets_.size()) {
    rebuild_legacy_info_set_index();
  }
}

void CFRSolver::rebuild_legacy_info_set_index() {
  info_set_ids_.clear();
  info_set_ids_.reserve(info_sets_.size());
  for (size_t i = 0; i < info_sets_.size(); ++i) {
    info_set_ids_.emplace(info_sets_[i].key, static_cast<int>(i));
  }
  legacy_info_set_index_valid_ = true;
}

CFRSolver::StrategyTablesView CFRSolver::strategy_tables_view() const {
  return {&info_set_ids_,
          &info_sets_,
          &action_ids_,
          &cumulative_regrets_,
          &cumulative_strategies_,
          &loaded_strategy_};
}

const absl::flat_hash_map<CFRSolver::InfoSetKey, int,
                          CFRSolver::InfoSetKeyHash>&
CFRSolver::strategy_info_set_ids() const {
  return strategy_tables_view_ != nullptr ? *strategy_tables_view_->info_set_ids
                                          : info_set_ids_;
}

const std::vector<CFRSolver::InfoSetData>& CFRSolver::strategy_info_sets()
    const {
  return strategy_tables_view_ != nullptr ? *strategy_tables_view_->info_sets
                                          : info_sets_;
}

const std::vector<int>& CFRSolver::strategy_action_ids() const {
  return strategy_tables_view_ != nullptr ? *strategy_tables_view_->action_ids
                                          : action_ids_;
}

const std::vector<float>& CFRSolver::strategy_cumulative_regrets() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->cumulative_regrets
             : cumulative_regrets_;
}

const std::vector<float>& CFRSolver::strategy_cumulative_strategies() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->cumulative_strategies
             : cumulative_strategies_;
}

const Strategy& CFRSolver::strategy_loaded_strategy() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->loaded_strategy
             : loaded_strategy_;
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
    for (uint16_t i = 0; i < info_set.action_count; ++i) {
      const size_t action_offset = info_set.action_offset + i;
      if (action_ids_[action_offset] == action_id) {
        return cumulative_regrets_[action_offset];
      }
    }
    return 0.0;
  }
  return 0.0;
}

void CFRSolver::run(int iterations, ComboId player_a_hand,
                    ComboId player_b_hand) {
  if (iterations <= 0) {
    return;
  }

  GameTree::Node& root = get_or_build_root();
  const int max_depth = config_.max_depth;
  TraversalScratch scratch;
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  for (int i = 0; i < iterations; ++i) {
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    cumulative_root_utility_ += cfr_with_ranges(
        root, player_a_cards, player_b_cards, reach_probabilities,
        iterations_run_, 0, max_depth, scratch, std::nullopt, std::nullopt);
    ++iterations_run_;
  }
}

void CFRSolver::run(int iterations, const Hand& player_a_hand,
                    const Hand& player_b_hand) {
  run(iterations, ProtoHandToComboId(player_a_hand),
      ProtoHandToComboId(player_b_hand));
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

    const int max_depth = config_.max_depth;
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
                      ComboId player_a_hand,
                      ComboId player_b_hand,
                      std::array<double, 2>& reach_probabilities,
                      int iteration,
                      int depth,
                      int max_depth) {
  TraversalScratch scratch;
  return cfr_with_ranges(node, player_a_hand, player_b_hand,
                         reach_probabilities, iteration, depth, max_depth,
                         scratch, std::nullopt, std::nullopt);
}

double CFRSolver::cfr(GameTree::Node& node,
                      const Hand& player_a_hand,
                      const Hand& player_b_hand,
                      std::array<double, 2>& reach_probabilities,
                      int iteration,
                      int depth,
                      int max_depth) {
  return cfr(node, ProtoHandToComboId(player_a_hand),
             ProtoHandToComboId(player_b_hand), reach_probabilities,
             iteration, depth, max_depth);
}

double CFRSolver::cfr_with_ranges(
    GameTree::Node& node,
    ComboId player_a_hand,
    ComboId player_b_hand,
    std::array<double, 2>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  return cfr_with_ranges(
      node, PrivateCards::FromCombo(player_a_hand),
      PrivateCards::FromCombo(player_b_hand), reach_probabilities, iteration,
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
    if (node.state.folded_player >= 0) {
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
        node.state, player_a_cards.combo, player_b_cards.combo,
        player_a_range,
        player_b_range);
    return continuation_value_provider_->value(*game_tree_, context);
  }
  
  // Get the player to act at this node
  int player = node.player_to_act;
  
  // Get the player's hand
  const PrivateCards& player_cards =
      (player == 0) ? player_a_cards : player_b_cards;
  
  EnsureLegalActionIds(node);
  const uint32_t public_state_id = static_cast<uint32_t>(node.id);
  const int info_set_id =
      get_or_create_compact_info_set_id(
          public_state_id, node.state, player, player_cards.combo,
          node.legal_action_ids);
  ActionChoices action_choices;
  action_choices.reserve(node.legal_actions.size());
  {
    InfoSetData& info_set = info_sets_[info_set_id];
    const size_t action_offset = info_set.action_offset;
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
          std::max(
              0.0,
              static_cast<double>(
                  cumulative_regrets_[action_offset + action_index]));
    }

    if (sum_positive_regrets > 0.0) {
      for (size_t action_index = 0; action_index < action_choices.size();
           ++action_index) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        ActionChoice& choice = action_choices[action_index];
        choice.probability =
            std::max(
                0.0,
                static_cast<double>(
                    cumulative_regrets_[action_offset + action_index])) /
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
    condition_ranges_for_actions(player_a_range->get(), node.state,
                                 public_state_id, player, action_choices,
                                 conditioned_player_ranges);
  } else if (condition_player_b_range) {
    condition_ranges_for_actions(player_b_range->get(), node.state,
                                 public_state_id, player, action_choices,
                                 conditioned_player_ranges);
  }

  // For each action, recursively call CFR and compute the expected value
  for (size_t choice_index = 0; choice_index < action_choices.size();
       ++choice_index) {
    ActionChoice& choice = action_choices[choice_index];
    const GameAction& action = choice.action.get();
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
    switch (node.state.street) {
      case StreetKind::kPreflop:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.preflop_updates);
        break;
      case StreetKind::kFlop:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.flop_updates);
        break;
      case StreetKind::kTurn:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.turn_updates);
        break;
      case StreetKind::kRiver:
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.river_updates);
        break;
    }

    // Compute the counterfactual reach probability of the opponent
    double opponent_reach_prob = reach_probabilities[1 - player];
    InfoSetData& info_set = info_sets_[info_set_id];
    const size_t action_offset = info_set.action_offset;
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
      float& cumulative_regret =
          cumulative_regrets_[action_offset + action_index];
      POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
      cumulative_regret =
          static_cast<float>(std::max(0.0, cumulative_regret + regret));
    }
    
    if (!config_.regret_only_training) {
      // CFR+ commonly weights later average-strategy samples more heavily.
      update_strategy(info_set_id, action_choices,
                      reach_probabilities[player] * (iteration + 1));
    }
  }
  
  return node_value;
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
    const GameState& state,
    int player,
    const PrivateCards& private_cards,
    const std::vector<GameAction>& legal_actions,
    int action_id) {
  if (legal_actions.empty()) {
    return 0.0;
  }
  const double uniform_probability = 1.0 / legal_actions.size();

  InfoSetKey key = make_info_set_key(state, player, private_cards);
  const auto& info_set_ids = strategy_info_set_ids();
  auto existing_info_set = info_set_ids.find(key);
  if (existing_info_set != info_set_ids.end()) {
    return average_strategy_action_probability(
        strategy_info_sets()[existing_info_set->second], legal_actions,
        action_id, uniform_probability);
  }

  const Strategy& loaded_strategy = strategy_loaded_strategy();
  if (!loaded_strategy.empty()) {
    return StrategyActionProbability(
        loaded_strategy, info_set_key_to_string(key), legal_actions,
        action_id);
  }
  return uniform_probability;
}

double CFRSolver::average_strategy_action_probability(
    const InfoSetData& info_set,
    const std::vector<GameAction>& legal_actions,
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
    const GameState& state,
    int player,
    const PrivateCards& private_cards,
    const std::vector<GameAction>& legal_actions,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(legal_actions.size(), 0.0);
  if (legal_actions.empty()) {
    return;
  }

  const double uniform_probability = 1.0 / legal_actions.size();
  InfoSetKey key = make_info_set_key(state, player, private_cards);
  const auto& info_set_ids = strategy_info_set_ids();
  auto existing_info_set = info_set_ids.find(key);
  if (existing_info_set != info_set_ids.end()) {
    average_strategy_probabilities(
        strategy_info_sets()[existing_info_set->second], legal_actions,
        uniform_probability, probabilities);
    return;
  }

  const Strategy& loaded_strategy = strategy_loaded_strategy();
  if (loaded_strategy.empty()) {
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }

  double probability_sum = 0.0;
  const std::string info_set_key = info_set_key_to_string(key);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    probabilities[i] = loaded_strategy.get_action_probability(
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
    GameTree::Node& node,
    int player,
    const PrivateCards& private_cards,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(node.legal_actions.size(), 0.0);
  if (node.legal_actions.empty()) {
    return;
  }

  const double uniform_probability = 1.0 / node.legal_actions.size();
  CompactInfoSetKey compact_key;
  compact_key.public_state_id = static_cast<uint32_t>(node.id);
  compact_key.private_combo = private_cards.combo;
  compact_key.player = static_cast<uint8_t>(player);
  auto existing_compact = compact_info_set_ids_.find(compact_key);
  if (existing_compact != compact_info_set_ids_.end()) {
    average_strategy_probabilities(
        strategy_info_sets()[existing_compact->second], node.legal_actions,
        uniform_probability, probabilities);
    return;
  }

  average_strategy_probabilities(
      node.state, player, private_cards, node.legal_actions, probabilities);
}

void CFRSolver::average_strategy_probabilities(
    const InfoSetData& info_set,
    const std::vector<GameAction>& legal_actions,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(legal_actions.size(), 0.0);
  double probability_sum = 0.0;
  const size_t action_offset = info_set.action_offset;
  const std::vector<int>& action_ids = strategy_action_ids();
  const std::vector<float>& cumulative_regrets =
      strategy_cumulative_regrets();
  const std::vector<float>& cumulative_strategies =
      strategy_cumulative_strategies();

  const bool aligned_action_ids =
      legal_actions.size() == info_set.action_count &&
      std::equal(legal_actions.begin(), legal_actions.end(),
                 action_ids.begin() + action_offset,
                 [](const GameAction& legal_action, int action_id) {
                   return ActionKey(legal_action) == action_id;
                 });
  if (aligned_action_ids) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      const size_t table_index = action_offset + i;
      probabilities[i] =
          config_.regret_only_training
              ? std::max(0.0,
                         static_cast<double>(
                             cumulative_regrets[table_index]))
              : static_cast<double>(cumulative_strategies[table_index]);
      probability_sum += probabilities[i];
    }
  } else {
    for (size_t legal_action_index = 0;
         legal_action_index < legal_actions.size(); ++legal_action_index) {
      const int legal_action_id = ActionKey(legal_actions[legal_action_index]);
      for (uint16_t action_index = 0; action_index < info_set.action_count;
           ++action_index) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const size_t table_index = action_offset + action_index;
        if (action_ids[table_index] == legal_action_id) {
          probabilities[legal_action_index] =
              config_.regret_only_training
                  ? std::max(
                        0.0,
                        static_cast<double>(cumulative_regrets[table_index]))
                  : static_cast<double>(cumulative_strategies[table_index]);
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
    const GameState& state,
    uint32_t public_state_id,
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
  const CardMask board_mask = state.board_mask;
  absl::InlinedVector<double, 8> positive_regrets(
      action_choices.size(), 0.0);
  for (size_t i = 0; i < range.size(); ++i) {
    if (range.weight(i) <= 0.0 || (range.mask(i) & board_mask) != 0) {
      continue;
    }

    double positive_regret_sum = 0.0;
    std::optional<int> info_set_id;
    CompactInfoSetKey compact_key;
    compact_key.public_state_id = public_state_id;
    compact_key.private_combo = range.combo(i);
    compact_key.player = static_cast<uint8_t>(player);
    auto existing_compact = compact_info_set_ids_.find(compact_key);
    if (existing_compact != compact_info_set_ids_.end()) {
      info_set_id = existing_compact->second;
    } else if (compact_info_set_ids_.empty()) {
      InfoSetKey key = make_info_set_key(state, player, range.combo(i));
      auto existing_legacy = info_set_ids_.find(key);
      if (existing_legacy != info_set_ids_.end()) {
        info_set_id = existing_legacy->second;
      }
    }

    if (info_set_id.has_value()) {
      const InfoSetData& info_set = info_sets_[*info_set_id];
      const size_t table_offset = info_set.action_offset;
      std::fill(positive_regrets.begin(), positive_regrets.end(), 0.0);
      const size_t action_count =
          std::min(action_choices.size(),
                   static_cast<size_t>(info_set.action_count));
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        const size_t table_index = table_offset + action_index;
        if (action_ids_[table_index] !=
            action_choices[action_index].action_id) {
          continue;
        }

        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const double positive_regret =
            std::max(
                0.0,
                static_cast<double>(cumulative_regrets_[table_index]));
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
    const GameState& state,
    ComboId player_a_hand,
    ComboId player_b_hand,
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
    const size_t action_offset = info_set.action_offset;
    for (uint16_t action_index = 0; action_index < info_set.action_count;
         ++action_index) {
      const size_t table_index = action_offset + action_index;
      sum += config_.regret_only_training
                 ? std::max(0.0,
                            static_cast<double>(
                                cumulative_regrets_[table_index]))
                 : static_cast<double>(cumulative_strategies_[table_index]);
    }

    Strategy::ActionProbabilities normalized_strategy;
    normalized_strategy.reserve(info_set.action_count);
    if (sum > 0.0) {
      for (uint16_t action_index = 0; action_index < info_set.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        const double strategy_weight =
            config_.regret_only_training
                ? std::max(0.0,
                           static_cast<double>(
                               cumulative_regrets_[table_index]))
                : static_cast<double>(cumulative_strategies_[table_index]);
        normalized_strategy[action_ids_[table_index]] = strategy_weight / sum;
      }
    } else if (info_set.action_count > 0) {
      double uniform_prob = 1.0 / info_set.action_count;
      for (uint16_t action_index = 0; action_index < info_set.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        normalized_strategy[action_ids_[table_index]] = uniform_prob;
      }
    }

    equilibrium_strategy.update(info_set_key_to_string(info_set.key),
                                normalized_strategy);
  }

  return equilibrium_strategy;
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  return evaluate_strategy_node(get_or_build_root(),
                                PrivateCards::FromCombo(player_a_hand),
                                PrivateCards::FromCombo(player_b_hand));
}

double CFRSolver::evaluate_strategy(const Hand& player_a_hand,
                                    const Hand& player_b_hand) {
  return evaluate_strategy(ProtoHandToComboId(player_a_hand),
                           ProtoHandToComboId(player_b_hand));
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

  SolverConfig config = config_;
  std::shared_ptr<TerminalUtilityCache> utility_cache = utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
  ensure_legacy_info_set_index();
  const StrategyTablesView strategy_tables = strategy_tables_view();
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, range_sampler,
                                       &strategy_tables, utility_cache,
                                       continuation_value_provider,
                                       shard_samples, seed]() mutable {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.strategy_tables_view_ = &strategy_tables;
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
  average_strategy_probabilities(node, player, player_cards, probabilities);

  double value = 0.0;
  for (size_t action_index = 0; action_index < node.legal_actions.size();
       ++action_index) {
    const GameAction& action = node.legal_actions[action_index];
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
                                      const PrivateCards& player_a_cards,
                                      const PrivateCards& player_b_cards,
                                      int best_response_player) {
  if (node.is_terminal) {
    double player_a_value =
        utility(node.state, player_a_cards, player_b_cards);
    return best_response_player == 0 ? player_a_value : -player_a_value;
  }
  if (node.is_chance_node) {
    return SampleChanceValue(
        *game_tree_, node, player_a_cards.mask() | player_b_cards.mask(),
        ChanceSamples(config_), rng_, nullptr, [&](GameTree::Node& child_node) {
          return best_response_value(child_node, player_a_cards, player_b_cards,
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

  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (const GameAction& action : node.legal_actions) {
      int action_id = ActionKey(action);
      GameTree::Node& child_node =
          CachedActionChild(*game_tree_, node, action, action_id,
                            nullptr);
      value = std::max(value, best_response_value(
                                  child_node, player_a_cards, player_b_cards,
                                  best_response_player));
    }
    return value;
  }

  StrategyProbabilities probabilities;
  average_strategy_probabilities(
      node, player, player_cards, probabilities);
  double value = 0.0;
  for (size_t action_index = 0; action_index < node.legal_actions.size();
       ++action_index) {
    const GameAction& action = node.legal_actions[action_index];
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);
    value += probabilities[action_index] *
             best_response_value(child_node, player_a_cards, player_b_cards,
                                 best_response_player);
  }
  return value;
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
          PrivateCards::FromCombo(opponent_hands.combo(i));
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
      const PrivateCards sampled_opponent = PrivateCards::FromCombo(
          opponent_hands.source_range().combos[sampled_opponent_index]);
      std::vector<CardId> cards =
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
    for (const GameAction& action : node.legal_actions) {
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
  for (size_t action_index = 0; action_index < node.legal_actions.size();
       ++action_index) {
    const GameAction& action = node.legal_actions[action_index];
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);

    WeightedHandRangeView child_opponents;
    child_opponents.reset_to_filtered(opponent_hands.source_range());
    child_opponents.reserve(opponent_hands.size());
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      const PrivateCards opponent_cards =
          PrivateCards::FromCombo(opponent_hands.combo(i));
      StrategyProbabilities probabilities;
      average_strategy_probabilities(node, player, opponent_cards,
                                     probabilities);
      double probability = probabilities[action_index];
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

  SolverConfig config = config_;
  std::shared_ptr<TerminalUtilityCache> utility_cache = utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
  ensure_legacy_info_set_index();
  const StrategyTablesView strategy_tables = strategy_tables_view();
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, &best_response_hands,
                                       &opponent_hands, &strategy_tables,
                                       utility_cache, continuation_value_provider,
                                       shard_samples, seed, best_response_player]() {
      CFRSolver worker(config, utility_cache, continuation_value_provider);
      worker.strategy_tables_view_ = &strategy_tables;
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
    const PrivateCards best_response_cards = PrivateCards::FromCombo(
        best_response_hands.combos[best_response_hand_distribution(rng_)]);
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
    std::vector<CardId> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), rng_);
    ComboId player_a_hand = CardsToComboId(deck.back(), deck[deck.size() - 2]);
    deck.pop_back();
    deck.pop_back();
    ComboId player_b_hand = CardsToComboId(deck.back(), deck[deck.size() - 2]);
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

double CFRSolver::calculate_exploitability(ComboId player_a_hand,
                                           ComboId player_b_hand) {
  GameTree::Node& root = get_or_build_root();
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  double strategy_player_a_value =
      evaluate_strategy_node(root, player_a_cards, player_b_cards);
  double player_a_gap =
      best_response_value(root, player_a_cards, player_b_cards, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_cards, player_b_cards, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

double CFRSolver::calculate_exploitability(const Hand& player_a_hand,
                                           const Hand& player_b_hand) {
  return calculate_exploitability(ProtoHandToComboId(player_a_hand),
                                  ProtoHandToComboId(player_b_hand));
}

GameAction CFRSolver::get_best_response_action(GameTree::Node& node,
                                               ComboId player_a_hand,
                                               ComboId player_b_hand,
                                               int best_response_player) {
  GameAction no_action;
  no_action.kind = ActionKind::kNoAction;
  if (node.is_terminal || node.is_chance_node ||
      node.legal_actions.empty() || node.player_to_act != best_response_player) {
    return no_action;
  }

  double best_value = -std::numeric_limits<double>::infinity();
  GameAction best_action = no_action;
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  for (const GameAction& action : node.legal_actions) {
    int action_id = ActionKey(action);
    GameTree::Node& child_node =
        CachedActionChild(*game_tree_, node, action, action_id,
                          nullptr);
    double value = best_response_value(child_node, player_a_cards,
                                       player_b_cards, best_response_player);
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
  *snapshot.mutable_config() = SolverConfigToProto(config_);
  snapshot.set_iterations_run(iterations_run_);
  snapshot.set_abstraction_version(kInfoSetKeyVersion);

  for (const std::string& info_set_key : equilibrium_strategy.get_info_sets()) {
    StrategyInfoSetSnapshot& info_set = *snapshot.add_info_sets();
    info_set.set_info_set_key(info_set_key);

    Strategy::ActionProbabilities action_probs =
        equilibrium_strategy.get_strategy(info_set_key);
    for (const auto& action_prob : action_probs) {
      StrategyActionSnapshot& action = *info_set.add_actions();
      *action.mutable_action() = ProtoActionFromKey(action_prob.first);
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
  action_ids_.clear();
  cumulative_regrets_.clear();
  cumulative_strategies_.clear();
  compact_info_set_ids_.clear();
  legacy_info_set_index_valid_ = true;

  for (const StrategyInfoSetSnapshot& info_set : snapshot.info_sets()) {
    Strategy::ActionProbabilities action_probs;
    for (const StrategyActionSnapshot& action : info_set.actions()) {
      if (action.has_action()) {
        const int action_id = ProtoActionKey(action.action());
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

double CFRSolver::utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_cards.size() + 2 < 5) {
    return 0.0;
  }

  return utility_cache_->get_or_compute(
      state, player_a_cards.combo, player_b_cards.combo, [&]() {
        return game_tree_->get_utility(
            state, player_a_cards.combo, player_b_cards.combo);
      });
}

double CFRSolver::uncached_utility(const GameState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_cards.size() + 2 < 5) {
    return 0.0;
  }

  return game_tree_->get_utility(
      state, player_a_cards.combo, player_b_cards.combo);
}

void CFRSolver::update_strategy(int info_set_id,
                                const ActionChoices& choices,
                                double reach_prob) {
  InfoSetData& info_set = info_sets_[info_set_id];
  const size_t action_offset = info_set.action_offset;
  for (size_t action_index = 0; action_index < choices.size();
       ++action_index) {
    POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
    cumulative_strategies_[action_offset + action_index] += static_cast<float>(
        reach_prob * choices[action_index].probability);
  }
}

} // namespace poker
