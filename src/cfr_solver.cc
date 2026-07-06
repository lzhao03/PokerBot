#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "src/card_utils.h"
#include "src/continuation_value.h"
#include "src/hand_range.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kParallelBestResponseSampleThreshold = 32;

// Atomically add `delta` to `*target` using a CAS loop.
// Safe to call from multiple threads concurrently. Relaxed ordering is
// sufficient: CFR+ convergence is robust to the bounded noise from concurrent
// stale reads (equivalent to async-regret MCCFR).
inline void AtomicFloatAdd(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float new_val;
    std::memcpy(&new_val, &old_bits, sizeof(float));
    new_val += delta;
    std::memcpy(&new_bits, &new_val, sizeof(float));
  } while (!__atomic_compare_exchange_n(
      reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
      /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

// Atomically apply CFR+ regret update: new = max(0, old + delta).
// Uses the same CAS loop as AtomicFloatAdd but clips at zero.
inline void AtomicCFRPlusRegretUpdate(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float old_val;
    std::memcpy(&old_val, &old_bits, sizeof(float));
    float new_val = std::max(0.0f, old_val + delta);
    std::memcpy(&new_bits, &new_val, sizeof(float));
  } while (!__atomic_compare_exchange_n(
      reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
      /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

// Relaxed atomic load of a float (naturally-aligned 32-bit load is atomic
// on all supported architectures, but we use __atomic_load_n for clarity).
inline float AtomicFloatLoad(const float* src) {
  int32_t bits = __atomic_load_n(reinterpret_cast<const int32_t*>(src),
                                 __ATOMIC_RELAXED);
  float val;
  std::memcpy(&val, &bits, sizeof(float));
  return val;
}

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

double TotalWeight(const WeightedHandRangeView& hands) {
  double total = 0.0;
  for (size_t i = 0; i < hands.size(); ++i) {
    total += hands.weight(i);
  }
  return total;
}

size_t ScratchDepthReserve(const SolverConfig& config, int max_depth) {
  if (max_depth > 0) {
    return static_cast<size_t>(max_depth) + 2;
  }
  const int stack_size = std::max(0, config.starting_stack_size);
  return std::max<size_t>(32, static_cast<size_t>(stack_size) + 12);
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
                               CardMask board_mask,
                               TrainingRangeView& compatible_hands) {
  if (hands.empty()) {
    compatible_hands.clear();
    return;
  }

  compatible_hands.reset_to_filtered();
  for (size_t i = 0; i < hands.size(); ++i) {
    if (hands.weight(i) > 0.0 && (hands.mask(i) & board_mask) == 0) {
      compatible_hands.add(hands.combo(i), hands.weight(i));
    }
  }
}

void PublicCompatibleRangeInto(const TrainingRangeView& hands,
                               const GameState& state,
                               TrainingRangeView& compatible_hands) {
  PublicCompatibleRangeInto(hands, state.board_mask, compatible_hands);
}

int ChanceCardsKey(absl::Span<const CardId> cards) {
  int encoded[5];
  const int n = static_cast<int>(cards.size());
  for (int i = 0; i < n; ++i) {
    encoded[i] = EncodedCard(cards[i]);
  }
  // Chance samples are small (max 5), so a simple sort network is cheaper than
  // calling std::sort on a dynamically-sized vector.
  if (n > 1) {
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (encoded[j] < encoded[i]) {
          std::swap(encoded[i], encoded[j]);
        }
      }
    }
  }

  int key = n;
  for (int i = 0; i < n; ++i) {
    key = key * 128 + encoded[i];
  }
  return -1 - key;
}

uint64_t PublicChanceChildKey(uint32_t parent_id, int child_key) {
  return (static_cast<uint64_t>(parent_id) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(child_key));
}

// Diagnostic legacy-tree helper. Returns nullptr if the child does not already
// exist AND either:
//   - frozen is true (parallel phase: no new nodes allowed), or
//   - max_public_states > 0 and the diagnostic tree is already at or above that
//     public-state cap.
// In the cap-reached case the sample is simply skipped (the caller averages
// over the remaining samples that do hit the tree).
GameTree::Node* CachedChanceChildOrNull(GameTree& game_tree,
                                        GameTree::Node& node,
                                        absl::Span<const CardId> cards,
                                        int64_t* created_nodes,
                                        bool frozen = false,
                                        int max_public_states = 0) {
  const int child_key = ChanceCardsKey(cards);
  // Chance-node children are stored in a side-table on GameTree, not in the
  // node's inline action array (which is reserved for player-action nodes).
  GameTree::NodeId existing = game_tree.find_chance_child(node.id, child_key);
  if (existing != GameTree::kInvalidNodeId) {
    return &game_tree.node(existing);
  }
  if (frozen) {
    return nullptr;
  }
  if (max_public_states > 0 &&
      static_cast<int>(game_tree.node_count()) >= max_public_states) {
    return nullptr;
  }
  if (created_nodes != nullptr) {
    ++*created_nodes;
  }
  return &game_tree.create_chance_child_node(node, child_key, cards);
}

// Diagnostic legacy-tree helper. Returns nullptr if the child does not already
// exist AND either frozen is true or the public-state cap has already been
// reached.
GameTree::Node* CachedActionChildOrNull(GameTree& game_tree,
                                        GameTree::Node& node,
                                        int action_index,
                                        int64_t* created_nodes,
                                        bool frozen = false,
                                        int max_public_states = 0) {
  GameTree::NodeId existing = node.child_for_action_index(action_index);
  if (existing != GameTree::kInvalidNodeId) {
    return &game_tree.node(existing);
  }
  if (frozen) {
    return nullptr;
  }
  if (max_public_states > 0 &&
      static_cast<int>(game_tree.node_count()) >= max_public_states) {
    return nullptr;
  }
  if (created_nodes != nullptr) {
    ++*created_nodes;
  }
  return &game_tree.create_child_node(node, action_index);
}

int RoundedContribution(const GameState& state, int player) {
  return state.player_contribution[player];
}

void HashCombine(size_t& seed, int value) {
  seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

void HashCombine(size_t& seed, uint64_t value) {
  seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
}

template <size_t N>
void HashArray(size_t& seed, const std::array<int, N>& values) {
  for (int value : values) {
    HashCombine(seed, value);
  }
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
                         EvaluateChild evaluate_child,
                         bool frozen = false,
                         int max_public_states = 0) {
  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const auto cards =
        SampleStreetCards(node.state, known_private_cards, rng);
    GameTree::Node* child_node =
        CachedChanceChildOrNull(game_tree, node, cards, created_nodes, frozen,
                                max_public_states);
    if (child_node == nullptr) {
      continue;  // tree cap or frozen; skip sample
    }
    value += evaluate_child(*child_node);
    ++evaluated;
  }
  return evaluated > 0 ? value / evaluated : 0.0;
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

bool CFRSolver::BettingHistoryKey::operator==(
    const BettingHistoryKey& other) const {
  return street == other.street && pot == other.pot &&
         stack_a == other.stack_a && stack_b == other.stack_b &&
         all_in == other.all_in && folded_player == other.folded_player &&
         player_to_act == other.player_to_act &&
         player_contribution_size == other.player_contribution_size &&
         player_contributions == other.player_contributions &&
         history_size == other.history_size &&
         history_values == other.history_values &&
         history_overflow == other.history_overflow;
}

size_t CFRSolver::BettingHistoryKeyHash::operator()(
    const BettingHistoryKey& key) const {
  size_t seed = 0;
  HashCombine(seed, key.street);
  HashCombine(seed, key.pot);
  HashCombine(seed, key.stack_a);
  HashCombine(seed, key.stack_b);
  HashCombine(seed, key.all_in);
  HashCombine(seed, key.folded_player);
  HashCombine(seed, key.player_to_act);
  HashCombine(seed, key.player_contribution_size);
  HashArray(seed, key.player_contributions);
  HashCombine(seed, key.history_size);
  HashArray(seed, key.history_values);
  for (int value : key.history_overflow) {
    HashCombine(seed, value);
  }
  return seed;
}

bool CFRSolver::PublicStateKey::operator==(
    const PublicStateKey& other) const {
  return betting_history_id == other.betting_history_id &&
         public_bucket == other.public_bucket;
}

size_t CFRSolver::PublicStateKeyHash::operator()(
    const PublicStateKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.betting_history_id));
  HashCombine(seed, key.public_bucket);
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
    game_tree_(std::make_shared<GameTree>(config)),
    rng_(12345),
    cumulative_root_utility_(0.0),
    utility_cache_(std::move(utility_cache)),
    continuation_value_provider_(std::move(continuation_value_provider)) {
  // Pre-allocate strategy table storage when limits are known upfront.
  // This gives fully deterministic peak memory: no reallocation after init.
  if (config_.max_info_sets > 0) {
    // avg ~4 actions per info set (between 2 and kMaxActionsPerNode).
    constexpr int kAvgActionsPerInfoSet = 4;
    const size_t info_set_cap = static_cast<size_t>(config_.max_info_sets);
    const size_t action_cap = info_set_cap * kAvgActionsPerInfoSet;
    action_ids_.reserve(action_cap);
    cumulative_regrets_.reserve(action_cap);
    cumulative_strategies_.reserve(action_cap);
  }
  if (config_.max_public_states > 0) {
    public_state_rows_.reserve(static_cast<size_t>(config_.max_public_states));
  }
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
      player_a_cumulative_weights.push_back(total_weight);
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
    player_a_cumulative_weights.push_back(total_weight);
  }

  if (total_weight <= 0.0f) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }
  total_player_a_weight = total_weight;
}

CFRSolver::RangeDeal CFRSolver::RangeSampler::sample(std::mt19937& rng) const {
  std::uniform_real_distribution<float> player_a_distribution(
      0.0f, total_player_a_weight);
  const float player_a_sample = player_a_distribution(rng);
  auto player_a_sampled = std::upper_bound(
      player_a_cumulative_weights.begin(), player_a_cumulative_weights.end(),
      player_a_sample);
  if (player_a_sampled == player_a_cumulative_weights.end()) {
    player_a_sampled = player_a_cumulative_weights.end() - 1;
  }
  const size_t player_a_active_index = static_cast<size_t>(
      player_a_sampled - player_a_cumulative_weights.begin());
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

CFRSolver::BettingHistoryKey CFRSolver::make_betting_history_key(
    const GameState& state) const {
  BettingHistoryKey key;
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

  const int history_value_count = state.history.size() * 3;
  if (history_value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(history_value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  auto add_history_value = [&key](int value) {
    if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
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

CFRSolver::BettingHistoryKey CFRSolver::make_betting_history_key(
    const CompactPublicState& state) const {
  BettingHistoryKey key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  key.player_contributions = state.player_contribution;

  const int history_value_count = state.history_size * 3;
  if (history_value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(history_value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  auto add_history_value = [&key](int value) {
    if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
      key.history_values[key.history_size] = value;
    } else {
      key.history_overflow.push_back(value);
    }
    ++key.history_size;
  };
  for (uint16_t i = 0; i < state.history_size; ++i) {
    const CompactAction action = compact_history_action(state, i);
    add_history_value(action.player);
    add_history_value(static_cast<int>(action.kind));
    add_history_value(action.amount);
  }

  return key;
}

CFRSolver::BettingHistoryRow CFRSolver::make_betting_history_row(
    const GameState& state) const {
  BettingHistoryRow row;
  row.street = static_cast<int>(state.street);
  row.pot = state.pot;
  row.stack = {state.stack[0], state.stack[1]};
  row.all_in = state.all_in ? 1 : 0;
  row.folded_player = state.folded_player;
  row.player_to_act = state.player_to_act;
  row.player_contributions = {
      RoundedContribution(state, 0),
      RoundedContribution(state, 1),
  };
  return row;
}

CFRSolver::BettingHistoryRow CFRSolver::make_betting_history_row(
    const CompactPublicState& state) const {
  BettingHistoryRow row;
  row.street = static_cast<int>(state.street);
  row.pot = state.pot;
  row.stack = state.stack;
  row.all_in = state.all_in ? 1 : 0;
  row.folded_player = state.folded_player;
  row.player_to_act = state.player_to_act;
  row.player_contributions = state.player_contribution;
  return row;
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    uint32_t betting_history_id,
    const GameState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

CFRSolver::PublicStateKey CFRSolver::make_public_state_key(
    uint32_t betting_history_id,
    const CompactPublicState& state) const {
  return {betting_history_id, card_abstraction_.public_bucket(state)};
}

CompactPublicState
CFRSolver::compact_public_state_from_game_state(
    uint32_t betting_history_id,
    const GameState& state) {
  CompactPublicState compact;
  compact.stack = {state.stack[0], state.stack[1]};
  compact.pot = state.pot;
  if (state.board_cards.size() > compact.board_cards.size()) {
    throw std::logic_error("GameState has too many board cards");
  }
  compact.board_count = static_cast<uint8_t>(state.board_cards.size());
  for (size_t i = 0; i < state.board_cards.size(); ++i) {
    compact.board_cards[i] = state.board_cards[i];
  }
  compact.board_mask = state.board_mask;
  if (state.history.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("GameState history exceeds compact row capacity");
  }
  compact.history_size = static_cast<uint16_t>(state.history.size());
  if (state.history.size() > CompactPublicState::kMaxHistoryActions) {
    compact.history_overflow_offset =
        static_cast<uint32_t>(public_state_history_overflow_.size());
    compact.history_overflow_size = static_cast<uint16_t>(
        state.history.size() - CompactPublicState::kMaxHistoryActions);
    public_state_history_overflow_.reserve(
        public_state_history_overflow_.size() +
        compact.history_overflow_size);
  }
  for (size_t i = 0; i < state.history.size(); ++i) {
    const GameAction& action = state.history[i];
    if (i < CompactPublicState::kMaxHistoryActions) {
      compact.history_amounts[i] = action.amount;
      compact.history_players[i] = static_cast<int8_t>(action.player);
      compact.history_kinds[i] = action.kind;
      continue;
    }

    CompactAction compact_action;
    compact_action.amount = action.amount;
    compact_action.player = static_cast<int8_t>(action.player);
    compact_action.kind = action.kind;
    public_state_history_overflow_.push_back(compact_action);
    compact.last_action = compact_action;
  }
  if (!state.history.empty()) {
    const GameAction& action = state.history.back();
    compact.last_action = {action.amount, static_cast<int8_t>(action.player),
                           action.kind};
  }
  compact.street = state.street;
  compact.all_in = state.all_in;
  compact.folded_player = state.folded_player;
  compact.player_to_act = state.player_to_act;
  compact.player_contribution = state.player_contribution;
  compact.player_contribution_count = state.player_contribution_count;
  compact.betting_history_id = betting_history_id;
  return compact;
}

GameState CFRSolver::materialize_game_state(
    const CompactPublicState& compact) const {
  GameState state;
  state.stack[0] = compact.stack[0];
  state.stack[1] = compact.stack[1];
  state.pot = compact.pot;
  for (int i = 0; i < compact.board_count; ++i) {
    AddBoardCard(state, compact.board_cards[static_cast<size_t>(i)]);
  }
  if (state.board_mask != compact.board_mask) {
    throw std::logic_error("Compact public state board mask mismatch");
  }

  state.history.reserve(compact.history_size);
  const uint16_t inline_history_size = std::min<uint16_t>(
      compact.history_size, CompactPublicState::kMaxHistoryActions);
  for (uint16_t i = 0; i < inline_history_size; ++i) {
    const size_t history_index = static_cast<size_t>(i);
    GameAction action;
    action.amount = compact.history_amounts[history_index];
    action.player = compact.history_players[history_index];
    action.kind = compact.history_kinds[history_index];
    state.history.push_back(action);
  }
  if (compact.history_size > CompactPublicState::kMaxHistoryActions) {
    const size_t overflow_size =
        static_cast<size_t>(compact.history_size -
                            CompactPublicState::kMaxHistoryActions);
    if (compact.history_overflow_size != overflow_size) {
      throw std::logic_error("Compact public state history overflow mismatch");
    }
    const std::vector<CompactAction>& overflow =
        strategy_public_state_history_overflow();
    const size_t offset = compact.history_overflow_offset;
    if (offset > overflow.size() || overflow.size() - offset < overflow_size) {
      throw std::logic_error("Compact public state history overflow missing");
    }
    for (size_t i = 0; i < overflow_size; ++i) {
      const CompactAction& compact_action = overflow[offset + i];
      GameAction action;
      action.amount = compact_action.amount;
      action.player = compact_action.player;
      action.kind = compact_action.kind;
      state.history.push_back(action);
    }
  }

  state.street = compact.street;
  state.all_in = compact.all_in;
  state.folded_player = compact.folded_player;
  state.player_to_act = compact.player_to_act;
  state.player_contribution = compact.player_contribution;
  state.player_contribution_count = compact.player_contribution_count;
  return state;
}

// Read an action from the compact inline history or the solver-owned overflow
// table. Overflow is rare, but keeping it out of PublicStateRow keeps normal
// rows pointer-free.
CompactAction CFRSolver::compact_history_action(
    const CompactPublicState& state,
    uint16_t action_index) const {
  if (action_index >= state.history_size) {
    throw std::logic_error("Compact history action index out of range");
  }
  if (action_index < CompactPublicState::kMaxHistoryActions) {
    const size_t index = static_cast<size_t>(action_index);
    return {state.history_amounts[index],
            state.history_players[index],
            state.history_kinds[index]};
  }

  const size_t overflow_index =
      static_cast<size_t>(action_index - CompactPublicState::kMaxHistoryActions);
  if (overflow_index >= state.history_overflow_size) {
    throw std::logic_error("Compact history overflow index out of range");
  }
  const std::vector<CompactAction>& overflow =
      strategy_public_state_history_overflow();
  const size_t offset = state.history_overflow_offset + overflow_index;
  if (offset >= overflow.size()) {
    throw std::logic_error("Compact history overflow missing");
  }
  return overflow[offset];
}

int CFRSolver::compact_outstanding_to_call(
    const CompactPublicState& state,
    int player) const {
  return std::max(
      0, state.player_contribution[Opponent(player)] -
             state.player_contribution[player]);
}

int CFRSolver::compact_commit_chips(CompactPublicState& state,
                                    int player,
                                    int requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  const int committed = std::min(requested, state.stack[player]);
  state.player_contribution[player] += committed;
  state.stack[player] -= committed;
  state.pot += committed;
  if (state.stack[player] == 0) {
    state.all_in = true;
  }
  return committed;
}

void CFRSolver::append_compact_history_action(
    CompactPublicState& state,
    const GameAction& action) {
  if (state.history_size == std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("Compact public state history is full");
  }

  const size_t history_index = state.history_size;
  if (history_index < CompactPublicState::kMaxHistoryActions) {
    state.history_amounts[history_index] = action.amount;
    state.history_players[history_index] = static_cast<int8_t>(action.player);
    state.history_kinds[history_index] = action.kind;
  } else {
    if (state.history_overflow_size == 0) {
      state.history_overflow_offset =
          static_cast<uint32_t>(public_state_history_overflow_.size());
    } else if (state.history_overflow_offset +
                   state.history_overflow_size !=
               public_state_history_overflow_.size()) {
      const uint32_t old_offset = state.history_overflow_offset;
      const uint16_t old_size = state.history_overflow_size;
      state.history_overflow_offset =
          static_cast<uint32_t>(public_state_history_overflow_.size());
      for (uint16_t i = 0; i < old_size; ++i) {
        public_state_history_overflow_.push_back(
            strategy_public_state_history_overflow()[old_offset + i]);
      }
    }
    state.history_overflow_size += 1;
    public_state_history_overflow_.push_back(
        {action.amount, static_cast<int8_t>(action.player), action.kind});
  }
  state.last_action = {action.amount, static_cast<int8_t>(action.player),
                       action.kind};
  state.history_size += 1;
}

CompactPublicState CFRSolver::apply_compact_action(
    const CompactPublicState& parent,
    const GameAction& action,
    uint32_t child_betting_history_id) {
  CompactPublicState child = parent;
  child.betting_history_id = child_betting_history_id;

  int player = child.player_to_act;
  if (!IsPlayer(player)) {
    throw std::invalid_argument("No player can act in this state");
  }
  if (child.folded_player >= 0) {
    throw std::invalid_argument("Cannot act after a player has folded");
  }
  if (child.stack[player] <= 0) {
    throw std::invalid_argument("Player has no chips to act");
  }

  const int opponent = Opponent(player);
  const int to_call = compact_outstanding_to_call(child, player);
  GameAction applied = action;
  applied.player = player;

  switch (action.kind) {
    case ActionKind::kFold:
      applied.amount = 0;
      child.folded_player = player;
      child.player_to_act = -1;
      break;
    case ActionKind::kCheck:
      if (to_call != 0) {
        throw std::invalid_argument("Cannot check facing a bet");
      }
      applied.amount = 0;
      child.player_to_act = opponent;
      break;
    case ActionKind::kCall: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot call without a bet");
      }
      applied.amount = compact_commit_chips(child, player, to_call);
      child.player_to_act = opponent;
      break;
    }
    case ActionKind::kBet: {
      if (to_call != 0) {
        throw std::invalid_argument("Cannot bet facing a bet");
      }
      if (action.amount >= child.stack[player]) {
        throw std::invalid_argument("Use all-in for full-stack bets");
      }
      applied.amount = compact_commit_chips(child, player, action.amount);
      child.player_to_act = opponent;
      break;
    }
    case ActionKind::kRaise: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot raise without a bet");
      }
      if (action.amount <= to_call || child.stack[player] <= to_call) {
        throw std::invalid_argument("Raise must exceed the call amount");
      }
      if (action.amount >= child.stack[player]) {
        throw std::invalid_argument("Use all-in for full-stack raises");
      }
      applied.amount = compact_commit_chips(child, player, action.amount);
      child.player_to_act = opponent;
      break;
    }
    case ActionKind::kAllIn:
      applied.amount =
          compact_commit_chips(child, player, child.stack[player]);
      child.player_to_act = opponent;
      break;
    case ActionKind::kNoAction:
      throw std::invalid_argument("Unknown action type");
  }

  append_compact_history_action(child, applied);
  return child;
}

void CFRSolver::reset_compact_history(CompactPublicState& state) {
  state.history_size = 0;
  state.last_action = CompactAction{};
  state.history_overflow_offset = std::numeric_limits<uint32_t>::max();
  state.history_overflow_size = 0;
}

int CFRSolver::compact_first_player_for_street(
    const CompactPublicState& state) const {
  return state.street == StreetKind::kPreflop ? 0 : 1;
}

void CFRSolver::add_compact_board_card(CompactPublicState& state,
                                       CardId card) {
  if (state.board_count >= state.board_cards.size()) {
    throw std::invalid_argument(
        "Compact board already has five cards on street " +
        std::to_string(static_cast<int>(state.street)) +
        " history=" + std::to_string(state.history_size) +
        " all_in=" + std::to_string(state.all_in ? 1 : 0) +
        " player_to_act=" + std::to_string(state.player_to_act) +
        " betting_history_id=" + std::to_string(state.betting_history_id));
  }
  const CardMask card_bit = CardBit(card);
  if ((state.board_mask & card_bit) != 0) {
    throw std::invalid_argument("Duplicate board card");
  }
  state.board_cards[state.board_count] = card;
  state.board_count += 1;
  state.board_mask |= card_bit;
}

CompactPublicState CFRSolver::apply_compact_chance(
    const CompactPublicState& parent,
    absl::Span<const CardId> cards,
    uint32_t child_betting_history_id) {
  CompactPublicState child = parent;
  child.betting_history_id = child_betting_history_id;

  switch (child.street) {
    case StreetKind::kPreflop:
      child.street = StreetKind::kFlop;
      break;
    case StreetKind::kFlop:
      child.street = StreetKind::kTurn;
      break;
    case StreetKind::kTurn:
      child.street = StreetKind::kRiver;
      break;
    case StreetKind::kRiver:
      break;
  }

  for (CardId card : cards) {
    add_compact_board_card(child, card);
  }
  reset_compact_history(child);
  child.player_to_act = compact_first_player_for_street(child);
  return child;
}

CFRSolver::PublicStateRow CFRSolver::make_public_state_row(
    uint32_t betting_history_id,
    const GameState& state) {
  return make_public_state_row(
      compact_public_state_from_game_state(betting_history_id, state));
}

CFRSolver::PublicStateRow CFRSolver::make_public_state_row(
    CompactPublicState state) {
  PublicStateRow row;
  row.public_bucket = card_abstraction_.public_bucket(state);
  row.is_terminal = game_tree_->is_terminal(state);
  row.player_to_act = game_tree_->get_player_to_act(state);
  row.state = std::move(state);
  row.is_chance_node = !row.is_terminal && row.player_to_act == -1;

  if (!row.is_terminal && !row.is_chance_node && IsPlayer(row.player_to_act)) {
    row.action_count = game_tree_->get_legal_actions(row.state, row.actions);
    for (int i = 0; i < row.action_count; ++i) {
      row.action_ids[static_cast<size_t>(i)] =
          GameTree::action_key(row.actions[static_cast<size_t>(i)]);
    }
  }

  return row;
}

CFRSolver::InfoSetRow CFRSolver::append_info_set_actions(
    const int* action_ids,
    int num_actions) {
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(action_ids_.size());
  row.action_count = static_cast<uint16_t>(num_actions);
  const size_t required_action_capacity = action_ids_.size() + num_actions;
  if (required_action_capacity > action_ids_.capacity()) {
    const size_t new_capacity =
        std::max(required_action_capacity,
                 action_ids_.empty() ? size_t{4096}
                                     : action_ids_.capacity() * 2);
    action_ids_.reserve(new_capacity);
    cumulative_regrets_.reserve(new_capacity);
    cumulative_strategies_.reserve(new_capacity);
  }
  for (int i = 0; i < num_actions; ++i) {
    action_ids_.push_back(action_ids[i]);
    cumulative_regrets_.push_back(0.0f);
    cumulative_strategies_.push_back(0.0f);
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
  }
  return row;
}

uint32_t CFRSolver::get_or_create_public_state_id(
    uint32_t betting_history_id,
    const GameState& state) {
  const std::optional<uint32_t> public_state_id =
      get_or_create_public_state_row(betting_history_id, state);
  if (!public_state_id.has_value()) {
    throw std::logic_error("Could not allocate public state row");
  }
  return *public_state_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    uint32_t betting_history_id,
    const GameState& state) {
  if (strategy_tables_view_ != nullptr) {
    const auto& public_state_ids = strategy_public_state_ids();
    auto existing = public_state_ids.find(
        make_public_state_key(betting_history_id, state));
    if (existing == public_state_ids.end()) {
      return std::nullopt;
    }
    return existing->second;
  }

  PublicStateKey key = make_public_state_key(betting_history_id, state);
  auto existing = public_state_ids_.find(key);
  if (existing != public_state_ids_.end()) {
    return existing->second;
  }

  if (config_.max_public_states > 0 &&
      static_cast<int>(public_state_rows_.size()) >=
          config_.max_public_states) {
    return std::nullopt;
  }

  const uint32_t public_state_id =
      static_cast<uint32_t>(public_state_ids_.size());
  public_state_ids_.emplace(std::move(key), public_state_id);
  public_state_rows_.push_back(
      make_public_state_row(betting_history_id, state));
  cache_betting_history_actions(betting_history_id,
                                public_state_rows_.back());
  return public_state_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    CompactPublicState state) {
  if (strategy_tables_view_ != nullptr) {
    const auto& public_state_ids = strategy_public_state_ids();
    auto existing = public_state_ids.find(
        make_public_state_key(state.betting_history_id, state));
    if (existing == public_state_ids.end()) {
      return std::nullopt;
    }
    return existing->second;
  }

  const uint32_t betting_history_id = state.betting_history_id;
  PublicStateKey key = make_public_state_key(betting_history_id, state);
  auto existing = public_state_ids_.find(key);
  if (existing != public_state_ids_.end()) {
    return existing->second;
  }

  if (config_.max_public_states > 0 &&
      static_cast<int>(public_state_rows_.size()) >=
          config_.max_public_states) {
    return std::nullopt;
  }

  const uint32_t public_state_id =
      static_cast<uint32_t>(public_state_ids_.size());
  public_state_ids_.emplace(std::move(key), public_state_id);
  public_state_rows_.push_back(make_public_state_row(std::move(state)));
  cache_betting_history_actions(betting_history_id,
                                public_state_rows_.back());
  return public_state_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_public_state_row(
    const GameState& state) {
  if (strategy_tables_view_ != nullptr) {
    const std::optional<uint32_t> betting_history_id =
        strategy_betting_history_id(state);
    if (!betting_history_id.has_value()) {
      return std::nullopt;
    }
    return get_or_create_public_state_row(*betting_history_id, state);
  }

  const uint32_t betting_history_id = get_or_create_betting_history_id(state);
  return get_or_create_public_state_row(betting_history_id, state);
}

uint32_t CFRSolver::get_or_create_public_state_id(const GameState& state) {
  const uint32_t betting_history_id = get_or_create_betting_history_id(state);
  return get_or_create_public_state_id(betting_history_id, state);
}

uint32_t CFRSolver::get_or_create_betting_history_id(const GameState& state) {
  BettingHistoryKey key = make_betting_history_key(state);
  auto existing = betting_history_ids_.find(key);
  if (existing != betting_history_ids_.end()) {
    if (betting_history_rows_.size() <= existing->second) {
      betting_history_rows_.resize(static_cast<size_t>(existing->second) + 1);
    }
    return existing->second;
  }
  const uint32_t betting_history_id =
      static_cast<uint32_t>(betting_history_ids_.size());
  betting_history_ids_.emplace(std::move(key), betting_history_id);
  betting_history_rows_.push_back(make_betting_history_row(state));
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_betting_history_id(
    const CompactPublicState& state) {
  BettingHistoryKey key = make_betting_history_key(state);
  auto existing = betting_history_ids_.find(key);
  if (existing != betting_history_ids_.end()) {
    if (betting_history_rows_.size() <= existing->second) {
      betting_history_rows_.resize(static_cast<size_t>(existing->second) + 1);
    }
    return existing->second;
  }
  const uint32_t betting_history_id =
      static_cast<uint32_t>(betting_history_ids_.size());
  betting_history_ids_.emplace(std::move(key), betting_history_id);
  betting_history_rows_.push_back(make_betting_history_row(state));
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_betting_history_id(GameTree::Node& node) {
  uint32_t betting_history_id = node.betting_history_id;
  if (betting_history_id == GameTree::Node::kInvalidBettingHistoryId) {
    betting_history_id = get_or_create_betting_history_id(node.state);
    node.betting_history_id = betting_history_id;
  }
  cache_betting_history_actions(betting_history_id, node);
  return betting_history_id;
}

uint32_t CFRSolver::get_or_create_action_child_betting_history_id(
    uint32_t parent_betting_history_id,
    int action_index,
    const CompactPublicState& child_state) {
  if (parent_betting_history_id < betting_history_rows_.size()) {
    BettingHistoryRow& parent_row =
        betting_history_rows_[parent_betting_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      const uint32_t child_id =
          parent_row.action_child_ids[static_cast<size_t>(action_index)];
      if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
        return child_id;
      }
    }
  }

  const uint32_t child_id = get_or_create_betting_history_id(child_state);
  if (parent_betting_history_id < betting_history_rows_.size()) {
    BettingHistoryRow& parent_row =
        betting_history_rows_[parent_betting_history_id];
    if (action_index >= 0 && action_index < parent_row.action_count) {
      parent_row.action_child_ids[static_cast<size_t>(action_index)] =
          child_id;
    }
  }
  return child_id;
}

uint32_t CFRSolver::get_or_create_chance_child_betting_history_id(
    uint32_t parent_betting_history_id,
    const CompactPublicState& child_state) {
  if (parent_betting_history_id < betting_history_rows_.size()) {
    BettingHistoryRow& parent_row =
        betting_history_rows_[parent_betting_history_id];
    if (parent_row.chance_child_id !=
        GameTree::Node::kInvalidBettingHistoryId) {
      return parent_row.chance_child_id;
    }
  }

  const uint32_t child_id = get_or_create_betting_history_id(child_state);
  if (parent_betting_history_id < betting_history_rows_.size()) {
    betting_history_rows_[parent_betting_history_id].chance_child_id =
        child_id;
  }
  return child_id;
}

uint32_t CFRSolver::get_or_create_public_state_id(GameTree::Node& node) {
  const uint32_t betting_history_id = get_or_create_betting_history_id(node);
  return get_or_create_public_state_id(node, betting_history_id);
}

uint32_t CFRSolver::get_or_create_public_state_id(
    GameTree::Node& node,
    uint32_t betting_history_id) {
  if (node.public_state_id != GameTree::Node::kInvalidPublicStateId) {
    return node.public_state_id;
  }

  PublicStateKey key = make_public_state_key(betting_history_id, node.state);
  auto existing = public_state_ids_.find(key);
  if (existing != public_state_ids_.end()) {
    node.public_state_id = existing->second;
    return node.public_state_id;
  }

  node.public_state_id = static_cast<uint32_t>(public_state_ids_.size());
  public_state_ids_.emplace(std::move(key), node.public_state_id);
  const bool can_store_row =
      !frozen_ && (config_.max_public_states <= 0 ||
                   static_cast<int>(public_state_rows_.size()) <
                       config_.max_public_states);
  if (can_store_row) {
    PublicStateRow row = make_public_state_row(betting_history_id, node.state);
    row.is_terminal = node.is_terminal;
    row.is_chance_node = node.is_chance_node;
    row.player_to_act = node.player_to_act;
    row.action_count = node.action_count;
    for (int i = 0; i < node.action_count; ++i) {
      row.actions[static_cast<size_t>(i)] = node.actions[i].action;
      row.action_ids[static_cast<size_t>(i)] = node.actions[i].key;
    }
    public_state_rows_.push_back(std::move(row));
  }
  return node.public_state_id;
}

void CFRSolver::cache_betting_history_actions(
    uint32_t betting_history_id,
    const GameTree::Node& node) {
  if (node.action_count == 0 ||
      betting_history_id >= betting_history_rows_.size()) {
    return;
  }

  BettingHistoryRow& row =
      betting_history_rows_[static_cast<size_t>(betting_history_id)];
  row.action_count = node.action_count;
  for (int i = 0; i < node.action_count; ++i) {
    row.action_ids[static_cast<size_t>(i)] = node.actions[i].key;
  }
}

void CFRSolver::cache_betting_history_actions(
    uint32_t betting_history_id,
    const PublicStateRow& row) {
  if (row.action_count == 0 ||
      betting_history_id >= betting_history_rows_.size()) {
    return;
  }

  BettingHistoryRow& betting_history =
      betting_history_rows_[static_cast<size_t>(betting_history_id)];
  betting_history.action_count = row.action_count;
  for (int i = 0; i < row.action_count; ++i) {
    betting_history.action_ids[static_cast<size_t>(i)] =
        row.action_ids[static_cast<size_t>(i)];
  }
}

void CFRSolver::validate_public_state_row_actions(
    uint32_t public_state_id) const {
  const auto& public_rows = strategy_public_state_rows();
  if (public_state_id >= public_rows.size()) {
    throw std::logic_error("Public state row is missing");
  }
  const PublicStateRow& row = public_rows[public_state_id];
  const auto& betting_history_rows = strategy_betting_history_rows();
  if (row.state.betting_history_id >= betting_history_rows.size()) {
    throw std::logic_error("Betting history row is missing");
  }
  const BettingHistoryRow& betting_history =
      betting_history_rows[row.state.betting_history_id];
  if (betting_history.action_count != row.action_count) {
    throw std::logic_error("Betting history actions are not aligned");
  }
  for (int i = 0; i < row.action_count; ++i) {
    if (betting_history.action_ids[static_cast<size_t>(i)] !=
        row.action_ids[static_cast<size_t>(i)]) {
      throw std::logic_error("Betting history action key mismatch");
    }
  }
}

std::optional<uint32_t> CFRSolver::get_or_create_action_child_public_state(
    uint32_t public_state_id,
    int action_index) {
  const auto& rows = strategy_public_state_rows();
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& read_row = rows[public_state_id];
  if (action_index < 0 || action_index >= read_row.action_count) {
    throw std::logic_error(
        "get_or_create_action_child_public_state: action index out of range");
  }

  const size_t action_slot = static_cast<size_t>(action_index);
  const uint32_t existing_child_id = read_row.action_child_ids[action_slot];
  if (existing_child_id != GameTree::Node::kInvalidPublicStateId) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_hits);
    if (existing_child_id == kCappedPublicStateId) {
      return std::nullopt;
    }
    return existing_child_id;
  }
  if (strategy_tables_view_ != nullptr) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(public_state_rows_.size()) >=
          config_.max_public_states) {
    public_state_rows_[public_state_id].action_child_ids[action_slot] =
        kCappedPublicStateId;
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  const GameAction action =
      public_state_rows_[public_state_id].actions[action_slot];
  const uint32_t parent_betting_history_id =
      public_state_rows_[public_state_id].state.betting_history_id;
  CompactPublicState child_state = apply_compact_action(
      public_state_rows_[public_state_id].state, action,
      GameTree::Node::kInvalidBettingHistoryId);
  const uint32_t child_betting_history_id =
      get_or_create_action_child_betting_history_id(
          parent_betting_history_id, action_index, child_state);
  child_state.betting_history_id = child_betting_history_id;
  std::optional<uint32_t> child_id =
      get_or_create_public_state_row(std::move(child_state));
  if (!child_id.has_value()) {
    public_state_rows_[public_state_id].action_child_ids[action_slot] =
        kCappedPublicStateId;
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  public_state_rows_[public_state_id].action_child_ids[action_slot] = *child_id;
  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.child_nodes_created);
  return child_id;
}

std::optional<uint32_t> CFRSolver::get_or_create_chance_child_public_state(
    uint32_t public_state_id,
    absl::Span<const CardId> cards) {
  const auto& rows = strategy_public_state_rows();
  if (public_state_id >= rows.size()) {
    return std::nullopt;
  }
  const int child_key = ChanceCardsKey(cards);
  const uint64_t map_key = PublicChanceChildKey(public_state_id, child_key);
  const auto& chance_children = strategy_public_chance_child_ids();
  auto existing = chance_children.find(map_key);
  if (existing != chance_children.end()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_hits);
    return existing->second;
  }
  if (strategy_tables_view_ != nullptr) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }
  if (config_.max_public_states > 0 &&
      static_cast<int>(public_state_rows_.size()) >=
          config_.max_public_states) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  const uint32_t parent_betting_history_id =
      public_state_rows_[public_state_id].state.betting_history_id;
  CompactPublicState child_state = apply_compact_chance(
      public_state_rows_[public_state_id].state, cards,
      GameTree::Node::kInvalidBettingHistoryId);
  const uint32_t child_betting_history_id =
      get_or_create_chance_child_betting_history_id(
          parent_betting_history_id, child_state);
  child_state.betting_history_id = child_betting_history_id;
  std::optional<uint32_t> child_id =
      get_or_create_public_state_row(std::move(child_state));
  if (!child_id.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return std::nullopt;
  }

  public_chance_child_ids_.emplace(map_key, *child_id);
  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.child_nodes_created);
  return child_id;
}

void CFRSolver::cache_action_betting_history_transition(
    GameTree::Node& node,
    int action_index,
    GameTree::Node& child_node) {
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error(
        "cache_action_betting_history_transition: action index out of range");
  }

  if (strategy_tables_view_ == nullptr) {
    const uint32_t parent_id = get_or_create_betting_history_id(node);
    if (betting_history_rows_.size() <= parent_id) {
      betting_history_rows_.resize(static_cast<size_t>(parent_id) + 1);
    }
    const size_t parent_index = static_cast<size_t>(parent_id);
    const size_t action_slot = static_cast<size_t>(action_index);
    uint32_t child_id =
        betting_history_rows_[parent_index].action_child_ids[action_slot];
    if (child_id == GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_misses);
      child_id = get_or_create_betting_history_id(child_node);
      betting_history_rows_[parent_index].action_child_ids[action_slot] =
          child_id;
    } else {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
    }
    return;
  }

  const std::optional<uint32_t> parent_lookup =
      strategy_betting_history_id(node);
  if (!parent_lookup.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return;
  }

  const uint32_t parent_id = *parent_lookup;
  node.betting_history_id = parent_id;
  const auto& rows = strategy_betting_history_rows();
  if (parent_id < rows.size()) {
    const uint32_t child_id =
        rows[parent_id].action_child_ids[static_cast<size_t>(action_index)];
    if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
      return;
    }
  }

  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  const std::optional<uint32_t> child_lookup =
      strategy_betting_history_id(child_node);
  if (child_lookup.has_value()) {
    child_node.betting_history_id = *child_lookup;
  }
}

void CFRSolver::cache_chance_betting_history_transition(
    GameTree::Node& node,
    GameTree::Node& child_node) {
  if (strategy_tables_view_ == nullptr) {
    const uint32_t parent_id = get_or_create_betting_history_id(node);
    if (betting_history_rows_.size() <= parent_id) {
      betting_history_rows_.resize(static_cast<size_t>(parent_id) + 1);
    }
    const size_t parent_index = static_cast<size_t>(parent_id);
    uint32_t child_id = betting_history_rows_[parent_index].chance_child_id;
    if (child_id == GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_misses);
      child_id = get_or_create_betting_history_id(child_node);
      betting_history_rows_[parent_index].chance_child_id = child_id;
    } else {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
    }
    return;
  }

  const std::optional<uint32_t> parent_lookup =
      strategy_betting_history_id(node);
  if (!parent_lookup.has_value()) {
    POKER_RECORD_TRAVERSAL_STAT(
        ++traversal_stats_.betting_history_transition_misses);
    return;
  }

  const uint32_t parent_id = *parent_lookup;
  node.betting_history_id = parent_id;
  const auto& rows = strategy_betting_history_rows();
  if (parent_id < rows.size()) {
    const uint32_t child_id = rows[parent_id].chance_child_id;
    if (child_id != GameTree::Node::kInvalidBettingHistoryId) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++traversal_stats_.betting_history_transition_hits);
      child_node.betting_history_id = child_id;
      return;
    }
  }

  POKER_RECORD_TRAVERSAL_STAT(
      ++traversal_stats_.betting_history_transition_misses);
  const std::optional<uint32_t> child_lookup =
      strategy_betting_history_id(child_node);
  if (child_lookup.has_value()) {
    child_node.betting_history_id = *child_lookup;
  }
}

CFRSolver::PublicInfoSetSlab& CFRSolver::get_or_create_public_info_set_slab(
    uint32_t public_state_id) {
  if (public_info_set_slabs_.size() <= public_state_id) {
    public_info_set_slabs_.resize(static_cast<size_t>(public_state_id) + 1);
  }
  std::unique_ptr<PublicInfoSetSlab>& slab =
      public_info_set_slabs_[public_state_id];
  if (slab == nullptr) {
    slab = std::make_unique<PublicInfoSetSlab>();
  }
  return *slab;
}

const CFRSolver::PublicInfoSetSlab* CFRSolver::public_info_set_slab(
    uint32_t public_state_id) const {
  const auto& slabs = strategy_public_info_set_slabs();
  if (public_state_id >= slabs.size()) {
    return nullptr;
  }
  return slabs[public_state_id].get();
}

const CFRSolver::InfoSetRow* CFRSolver::find_info_set_row(
    InfoSetAddress address) const {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return nullptr;
  }
  const PublicInfoSetSlab* slab = public_info_set_slab(address.public_state_id);
  if (slab == nullptr) {
    return nullptr;
  }
  const PublicInfoSetSlabPlayer& player_slab = slab->players[address.player];
  return find_info_set_row(player_slab, address.private_bucket);
}

const CFRSolver::InfoSetRow* CFRSolver::find_info_set_row(
    const PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  if (private_bucket >= kComboCount) {
    return nullptr;
  }
  const size_t chunk_index = private_bucket / kPrivateBucketChunkSize;
  const size_t chunk_offset = private_bucket % kPrivateBucketChunkSize;
  const std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    return nullptr;
  }
  const int32_t row_id = chunk->rows[chunk_offset];
  if (row_id < 0 ||
      static_cast<size_t>(row_id) >= player_slab.rows.size()) {
    return nullptr;
  }
  return &player_slab.rows[row_id];
}

int32_t& CFRSolver::get_or_create_private_row_slot(
    PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  const size_t chunk_index = private_bucket / kPrivateBucketChunkSize;
  const size_t chunk_offset = private_bucket % kPrivateBucketChunkSize;
  std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    chunk = std::make_unique<PrivateRowChunk>();
  }
  return chunk->rows[chunk_offset];
}

std::optional<CFRSolver::InfoSetRow> CFRSolver::get_or_create_info_set_row(
    InfoSetAddress address,
    const int* action_ids,
    int num_actions) {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return std::nullopt;
  }

  if (const InfoSetRow* row = find_info_set_row(address)) {
    return *row;
  }

  if (frozen_ || (config_.max_info_sets > 0 &&
                  static_cast<int>(info_set_count_) >= config_.max_info_sets)) {
    return std::nullopt;
  }

  InfoSetRow row = append_info_set_actions(action_ids, num_actions);
  PublicInfoSetSlab& slab =
      get_or_create_public_info_set_slab(address.public_state_id);
  PublicInfoSetSlabPlayer& player_slab = slab.players[address.player];
  int32_t& row_id =
      get_or_create_private_row_slot(player_slab, address.private_bucket);
  row_id = static_cast<int32_t>(player_slab.rows.size());
  player_slab.rows.push_back(row);
  ++info_set_count_;
  return row;
}

CFRSolver::StrategyTablesView CFRSolver::strategy_tables_view() {
  return {&betting_history_ids_,
          &betting_history_rows_,
          &public_state_ids_,
          &public_state_rows_,
          &public_chance_child_ids_,
          &public_state_history_overflow_,
          &public_info_set_slabs_,
          &action_ids_,
          &cumulative_regrets_,
          &cumulative_strategies_};
}

std::optional<uint32_t> CFRSolver::strategy_public_state_id(
    GameTree::Node& node) {
  if (node.public_state_id != GameTree::Node::kInvalidPublicStateId) {
    return node.public_state_id;
  }

  const std::optional<uint32_t> betting_history_id =
      strategy_betting_history_id(node);
  if (!betting_history_id.has_value()) {
    return std::nullopt;
  }
  const auto& public_state_ids = strategy_public_state_ids();
  auto public_state =
      public_state_ids.find(make_public_state_key(*betting_history_id,
                                                  node.state));
  if (public_state == public_state_ids.end()) {
    return std::nullopt;
  }

  node.public_state_id = public_state->second;
  return node.public_state_id;
}

std::optional<uint32_t> CFRSolver::strategy_betting_history_id(
    const GameState& state) const {
  const auto& betting_history_ids = strategy_betting_history_ids();
  auto betting_history =
      betting_history_ids.find(make_betting_history_key(state));
  if (betting_history == betting_history_ids.end()) {
    return std::nullopt;
  }
  return betting_history->second;
}

std::optional<uint32_t> CFRSolver::strategy_betting_history_id(
    GameTree::Node& node) {
  if (node.betting_history_id !=
      GameTree::Node::kInvalidBettingHistoryId) {
    return node.betting_history_id;
  }

  std::optional<uint32_t> betting_history_id =
      strategy_betting_history_id(node.state);
  if (betting_history_id.has_value()) {
    node.betting_history_id = *betting_history_id;
  }
  return betting_history_id;
}

const absl::flat_hash_map<CFRSolver::PublicStateKey, uint32_t,
                          CFRSolver::PublicStateKeyHash>&
CFRSolver::strategy_public_state_ids() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->public_state_ids
             : public_state_ids_;
}

const std::vector<CFRSolver::PublicStateRow>&
CFRSolver::strategy_public_state_rows() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->public_state_rows
             : public_state_rows_;
}

const absl::flat_hash_map<uint64_t, uint32_t>&
CFRSolver::strategy_public_chance_child_ids() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->public_chance_child_ids
             : public_chance_child_ids_;
}

const std::vector<CompactAction>&
CFRSolver::strategy_public_state_history_overflow() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->public_state_history_overflow
             : public_state_history_overflow_;
}

const absl::flat_hash_map<CFRSolver::BettingHistoryKey, uint32_t,
                          CFRSolver::BettingHistoryKeyHash>&
CFRSolver::strategy_betting_history_ids() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->betting_history_ids
             : betting_history_ids_;
}

const std::vector<CFRSolver::BettingHistoryRow>&
CFRSolver::strategy_betting_history_rows() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->betting_history_rows
             : betting_history_rows_;
}

const std::vector<std::unique_ptr<CFRSolver::PublicInfoSetSlab>>&
CFRSolver::strategy_public_info_set_slabs() const {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->public_info_set_slabs
             : public_info_set_slabs_;
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

std::vector<float>& CFRSolver::mutable_strategy_cumulative_regrets() {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->cumulative_regrets
             : cumulative_regrets_;
}

std::vector<float>& CFRSolver::mutable_strategy_cumulative_strategies() {
  return strategy_tables_view_ != nullptr
             ? *strategy_tables_view_->cumulative_strategies
             : cumulative_strategies_;
}

void CFRSolver::add_traversal_stats(const TraversalStats& stats) {
  traversal_stats_.cfr_updates += stats.cfr_updates;
  traversal_stats_.preflop_updates += stats.preflop_updates;
  traversal_stats_.flop_updates += stats.flop_updates;
  traversal_stats_.turn_updates += stats.turn_updates;
  traversal_stats_.river_updates += stats.river_updates;
  traversal_stats_.child_nodes_created += stats.child_nodes_created;
  traversal_stats_.chance_samples += stats.chance_samples;
  traversal_stats_.terminal_utility_calls += stats.terminal_utility_calls;
  traversal_stats_.fold_utility_calls += stats.fold_utility_calls;
  traversal_stats_.showdown_utility_calls += stats.showdown_utility_calls;
  traversal_stats_.action_entry_touches += stats.action_entry_touches;
  traversal_stats_.betting_history_transition_hits +=
      stats.betting_history_transition_hits;
  traversal_stats_.betting_history_transition_misses +=
      stats.betting_history_transition_misses;
}

void CFRSolver::run(int iterations, ComboId player_a_hand,
                    ComboId player_b_hand) {
  if (iterations <= 0) {
    return;
  }

  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return;
  }
  const int max_depth = config_.max_depth;
  TraversalScratch scratch;
  scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  for (int i = 0; i < iterations; ++i) {
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    cumulative_root_utility_ += cfr_with_ranges(
        *root_public_state_id, player_a_cards, player_b_cards, reach_probabilities,
        iterations_run_.load(std::memory_order_relaxed), 0, max_depth, scratch,
        std::nullopt, std::nullopt);
    iterations_run_.fetch_add(1, std::memory_order_relaxed);
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

  VLOG(1) << "Preparing compact public-state rows...";
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return;
  }
  VLOG(1) << "Compact root row has "
          << static_cast<int>(
                 public_state_rows_[*root_public_state_id].action_count)
          << " legal actions";

  const int num_threads =
      config_.num_training_threads <= 1 ? 1 : config_.num_training_threads;

  // Determine single-threaded warmup count.
  // Warmup populates the game tree and info set table before we freeze.
  // - If num_threads == 1 there is no parallel phase; warmup = all iterations.
  // - If warmup_iterations is set explicitly, honour it (capped to iterations).
  // - Otherwise: run until max_info_sets is reached OR info set growth stalls,
  //   using at most half the requested iterations as a ceiling.
  int warmup_count = iterations;
  if (num_threads > 1 && !frozen_) {
    if (config_.warmup_iterations > 0) {
      warmup_count = std::min(config_.warmup_iterations, iterations);
    } else if (config_.max_info_sets > 0) {
      // Run until the cap is hit; cap the warmup at half the total budget.
      warmup_count = std::min(iterations / 2 + 1, iterations);
    } else {
      // No cap: default warmup = 10% of iterations, minimum 100.
      warmup_count = std::max(100, iterations / 10);
      warmup_count = std::min(warmup_count, iterations);
    }
  }

  // Phase 1: single-threaded warmup (allocates public states + info sets).
  LOG(INFO) << "Starting CFR iterations...";
  VLOG(1) << "Warmup phase: " << warmup_count << " single-threaded iterations";
  const int max_depth = config_.max_depth;
  TraversalScratch scratch;
  scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));
  for (int i = 0; i < warmup_count; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    PrivateCards player_a_cards = PrivateCards::FromCombo(deal.player_a_combo);
    PrivateCards player_b_cards = PrivateCards::FromCombo(deal.player_b_combo);

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_.load(std::memory_order_relaxed);
    std::array<double, 2> reach_probabilities = {1.0, 1.0};
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    double dealt_value = cfr_with_ranges(
        *root_public_state_id, player_a_cards, player_b_cards,
        reach_probabilities,
        cfr_iteration, 0, max_depth, scratch, player_a_context_range,
        player_b_context_range);

    cumulative_root_utility_ += dealt_value;
    iterations_run_.fetch_add(1, std::memory_order_relaxed);
  }

  const int remaining = iterations - warmup_count;
  if (remaining > 0) {
    // Phase 2: freeze the tables and run remaining iterations in parallel.
    frozen_ = true;
    LOG(INFO) << "Frozen after warmup: " << get_info_set_count()
              << " info sets, " << iterations_run_.load(std::memory_order_relaxed)
              << " warmup iterations. Starting parallel phase ("
              << remaining << " iterations, " << num_threads << " threads)...";
    run_iterations_parallel(remaining, num_threads, *root_public_state_id,
                            range_sampler,
                            player_a_training_range, player_b_training_range);
  }

  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_.load(std::memory_order_relaxed);
  LOG(INFO) << "Information sets: " << get_info_set_count();
  LOG(INFO) << "Public states: " << get_public_state_count();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

void CFRSolver::run_iterations_parallel(
    int iterations,
    int num_threads,
    uint32_t root_public_state_id,
    const RangeSampler& range_sampler,
    const TrainingRange& player_a_training_range,
    const TrainingRange& player_b_training_range) {
  // Each worker gets a snapshot of the frozen strategy tables via
  // strategy_tables_view_ and writes only to the shared atomic float arrays.
  // Workers use their own RNG and TraversalScratch; no locks needed.
  const StrategyTablesView tables = strategy_tables_view();

  ThreadPoolExecutor executor(num_threads);
  std::uniform_int_distribution<unsigned int> seed_dist;
  std::vector<std::future<std::pair<double, TraversalStats>>> futures;
  futures.reserve(num_threads);

  int iterations_remaining = iterations;
  for (int t = 0; t < num_threads; ++t) {
    const int shard = iterations_remaining / (num_threads - t);
    iterations_remaining -= shard;
    if (shard <= 0) {
      continue;
    }

    const unsigned int seed = seed_dist(rng_);
    futures.push_back(executor.submit(
        [this, shard, seed, root_public_state_id, &range_sampler, &tables,
         &player_a_training_range, &player_b_training_range]() mutable {
          // Build a lightweight worker that shares frozen tables.
          CFRSolver worker(config_, utility_cache_,
                           continuation_value_provider_);
          worker.strategy_tables_view_ = &tables;
          worker.frozen_ = true;
          worker.rng_.seed(seed);
          // Per-worker range views (read-only, built from shared training data).
          TrainingRangeView player_a_hands_view;
          TrainingRangeView player_b_hands_view;
          player_a_hands_view.reset_to_all(player_a_training_range);
          player_b_hands_view.reset_to_all(player_b_training_range);

          const int max_depth = config_.max_depth;
          TraversalScratch scratch;
          scratch.reserve_depth(ScratchDepthReserve(config_, max_depth));

          double local_utility = 0.0;
          for (int i = 0; i < shard; ++i) {
            const RangeDeal deal = range_sampler.sample(worker.rng_);
            PrivateCards player_a_cards =
                PrivateCards::FromCombo(deal.player_a_combo);
            PrivateCards player_b_cards =
                PrivateCards::FromCombo(deal.player_b_combo);

            // Use the global iteration counter for CFR+ linear weighting.
            const int cfr_iteration =
                iterations_run_.fetch_add(1, std::memory_order_relaxed);
            std::array<double, 2> reach_probabilities = {1.0, 1.0};
            OptionalTrainingRange player_a_context_range;
            OptionalTrainingRange player_b_context_range;
            if (max_depth > 0) {
              player_a_context_range = std::cref(player_a_hands_view);
              player_b_context_range = std::cref(player_b_hands_view);
            }
            local_utility += worker.cfr_with_ranges(
                root_public_state_id, player_a_cards, player_b_cards,
                reach_probabilities, cfr_iteration, 0, max_depth, scratch,
                player_a_context_range, player_b_context_range);
          }
          return std::make_pair(local_utility, worker.get_traversal_stats());
        }));
  }

  // Accumulate per-worker root utilities and traversal stats into the main solver.
  for (std::future<std::pair<double, TraversalStats>>& f : futures) {
    const auto result = f.get();
    cumulative_root_utility_ += result.first;
    add_traversal_stats(result.second);
  }
}

double CFRSolver::cfr_with_ranges(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  const auto& public_state_rows = strategy_public_state_rows();
  if (public_state_id >= public_state_rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = public_state_rows[public_state_id];

  if (row.is_terminal) {
    POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.terminal_utility_calls);
    if (row.state.folded_player >= 0) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.fold_utility_calls);
    } else {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.showdown_utility_calls);
    }
    if (max_depth > 0) {
      return uncached_utility(row.state, player_a_cards, player_b_cards);
    }
    return utility(row.state, player_a_cards, player_b_cards);
  }

  if (row.is_chance_node) {
    return chance_sampling_cfr(public_state_id, player_a_cards, player_b_cards,
                               reach_probabilities, iteration, depth,
                               max_depth, scratch, player_a_range,
                               player_b_range);
  }

  if (max_depth > 0 && depth >= max_depth) {
    // Continuation providers still consume GameState. This cutoff path is
    // intentionally left materialized until continuation contexts move compact.
    const GameState state = materialize_game_state(row.state);
    ContinuationContext context = build_continuation_context(
        state, player_a_cards.combo, player_b_cards.combo, player_a_range,
        player_b_range);
    return continuation_value_provider_->value(*game_tree_, context);
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player) || row.action_count == 0) {
    return 0.0;
  }
  const StreetKind street = row.state.street;
  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;

  const InfoSetAddress info_set_address{
      public_state_id, player,
      card_abstraction_.private_bucket(player_cards.combo, row.state)};
  const std::optional<InfoSetRow> info_set_row =
      get_or_create_info_set_row(info_set_address,
                                 row.action_ids.data(),
                                 row.action_count);

  const size_t action_count = row.action_count;
  double action_probabilities[GameTree::kMaxActionsPerNode];
  double action_values[GameTree::kMaxActionsPerNode];
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    action_values[action_index] = 0.0;
  }
  const InfoSetRow* info_set_row_ptr =
      info_set_row.has_value() ? &*info_set_row : nullptr;
  if (info_set_row_ptr != nullptr) {
    auto& regrets = mutable_strategy_cumulative_regrets();
    const size_t action_offset = info_set_row_ptr->action_offset;
    double sum_positive_regrets = 0.0;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      const double positive_regret = std::max(
          0.0,
          static_cast<double>(
              AtomicFloatLoad(&regrets[action_offset + action_index])));
      action_probabilities[action_index] = positive_regret;
      sum_positive_regrets += positive_regret;
    }

    if (sum_positive_regrets > 0.0) {
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] /= sum_positive_regrets;
      }
    } else {
      const double uniform_prob = 1.0 / action_count;
      for (size_t action_index = 0; action_index < action_count;
           ++action_index) {
        action_probabilities[action_index] = uniform_prob;
      }
    }
  } else {
    const double uniform_prob = 1.0 / action_count;
    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      action_probabilities[action_index] = uniform_prob;
    }
  }

  double node_value = 0.0;
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  ConditionedRanges& conditioned_player_ranges =
      scratch_frame.conditioned_ranges;
  const bool condition_player_a_range =
      player == 0 && player_a_range.has_value();
  const bool condition_player_b_range =
      player == 1 && player_b_range.has_value();
  if (condition_player_a_range) {
    condition_ranges_for_actions(player_a_range->get(), row.state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  } else if (condition_player_b_range) {
    condition_ranges_for_actions(player_b_range->get(), row.state,
                                 public_state_id, player,
                                 row.action_ids.data(), action_count,
                                 conditioned_player_ranges);
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    std::optional<uint32_t> child_public_state_id =
        get_or_create_action_child_public_state(
            public_state_id, static_cast<int>(action_index));
    if (!child_public_state_id.has_value()) {
      continue;
    }

    const double previous_reach_probability = reach_probabilities[player];
    reach_probabilities[player] =
        previous_reach_probability * action_probabilities[action_index];

    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (condition_player_a_range) {
      child_player_a_range = std::cref(conditioned_player_ranges[action_index]);
    } else if (condition_player_b_range) {
      child_player_b_range = std::cref(conditioned_player_ranges[action_index]);
    }

    const double action_value = cfr_with_ranges(
        *child_public_state_id, player_a_cards, player_b_cards,
        reach_probabilities, iteration, depth + 1, max_depth, scratch,
        child_player_a_range, child_player_b_range);
    action_values[action_index] = action_value;
    reach_probabilities[player] = previous_reach_probability;
    node_value += action_probabilities[action_index] * action_value;
  }

  cfr_update_count_.fetch_add(1, std::memory_order_relaxed);
  POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.cfr_updates);
  POKER_RECORD_TRAVERSAL_STAT(
      traversal_stats_.max_decision_depth =
          std::max(traversal_stats_.max_decision_depth, depth));
  switch (street) {
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

  if (info_set_row.has_value()) {
    const double opponent_reach_prob = reach_probabilities[1 - player];
    const size_t action_offset = info_set_row->action_offset;
    auto& regrets = mutable_strategy_cumulative_regrets();
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
      AtomicCFRPlusRegretUpdate(
          &regrets[action_offset + action_index],
          static_cast<float>(regret));
    }

    if (!config_.regret_only_training) {
      update_strategy(*info_set_row, action_probabilities, action_count,
                      reach_probabilities[player] * (iteration + 1));
    }
  }

  return node_value;
}

double CFRSolver::chance_sampling_cfr(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards,
    std::array<double, 2>& reach_probabilities,
    int iteration,
    int depth,
    int max_depth,
    TraversalScratch& scratch,
    OptionalTrainingRange player_a_range,
    OptionalTrainingRange player_b_range) {
  const auto& rows = strategy_public_state_rows();
  if (public_state_id >= rows.size()) {
    return 0.0;
  }
  // Child creation may grow public_state_rows_, so keep a local copy instead
  // of holding a reference into the vector across samples.
  const CompactPublicState state = rows[public_state_id].state;

  const int samples = ChanceSamples(config_);
  POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.chance_samples += samples);
  RangeScratchFrame& scratch_frame = scratch.frame(depth);
  TrainingRangeView& public_player_a_range =
      scratch_frame.public_player_a_range;
  TrainingRangeView& public_player_b_range =
      scratch_frame.public_player_b_range;

  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const auto cards = SampleStreetCards(
        state, player_a_cards.mask() | player_b_cards.mask(), rng_);
    std::optional<uint32_t> child_public_state_id =
        get_or_create_chance_child_public_state(public_state_id, cards);
    if (!child_public_state_id.has_value()) {
      continue;
    }

    const PublicStateRow& child_row =
        strategy_public_state_rows()[*child_public_state_id];
    OptionalTrainingRange child_player_a_range = player_a_range;
    OptionalTrainingRange child_player_b_range = player_b_range;
    if (player_a_range.has_value()) {
      PublicCompatibleRangeInto(
          player_a_range->get(), child_row.state.board_mask,
          public_player_a_range);
      child_player_a_range = std::cref(public_player_a_range);
    }
    if (player_b_range.has_value()) {
      PublicCompatibleRangeInto(
          player_b_range->get(), child_row.state.board_mask,
          public_player_b_range);
      child_player_b_range = std::cref(public_player_b_range);
    }

    value += cfr_with_ranges(*child_public_state_id, player_a_cards,
                             player_b_cards, reach_probabilities, iteration,
                             depth, max_depth, scratch, child_player_a_range,
                             child_player_b_range);
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

void CFRSolver::average_strategy_probabilities(
    GameTree::Node& node,
    int player,
    const PrivateCards& private_cards,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(node.action_count, 0.0);
  if (node.action_count == 0) {
    return;
  }

  const double uniform_probability = 1.0 / node.action_count;
  const PrivateBucketId private_bucket =
      card_abstraction_.private_bucket(private_cards.combo, node.state);
  auto try_public_state = [&](uint32_t public_state_id) {
    const InfoSetRow* row =
        find_info_set_row({public_state_id, player, private_bucket});
    if (row == nullptr) {
      return false;
    }
    average_strategy_probabilities(
        *row, node, uniform_probability, probabilities);
    return true;
  };

  if (node.public_state_id != GameTree::Node::kInvalidPublicStateId) {
    if (try_public_state(node.public_state_id)) {
      return;
    }
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }

  const std::optional<uint32_t> public_state_id =
      strategy_public_state_id(node);
  if (public_state_id.has_value()) {
    if (try_public_state(*public_state_id)) {
      return;
    }
  }

  std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
}

void CFRSolver::average_strategy_probabilities(
    uint32_t public_state_id,
    const PublicStateRow& row,
    int player,
    const PrivateCards& private_cards,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(row.action_count, 0.0);
  if (row.action_count == 0) {
    return;
  }

  const double uniform_probability = 1.0 / row.action_count;
  const PrivateBucketId private_bucket =
      card_abstraction_.private_bucket(private_cards.combo, row.state);
  const InfoSetRow* info_set_row =
      find_info_set_row({public_state_id, player, private_bucket});
  if (info_set_row == nullptr) {
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return;
  }

  average_strategy_probabilities(
      *info_set_row, row, uniform_probability, probabilities);
}

void CFRSolver::average_strategy_probabilities(
    const InfoSetRow& row,
    const GameTree::Node& node,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  const int num_actions = node.action_count;
  probabilities.clear();
  probabilities.resize(num_actions, 0.0);
  double probability_sum = 0.0;
  const size_t action_offset = row.action_offset;
  const std::vector<int>& action_ids = strategy_action_ids();
  const std::vector<float>& cumulative_regrets =
      strategy_cumulative_regrets();
  const std::vector<float>& cumulative_strategies =
      strategy_cumulative_strategies();

  const bool aligned_action_ids =
      num_actions == row.action_count &&
      [&]() {
        for (int i = 0; i < num_actions; ++i) {
          if (node.actions[i].key != action_ids[action_offset + i]) return false;
        }
        return true;
      }();
  if (aligned_action_ids) {
    for (int i = 0; i < num_actions; ++i) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      const size_t table_index = action_offset + i;
      probabilities[i] =
          config_.regret_only_training
              ? std::max(0.0,
                         static_cast<double>(
                             AtomicFloatLoad(&cumulative_regrets[table_index])))
              : static_cast<double>(
                    AtomicFloatLoad(&cumulative_strategies[table_index]));
      probability_sum += probabilities[i];
    }
  } else {
    for (int legal_action_index = 0; legal_action_index < num_actions;
         ++legal_action_index) {
      const int legal_action_id = node.actions[legal_action_index].key;
      for (uint16_t action_index = 0; action_index < row.action_count;
           ++action_index) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const size_t table_index = action_offset + action_index;
        if (action_ids[table_index] == legal_action_id) {
          probabilities[legal_action_index] =
              config_.regret_only_training
                  ? std::max(
                        0.0,
                        static_cast<double>(
                            AtomicFloatLoad(&cumulative_regrets[table_index])))
                  : static_cast<double>(
                        AtomicFloatLoad(&cumulative_strategies[table_index]));
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

void CFRSolver::average_strategy_probabilities(
    const InfoSetRow& info_set_row,
    const PublicStateRow& public_state_row,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  const int num_actions = public_state_row.action_count;
  probabilities.clear();
  probabilities.resize(num_actions, 0.0);
  double probability_sum = 0.0;
  const size_t action_offset = info_set_row.action_offset;
  const std::vector<int>& action_ids = strategy_action_ids();
  const std::vector<float>& cumulative_regrets =
      strategy_cumulative_regrets();
  const std::vector<float>& cumulative_strategies =
      strategy_cumulative_strategies();

  const bool aligned_action_ids =
      num_actions == info_set_row.action_count &&
      [&]() {
        for (int i = 0; i < num_actions; ++i) {
          if (public_state_row.action_ids[static_cast<size_t>(i)] !=
              action_ids[action_offset + i]) {
            return false;
          }
        }
        return true;
      }();
  if (aligned_action_ids) {
    for (int i = 0; i < num_actions; ++i) {
      POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
      const size_t table_index = action_offset + i;
      probabilities[i] =
          config_.regret_only_training
              ? std::max(0.0,
                         static_cast<double>(
                             AtomicFloatLoad(&cumulative_regrets[table_index])))
              : static_cast<double>(
                    AtomicFloatLoad(&cumulative_strategies[table_index]));
      probability_sum += probabilities[i];
    }
  } else {
    for (int legal_action_index = 0; legal_action_index < num_actions;
         ++legal_action_index) {
      const int legal_action_id =
          public_state_row.action_ids[static_cast<size_t>(legal_action_index)];
      for (uint16_t action_index = 0;
           action_index < info_set_row.action_count; ++action_index) {
        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const size_t table_index = action_offset + action_index;
        if (action_ids[table_index] == legal_action_id) {
          probabilities[legal_action_index] =
              config_.regret_only_training
                  ? std::max(
                        0.0,
                        static_cast<double>(
                            AtomicFloatLoad(&cumulative_regrets[table_index])))
                  : static_cast<double>(
                        AtomicFloatLoad(&cumulative_strategies[table_index]));
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
    const CompactPublicState& state,
    uint32_t public_state_id,
    int player,
    const int* conditioned_action_ids,
    size_t action_count,
    ConditionedRanges& conditioned_ranges) {
  if (action_count == 0) {
    return;
  }

  if (conditioned_ranges.size() < action_count) {
    conditioned_ranges.resize(action_count);
  }
  for (size_t i = 0; i < action_count; ++i) {
    conditioned_ranges[i].reset_to_filtered();
  }
  const size_t range_size = range.size();
  if (range_size == 0) {
    return;
  }

  const double fallback_probability = 1.0 / action_count;
  const CardMask board_mask = state.board_mask;
  const PublicInfoSetSlab* public_slab =
      public_info_set_slab(public_state_id);
  const PublicInfoSetSlabPlayer* player_slab =
      public_slab != nullptr ? &public_slab->players[player] : nullptr;
  const auto& action_ids = strategy_action_ids();
  const auto& regrets = strategy_cumulative_regrets();
  double positive_regrets[GameTree::kMaxActionsPerNode] = {};
  for (size_t i = 0; i < range_size; ++i) {
    const float range_weight = range.weight(i);
    const ComboId combo_id = range.combo(i);
    if (range_weight <= 0.0 || (ComboMask(combo_id) & board_mask) != 0) {
      continue;
    }

    double positive_regret_sum = 0.0;
    const PrivateBucketId private_bucket =
        card_abstraction_.private_bucket(combo_id, state);
    const InfoSetRow* row = nullptr;
    if (player_slab != nullptr) {
      row = find_info_set_row(*player_slab, private_bucket);
    }

    if (row != nullptr) {
      const size_t table_offset = row->action_offset;
      std::fill(positive_regrets, positive_regrets + action_count, 0.0);
      const size_t info_set_action_count =
          std::min(action_count,
                   static_cast<size_t>(row->action_count));
      for (size_t action_index = 0; action_index < info_set_action_count;
           ++action_index) {
        const size_t table_index = table_offset + action_index;
        if (action_ids[table_index] !=
            conditioned_action_ids[action_index]) {
          continue;
        }

        POKER_RECORD_TRAVERSAL_STAT(++traversal_stats_.action_entry_touches);
        const double positive_regret =
            std::max(
                0.0,
                static_cast<double>(
                    AtomicFloatLoad(&regrets[table_index])));
        positive_regrets[action_index] = positive_regret;
        positive_regret_sum += positive_regret;
      }
    }

    for (size_t action_index = 0; action_index < action_count;
         ++action_index) {
      const double probability =
          positive_regret_sum > 0.0
              ? positive_regrets[action_index] / positive_regret_sum
              : fallback_probability;
      const double conditioned_weight = range_weight * probability;
      if (conditioned_weight > 0.0) {
        conditioned_ranges[action_index].add(
            combo_id, static_cast<float>(conditioned_weight));
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

namespace {

double StrategyWeight(const SolverConfig& config,
                      const std::vector<float>& cumulative_regrets,
                      const std::vector<float>& cumulative_strategies,
                      size_t table_index) {
  return config.regret_only_training
             ? std::max(0.0,
                        static_cast<double>(
                            AtomicFloatLoad(&cumulative_regrets[table_index])))
             : static_cast<double>(
                   AtomicFloatLoad(&cumulative_strategies[table_index]));
}

}  // namespace

CFRSolver::StrategyProfile CFRSolver::get_strategy_profile() const {
  const auto& slabs = strategy_public_info_set_slabs();
  const auto& action_ids = strategy_action_ids();
  const auto& regrets = strategy_cumulative_regrets();
  const auto& strategies = strategy_cumulative_strategies();

  StrategyProfile profile;
  profile.info_sets.reserve(get_info_set_count());

  auto export_row = [&](uint32_t public_state_id,
                        uint8_t player,
                        PrivateBucketId private_bucket,
                        const InfoSetRow& row) {
    double sum = 0.0;
    const size_t action_offset = row.action_offset;
    for (uint16_t action_index = 0; action_index < row.action_count;
         ++action_index) {
      const size_t table_index = action_offset + action_index;
      sum += StrategyWeight(config_, regrets, strategies, table_index);
    }

    StrategyInfoSet exported;
    exported.key.public_state_id = public_state_id;
    exported.key.private_bucket = private_bucket;
    exported.key.player = player;
    exported.action_ids.reserve(row.action_count);
    exported.probabilities.reserve(row.action_count);
    if (sum > 0.0) {
      for (uint16_t action_index = 0; action_index < row.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        exported.action_ids.push_back(action_ids[table_index]);
        exported.probabilities.push_back(
            StrategyWeight(config_, regrets, strategies, table_index) / sum);
      }
    } else if (row.action_count > 0) {
      double uniform_prob = 1.0 / row.action_count;
      for (uint16_t action_index = 0; action_index < row.action_count;
           ++action_index) {
        const size_t table_index = action_offset + action_index;
        exported.action_ids.push_back(action_ids[table_index]);
        exported.probabilities.push_back(uniform_prob);
      }
    }

    profile.info_sets.push_back(std::move(exported));
  };

  for (uint32_t public_state_id = 0; public_state_id < slabs.size();
       ++public_state_id) {
    const std::unique_ptr<PublicInfoSetSlab>& slab = slabs[public_state_id];
    if (slab == nullptr) {
      continue;
    }
    for (uint8_t player = 0; player < kPlayerCount; ++player) {
      const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
      for (size_t chunk_index = 0;
           chunk_index < player_slab.private_row_chunks.size();
           ++chunk_index) {
        const std::unique_ptr<PrivateRowChunk>& chunk =
            player_slab.private_row_chunks[chunk_index];
        if (chunk == nullptr) {
          continue;
        }
        for (size_t chunk_offset = 0; chunk_offset < chunk->rows.size();
             ++chunk_offset) {
          const PrivateBucketId private_bucket = static_cast<PrivateBucketId>(
              chunk_index * kPrivateBucketChunkSize + chunk_offset);
          if (private_bucket >= kComboCount) {
            break;
          }
          const int32_t row_id = chunk->rows[chunk_offset];
          if (row_id < 0 ||
              static_cast<size_t>(row_id) >= player_slab.rows.size()) {
            continue;
          }
          export_row(public_state_id, player, private_bucket,
                     player_slab.rows[row_id]);
        }
      }
    }
  }

  return profile;
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }
  return evaluate_strategy_node(*root_public_state_id,
                                PrivateCards::FromCombo(player_a_hand),
                                PrivateCards::FromCombo(player_b_hand));
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
  const std::optional<uint32_t> root_public_state_id =
      get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }

  if (samples < kParallelEvaluationSampleThreshold) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  ThreadPoolExecutor executor(worker_count);
  std::uniform_int_distribution<unsigned int> seed_distribution;
  std::vector<unsigned int> worker_seeds;
  worker_seeds.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    worker_seeds.push_back(seed_distribution(rng_));
  }

  SolverConfig config = config_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
  const StrategyTablesView strategy_tables = strategy_tables_view();
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, range_sampler,
                                       root_public_state_id =
                                           *root_public_state_id,
                                       &strategy_tables,
                                       continuation_value_provider,
                                       shard_samples, seed]() mutable {
      CFRSolver worker(config, std::make_shared<TerminalUtilityCache>(),
                       continuation_value_provider);
      worker.strategy_tables_view_ = &strategy_tables;
      worker.rng_.seed(seed);
      const double value = worker.evaluate_strategy_samples(
          shard_samples, root_public_state_id, range_sampler);
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
    uint32_t root_public_state_id,
    RangeSampler range_sampler) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    total += evaluate_strategy_node(
        root_public_state_id, PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo));
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    uint32_t public_state_id,
    const PrivateCards& player_a_cards,
    const PrivateCards& player_b_cards) {
  const auto& rows = strategy_public_state_rows();
  if (public_state_id >= rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = rows[public_state_id];

  if (row.is_terminal) {
    return utility(row.state, player_a_cards, player_b_cards);
  }
  if (row.is_chance_node) {
    const CompactPublicState state = row.state;
    const int samples = ChanceSamples(config_);
    double value = 0.0;
    int evaluated = 0;
    for (int i = 0; i < samples; ++i) {
      const auto cards = SampleStreetCards(
          state, player_a_cards.mask() | player_b_cards.mask(), rng_);
      std::optional<uint32_t> child_public_state_id =
          get_or_create_chance_child_public_state(public_state_id, cards);
      if (!child_public_state_id.has_value()) {
        continue;
      }
      value += evaluate_strategy_node(
          *child_public_state_id, player_a_cards, player_b_cards);
      ++evaluated;
    }
    return evaluated > 0 ? value / evaluated : 0.0;
  }
  if (row.action_count == 0) {
    return 0.0;
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player)) {
    return 0.0;
  }

  const PrivateCards& player_cards =
      player == 0 ? player_a_cards : player_b_cards;
  StrategyProbabilities probabilities;
  average_strategy_probabilities(
      public_state_id, row, player, player_cards, probabilities);

  double value = 0.0;
  const int action_count = row.action_count;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    std::optional<uint32_t> child_public_state_id =
        get_or_create_action_child_public_state(public_state_id, action_index);
    if (!child_public_state_id.has_value()) {
      continue;
    }
    value += probabilities[action_index] *
             evaluate_strategy_node(*child_public_state_id, player_a_cards,
                                    player_b_cards);
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
          cache_chance_betting_history_transition(node, child_node);
          return best_response_value(child_node, player_a_cards, player_b_cards,
                                     best_response_player);
        },
        false, config_.max_public_states);
  }
  if (node.action_count == 0) {
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
    for (int i = 0; i < node.action_count; ++i) {
      GameTree::Node* child_node = CachedActionChildOrNull(
          *game_tree_, node, i, nullptr, false, config_.max_public_states);
      if (child_node != nullptr) {
        cache_action_betting_history_transition(node, i, *child_node);
      }
      const double child_value =
          child_node != nullptr
              ? best_response_value(*child_node, player_a_cards,
                                    player_b_cards, best_response_player)
              : 0.0;
      value = std::max(value, child_value);
    }
    return value;
  }

  StrategyProbabilities probabilities;
  average_strategy_probabilities(node, player, player_cards, probabilities);
  double value = 0.0;
  for (int action_index = 0; action_index < node.action_count; ++action_index) {
    GameTree::Node* child_node = CachedActionChildOrNull(
        *game_tree_, node, action_index, nullptr,
        false, config_.max_public_states);
    if (child_node == nullptr) {
      continue;
    }
    cache_action_betting_history_transition(node, action_index, *child_node);
    value += probabilities[action_index] *
             best_response_value(*child_node, player_a_cards, player_b_cards,
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
    int evaluated = 0;
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
      const auto cards =
          SampleStreetCards(
              node.state, best_response_cards.mask() | sampled_opponent.mask(),
              rng_);
      GameTree::Node* child_node = CachedChanceChildOrNull(
          *game_tree_, node, cards, nullptr, false, config_.max_public_states);
      if (child_node == nullptr) {
        continue;
      }
      cache_chance_betting_history_transition(node, *child_node);
      WeightedHandRangeView child_opponents;
      child_opponents.reset_to_filtered(opponent_hands.source_range());
      child_opponents.add(sampled_opponent_index, 1.0);
      value += best_response_value_against_range(
          *child_node, best_response_cards, child_opponents,
          best_response_player);
      ++evaluated;
    }
    return evaluated > 0 ? value / evaluated : 0.0;
  }

  if (node.action_count == 0) {
    return 0.0;
  }

  int player = node.player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < node.action_count; ++i) {
      GameTree::Node* child_node = CachedActionChildOrNull(
          *game_tree_, node, i, nullptr, false, config_.max_public_states);
      if (child_node != nullptr) {
        cache_action_betting_history_transition(node, i, *child_node);
      }
      const double child_value =
          child_node != nullptr
              ? best_response_value_against_range(
                    *child_node, best_response_cards, opponent_hands,
                    best_response_player)
              : 0.0;
      value = std::max(value, child_value);
    }
    return value;
  }

  const double fallback_probability = 1.0 / node.action_count;
  const std::optional<uint32_t> public_state_id =
      strategy_public_state_id(node);

  absl::InlinedVector<WeightedHandRangeView, 8> child_opponents;
  child_opponents.resize(node.action_count);
  for (WeightedHandRangeView& child_opponent : child_opponents) {
    child_opponent.reset_to_filtered(opponent_hands.source_range());
    child_opponent.reserve(opponent_hands.size());
  }

  const PublicInfoSetSlab* public_slab =
      public_state_id.has_value() ? public_info_set_slab(*public_state_id)
                                  : nullptr;
  const PublicInfoSetSlabPlayer* player_slab =
      public_slab != nullptr ? &public_slab->players[player] : nullptr;
  StrategyProbabilities probabilities;
  probabilities.reserve(node.action_count);
  for (size_t i = 0; i < opponent_hands.size(); ++i) {
    probabilities.clear();
    probabilities.resize(node.action_count, fallback_probability);

    if (player_slab != nullptr) {
      const ComboId opponent_combo = opponent_hands.combo(i);
      const PrivateBucketId private_bucket =
          card_abstraction_.private_bucket(opponent_combo, node.state);
      if (const InfoSetRow* row =
              find_info_set_row(*player_slab, private_bucket)) {
        average_strategy_probabilities(*row, node, fallback_probability,
                                       probabilities);
      }
    }

    const size_t opponent_source_index = opponent_hands.source_index(i);
    const double opponent_weight = opponent_hands.weight(i);
    for (int action_index = 0; action_index < node.action_count;
         ++action_index) {
      const double probability = probabilities[action_index];
      if (probability > 0.0) {
        child_opponents[action_index].add(opponent_source_index,
                                          opponent_weight * probability);
      }
    }
  }

  double value = 0.0;
  for (int action_index = 0; action_index < node.action_count; ++action_index) {
    GameTree::Node* child_node = CachedActionChildOrNull(
        *game_tree_, node, action_index, nullptr,
        false, config_.max_public_states);
    if (child_node == nullptr) {
      continue;
    }
    cache_action_betting_history_transition(node, action_index, *child_node);

    double child_weight = TotalWeight(child_opponents[action_index]);
    if (child_weight > 0.0) {
      value += (child_weight / total_weight) *
               best_response_value_against_range(
                   *child_node, best_response_cards,
                   child_opponents[action_index],
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
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      continuation_value_provider_;
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
                                       continuation_value_provider,
                                       shard_samples, seed, best_response_player]() {
      CFRSolver worker(config, std::make_shared<TerminalUtilityCache>(),
                       continuation_value_provider);
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
      evaluate_strategy(player_a_hand, player_b_hand);
  double player_a_gap =
      best_response_value(root, player_a_cards, player_b_cards, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_cards, player_b_cards, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

GameAction CFRSolver::get_best_response_action(GameTree::Node& node,
                                               ComboId player_a_hand,
                                               ComboId player_b_hand,
                                               int best_response_player) {
  GameAction no_action;
  no_action.kind = ActionKind::kNoAction;
  if (node.is_terminal || node.is_chance_node ||
      node.action_count == 0 || node.player_to_act != best_response_player) {
    return no_action;
  }

  double best_value = -std::numeric_limits<double>::infinity();
  GameAction best_action = no_action;
  const PrivateCards player_a_cards = PrivateCards::FromCombo(player_a_hand);
  const PrivateCards player_b_cards = PrivateCards::FromCombo(player_b_hand);
  for (int i = 0; i < node.action_count; ++i) {
    const GameAction& action = node.actions[i].action;
    GameTree::Node* child_node = CachedActionChildOrNull(
        *game_tree_, node, i, nullptr, false, config_.max_public_states);
    if (child_node != nullptr) {
      cache_action_betting_history_transition(node, i, *child_node);
    }
    const double value =
        child_node != nullptr
            ? best_response_value(*child_node, player_a_cards, player_b_cards,
                                  best_response_player)
            : 0.0;
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  return best_action;
}

double CFRSolver::get_expected_value(int player_id) const {
  const int iters = iterations_run_.load(std::memory_order_relaxed);
  if (iters == 0) {
    return 0.0;
  }
  double player_a_ev = cumulative_root_utility_ / iters;
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

double CFRSolver::utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_count + 2 < 5) {
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

double CFRSolver::uncached_utility(const CompactPublicState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }

  if (state.board_count + 2 < 5) {
    return 0.0;
  }

  return game_tree_->get_utility(
      state, player_a_cards.combo, player_b_cards.combo);
}

void CFRSolver::update_strategy(const InfoSetRow& row,
                                const double* action_probabilities,
                                size_t action_count,
                                double reach_prob) {
  auto& strategies = mutable_strategy_cumulative_strategies();
  const size_t action_offset = row.action_offset;
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    POKER_RECORD_TRAVERSAL_STAT(traversal_stats_.action_entry_touches += 2);
    const float delta = static_cast<float>(
        reach_prob * action_probabilities[action_index]);
    AtomicFloatAdd(&strategies[action_offset + action_index], delta);
  }
}

} // namespace poker
