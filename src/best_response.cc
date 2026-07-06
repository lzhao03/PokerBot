#include "src/best_response.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/card_utils.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"

namespace poker {
namespace {

constexpr int kParallelBestResponseSampleThreshold = 32;

inline float AtomicFloatLoad(const float* src) {
  int32_t bits = __atomic_load_n(reinterpret_cast<const int32_t*>(src),
                                 __ATOMIC_RELAXED);
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

#ifndef POKER_ENABLE_TRAVERSAL_STATS
#define POKER_ENABLE_TRAVERSAL_STATS 1
#endif

#if POKER_ENABLE_TRAVERSAL_STATS
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
    statement;                                 \
  } while (false)
#else
#define POKER_RECORD_TRAVERSAL_STAT(statement) \
  do {                                         \
  } while (false)
#endif

int ChanceCardsKey(absl::Span<const CardId> cards) {
  int encoded[5];
  const int n = static_cast<int>(cards.size());
  for (int i = 0; i < n; ++i) {
    encoded[i] = EncodedCard(cards[i]);
  }
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

GameTree::Node* CachedChanceChildOrNull(GameTree& game_tree,
                                        GameTree::Node& node,
                                        absl::Span<const CardId> cards,
                                        int64_t* created_nodes,
                                        bool frozen = false,
                                        int max_public_states = 0) {
  const int child_key = ChanceCardsKey(cards);
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
      continue;
    }
    value += evaluate_child(*child_node);
    ++evaluated;
  }
  return evaluated > 0 ? value / evaluated : 0.0;
}

}  // namespace

void BestResponseEvaluator::average_strategy_probabilities(
    GameTree::Node& node,
    int player,
    ComboId private_combo,
    StrategyProbabilities& probabilities) {
  probabilities.clear();
  probabilities.resize(node.action_count, 0.0);
  if (node.action_count == 0) {
    return;
  }

  const double uniform_probability = 1.0 / node.action_count;
  const auto private_bucket =
      solver_.card_abstraction_.private_bucket(private_combo, node.state);
  auto try_public_state = [&](uint32_t public_state_id) {
    const StrategyTables::InfoSetRow* row =
        solver_.find_info_set_row({public_state_id, player, private_bucket});
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
    std::fill(probabilities.begin(), probabilities.end(),
              uniform_probability);
    return;
  }

  const std::optional<uint32_t> public_state_id =
      solver_.strategy_public_state_id(node);
  if (public_state_id.has_value()) {
    if (try_public_state(*public_state_id)) {
      return;
    }
  }

  std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
}

void BestResponseEvaluator::average_strategy_probabilities(
    const StrategyTables::InfoSetRow& row,
    const GameTree::Node& node,
    double fallback_probability,
    StrategyProbabilities& probabilities) {
  const int num_actions = node.action_count;
  probabilities.clear();
  probabilities.resize(num_actions, 0.0);
  double probability_sum = 0.0;
  const size_t action_offset = row.action_offset;
  const std::vector<int>& action_ids = solver_.tables_->action_ids;
  const std::vector<float>& cumulative_regrets =
      solver_.tables_->cumulative_regrets;
  const std::vector<float>& cumulative_strategies =
      solver_.tables_->cumulative_strategies;

  const bool aligned_action_ids =
      num_actions == row.action_count &&
      [&]() {
        for (int i = 0; i < num_actions; ++i) {
          if (node.actions[i].key != action_ids[action_offset + i]) {
            return false;
          }
        }
        return true;
      }();
  if (aligned_action_ids) {
    for (int i = 0; i < num_actions; ++i) {
      POKER_RECORD_TRAVERSAL_STAT(
          ++solver_.traversal_stats_.action_entry_touches);
      const size_t table_index = action_offset + i;
      probabilities[i] =
          solver_.config_.regret_only_training
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
        POKER_RECORD_TRAVERSAL_STAT(
            ++solver_.traversal_stats_.action_entry_touches);
        const size_t table_index = action_offset + action_index;
        if (action_ids[table_index] == legal_action_id) {
          probabilities[legal_action_index] =
              solver_.config_.regret_only_training
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
    std::fill(probabilities.begin(), probabilities.end(),
              fallback_probability);
    return;
  }
  for (double& probability : probabilities) {
    probability /= probability_sum;
  }
}

double BestResponseEvaluator::best_response_value(
    GameTree::Node& node,
    ComboId player_a_hand,
    ComboId player_b_hand,
    int best_response_player) {
  if (node.is_terminal) {
    double player_a_value = utility(node.state, player_a_hand, player_b_hand);
    return best_response_player == 0 ? player_a_value : -player_a_value;
  }
  if (node.is_chance_node) {
    return SampleChanceValue(
        *solver_.game_tree_, node,
        ComboMask(player_a_hand) | ComboMask(player_b_hand),
        ChanceSamples(solver_.config_), solver_.rng_, nullptr,
        [&](GameTree::Node& child_node) {
          solver_.cache_chance_betting_history_transition(node, child_node);
          return best_response_value(child_node, player_a_hand, player_b_hand,
                                     best_response_player);
        },
        false, solver_.config_.max_public_states);
  }
  if (node.action_count == 0) {
    return 0.0;
  }

  int player = node.player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const ComboId player_hand = player == 0 ? player_a_hand : player_b_hand;

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < node.action_count; ++i) {
      GameTree::Node* child_node = CachedActionChildOrNull(
          *solver_.game_tree_, node, i, nullptr, false,
          solver_.config_.max_public_states);
      if (child_node != nullptr) {
        solver_.cache_action_betting_history_transition(node, i, *child_node);
      }
      const double child_value =
          child_node != nullptr
              ? best_response_value(*child_node, player_a_hand, player_b_hand,
                                    best_response_player)
              : 0.0;
      value = std::max(value, child_value);
    }
    return value;
  }

  StrategyProbabilities probabilities;
  average_strategy_probabilities(node, player, player_hand, probabilities);
  double value = 0.0;
  for (int action_index = 0; action_index < node.action_count; ++action_index) {
    GameTree::Node* child_node = CachedActionChildOrNull(
        *solver_.game_tree_, node, action_index, nullptr,
        false, solver_.config_.max_public_states);
    if (child_node == nullptr) {
      continue;
    }
    solver_.cache_action_betting_history_transition(node, action_index,
                                                    *child_node);
    value += probabilities[action_index] *
             best_response_value(*child_node, player_a_hand, player_b_hand,
                                 best_response_player);
  }
  return value;
}

double BestResponseEvaluator::best_response_value_against_range(
    GameTree::Node& node,
    ComboId best_response_hand,
    const WeightedHandRangeView& opponent_hands,
    int best_response_player) {
  double total_weight = TotalWeight(opponent_hands);
  if (total_weight <= 0.0) {
    return 0.0;
  }

  if (node.is_terminal) {
    double value = 0.0;
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      const ComboId opponent_hand = opponent_hands.combo(i);
      const ComboId player_a_hand =
          best_response_player == 0 ? best_response_hand : opponent_hand;
      const ComboId player_b_hand =
          best_response_player == 0 ? opponent_hand : best_response_hand;
      double player_a_value =
          utility(node.state, player_a_hand, player_b_hand);
      value += opponent_hands.weight(i) *
               (best_response_player == 0 ? player_a_value : -player_a_value);
    }
    return value / total_weight;
  }

  if (node.is_chance_node) {
    double value = 0.0;
    int evaluated = 0;
    int samples = ChanceSamples(solver_.config_);
    std::vector<double> opponent_weights;
    opponent_weights.reserve(opponent_hands.size());
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      opponent_weights.push_back(opponent_hands.weight(i));
    }
    std::discrete_distribution<size_t> opponent_distribution(
        opponent_weights.begin(), opponent_weights.end());
    for (int i = 0; i < samples; ++i) {
      size_t sampled_opponent_view_index = opponent_distribution(solver_.rng_);
      size_t sampled_opponent_index =
          opponent_hands.source_index(sampled_opponent_view_index);
      const ComboId sampled_opponent =
          opponent_hands.source_range().combos[sampled_opponent_index];
      const auto cards =
          SampleStreetCards(node.state,
                            ComboMask(best_response_hand) |
                                ComboMask(sampled_opponent),
                            solver_.rng_);
      GameTree::Node* child_node = CachedChanceChildOrNull(
          *solver_.game_tree_, node, cards, nullptr, false,
          solver_.config_.max_public_states);
      if (child_node == nullptr) {
        continue;
      }
      solver_.cache_chance_betting_history_transition(node, *child_node);
      WeightedHandRangeView child_opponents;
      child_opponents.reset_to_filtered(opponent_hands.source_range());
      child_opponents.add(sampled_opponent_index, 1.0);
      value += best_response_value_against_range(
          *child_node, best_response_hand, child_opponents,
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
          *solver_.game_tree_, node, i, nullptr, false,
          solver_.config_.max_public_states);
      if (child_node != nullptr) {
        solver_.cache_action_betting_history_transition(node, i, *child_node);
      }
      const double child_value =
          child_node != nullptr
              ? best_response_value_against_range(
                    *child_node, best_response_hand, opponent_hands,
                    best_response_player)
              : 0.0;
      value = std::max(value, child_value);
    }
    return value;
  }

  const double fallback_probability = 1.0 / node.action_count;
  const std::optional<uint32_t> public_state_id =
      solver_.strategy_public_state_id(node);

  absl::InlinedVector<WeightedHandRangeView, 8> child_opponents;
  child_opponents.resize(node.action_count);
  for (WeightedHandRangeView& child_opponent : child_opponents) {
    child_opponent.reset_to_filtered(opponent_hands.source_range());
    child_opponent.reserve(opponent_hands.size());
  }

  const StrategyTables::PublicInfoSetSlab* public_slab =
      public_state_id.has_value() ? solver_.public_info_set_slab(*public_state_id)
                                  : nullptr;
  const StrategyTables::PublicInfoSetSlabPlayer* player_slab =
      public_slab != nullptr ? &public_slab->players[player] : nullptr;
  StrategyProbabilities probabilities;
  probabilities.reserve(node.action_count);
  for (size_t i = 0; i < opponent_hands.size(); ++i) {
    probabilities.clear();
    probabilities.resize(node.action_count, fallback_probability);

    if (player_slab != nullptr) {
      const ComboId opponent_combo = opponent_hands.combo(i);
      const auto private_bucket =
          solver_.card_abstraction_.private_bucket(opponent_combo, node.state);
      if (const StrategyTables::InfoSetRow* row =
              solver_.find_info_set_row(*player_slab, private_bucket)) {
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
        *solver_.game_tree_, node, action_index, nullptr,
        false, solver_.config_.max_public_states);
    if (child_node == nullptr) {
      continue;
    }
    solver_.cache_action_betting_history_transition(node, action_index,
                                                    *child_node);

    double child_weight = TotalWeight(child_opponents[action_index]);
    if (child_weight > 0.0) {
      value += (child_weight / total_weight) *
               best_response_value_against_range(
                   *child_node, best_response_hand,
                   child_opponents[action_index],
                   best_response_player);
    }
  }
  return value;
}

double BestResponseEvaluator::sampled_range_best_response_value(
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
    worker_seeds.push_back(seed_distribution(solver_.rng_));
  }

  SolverConfig config = solver_.config_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider =
      solver_.continuation_value_provider_;
  std::shared_ptr<StrategyTables> strategy_tables = solver_.tables_;
  std::vector<std::future<std::pair<double, int64_t>>> futures;
  futures.reserve(worker_count);
  int samples_remaining = samples;
  for (int i = 0; i < worker_count; ++i) {
    int shard_samples = samples_remaining / (worker_count - i);
    samples_remaining -= shard_samples;
    unsigned int seed = worker_seeds[i];
    futures.push_back(executor.submit([config, &best_response_hands,
                                       &opponent_hands, strategy_tables,
                                       continuation_value_provider,
                                       shard_samples, seed, best_response_player]() {
      CFRSolver worker(config, std::make_shared<TerminalUtilityCache>(),
                       continuation_value_provider);
      worker.tables_ = strategy_tables;
      worker.frozen_ = true;
      worker.rng_.seed(seed);
      BestResponseEvaluator evaluator(worker);
      const double value = evaluator.sampled_range_best_response_samples(
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
        solver_.traversal_stats_.action_entry_touches += result.second);
  }
  return total / samples;
}

double BestResponseEvaluator::sampled_range_best_response_samples(
    int samples,
    const WeightedHandRange& best_response_hands,
    const WeightedHandRange& opponent_hands,
    int best_response_player) {
  if (samples <= 0) {
    return 0.0;
  }

  std::discrete_distribution<size_t> best_response_hand_distribution(
      best_response_hands.weights.begin(), best_response_hands.weights.end());
  GameTree::Node& root = solver_.get_or_build_root();
  WeightedHandRangeView opponent_view(opponent_hands);
  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    const ComboId best_response_hand =
        best_response_hands.combos[best_response_hand_distribution(
            solver_.rng_)];
    WeightedHandRangeView compatible_opponents =
        CompatibleHands(opponent_view, ComboMask(best_response_hand),
                        root.state);
    if (compatible_opponents.empty()) {
      throw std::invalid_argument(
          "Could not sample non-overlapping hands from ranges");
    }
    total += best_response_value_against_range(root, best_response_hand,
                                               compatible_opponents,
                                               best_response_player);
  }
  return total / samples;
}

double BestResponseEvaluator::calculate_exploitability() {
  return calculate_exploitability(1);
}

double BestResponseEvaluator::calculate_exploitability(int samples) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<CardId> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), solver_.rng_);
    ComboId player_a_hand = CardsToComboId(deck.back(), deck[deck.size() - 2]);
    deck.pop_back();
    deck.pop_back();
    ComboId player_b_hand = CardsToComboId(deck.back(), deck[deck.size() - 2]);
    total += calculate_exploitability(player_a_hand, player_b_hand);
  }
  return total / samples;
}

double BestResponseEvaluator::calculate_exploitability(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  double strategy_player_a_value =
      solver_.evaluate_strategy(samples, player_a_range, player_b_range);
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

double BestResponseEvaluator::calculate_player_a_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return sampled_range_best_response_value(samples, player_a_range,
                                           player_b_range, 0);
}

double BestResponseEvaluator::calculate_player_b_best_response_value(
    int samples,
    const HandRange& player_a_range,
    const HandRange& player_b_range) {
  return sampled_range_best_response_value(samples, player_b_range,
                                           player_a_range, 1);
}

double BestResponseEvaluator::calculate_exploitability(
    ComboId player_a_hand,
    ComboId player_b_hand) {
  GameTree::Node& root = solver_.get_or_build_root();
  double strategy_player_a_value =
      solver_.evaluate_strategy(player_a_hand, player_b_hand);
  double player_a_gap =
      best_response_value(root, player_a_hand, player_b_hand, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_hand, player_b_hand, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

GameAction BestResponseEvaluator::get_best_response_action(
    GameTree::Node& node,
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
  for (int i = 0; i < node.action_count; ++i) {
    const GameAction& action = node.actions[i].action;
    GameTree::Node* child_node = CachedActionChildOrNull(
        *solver_.game_tree_, node, i, nullptr, false,
        solver_.config_.max_public_states);
    if (child_node != nullptr) {
      solver_.cache_action_betting_history_transition(node, i, *child_node);
    }
    const double value =
        child_node != nullptr
            ? best_response_value(*child_node, player_a_hand, player_b_hand,
                                  best_response_player)
            : 0.0;
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  return best_action;
}

double BestResponseEvaluator::utility(const GameState& state,
                                      ComboId player_a_hand,
                                      ComboId player_b_hand) {
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

  return solver_.utility_cache_->get_or_compute(
      state, player_a_hand, player_b_hand, [&]() {
        return solver_.game_tree_->get_utility(
            state, player_a_hand, player_b_hand);
      });
}

}  // namespace poker
