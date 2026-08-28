#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "absl/status/status.h"

namespace poker {
StrategyLookup MakeStrategyLookup(const Policy& policy) {
  return [&policy](InfoSetKey key, std::span<float> output) { return policy.strategy(key, output); };
}

namespace {

bool LookupOrUniform(const StrategyLookup& lookup, InfoSetKey key, std::span<float> probabilities) {
  if (lookup(key, probabilities)) return true;
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(), 1.0f / static_cast<float>(probabilities.size()));
  }
  return false;
}

struct EvaluationCounters {
  std::array<uint64_t, kPlayerCount> lookups = {};
  std::array<uint64_t, kPlayerCount> missing = {};
  std::array<double, kPlayerCount> weighted_lookups = {};
  std::array<double, kPlayerCount> weighted_missing = {};
  absl::flat_hash_map<InfoSetKey, double> reach_by_info_set;
  bool measure_reach_coverage = false;
};

struct ProfileEstimate {
  ValueEstimate value;
  EvaluationCounters counters;
};

std::mt19937 MakeEvaluationRng(uint64_t seed) {
  const std::array<uint32_t, 2> words = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  std::seed_seq sequence(words.begin(), words.end());
  return std::mt19937(sequence);
}

ProfileEstimate EstimateProfile(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public,
    const StrategyLookup& player_a, const StrategyLookup& player_b, uint64_t samples,
    uint64_t seed, bool measure_reach_coverage = false, bool sample_actions = false) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const std::array<const StrategyLookup*, kPlayerCount> policies = {&player_a, &player_b};
  EvaluationCounters counters;
  counters.measure_reach_coverage = measure_reach_coverage;
  double mean = 0.0;
  double squared_error = 0.0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Deal deal = deals.sample(rng);
    TerminalEvaluator terminal_utility(deal.hands);
    const ObservedPosition initial_position = ObservedPosition::Observe(initial_public, deal);
    ChanceSampler sample_chance{solver_config.card_abstraction, deal, rng};
    auto evaluate = [&](auto&& self, HistoryId history_id,
                        const ObservedPosition& position,
                        std::array<double, kPlayerCount> reach) -> double {
      while (true) {
        const HistoryNode& node = history.nodes[Index(history_id)];
        if (IsTerminal(node.state)) {
          return terminal_utility(node.state, position.board(), Player::A);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          const HistoryId child = history.children[node.children_begin];
          const auto& child_state = history.nodes[Index(child)].state;
          double value = 0.0;
          for (int draw = 0; draw < solver_config.chance_samples; ++draw) {
            const auto next = sample_chance(position, *chance, child_state);
            value += self(self, child, next, reach);
          }
          return value / solver_config.chance_samples;
        }

        const Player actor = std::get<DecisionState>(node.state).actor;
        const size_t player = Index(actor);
        const uint8_t action_count = node.child_count;
        const InfoSetKey key = position.info_set_key(history_id, actor);
        const double decision_reach = reach[0] * reach[1];
        if (counters.measure_reach_coverage && decision_reach > 0.0) {
          counters.reach_by_info_set[key] += decision_reach;
        }
        ++counters.lookups[player];
        counters.weighted_lookups[player] += decision_reach;
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        if (!LookupOrUniform(*policies[player], key, strategy)) {
          ++counters.missing[player];
          counters.weighted_missing[player] += decision_reach;
        }

        if (sample_actions) {
          float action_sample = std::uniform_real_distribution<float>{}(rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count && action_sample >= probabilities[sampled_action]) {
            action_sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history_id = history.children[node.children_begin + sampled_action];
          continue;
        }

        double value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          auto child_reach = reach;
          child_reach[player] *= probabilities[action];
          const HistoryId child = history.children[node.children_begin + action];
          value += probabilities[action] * self(self, child, position, child_reach);
        }
        return value;
      }
    };
    const double value = evaluate(evaluate, HistoryId{}, initial_position, {1.0, 1.0});
    const double delta = value - mean;
    mean += delta / static_cast<double>(sample + 1);
    squared_error += delta * (value - mean);
  }
  const double sample_count = static_cast<double>(samples);
  const double standard_error =
      samples > 1 ? std::sqrt(squared_error / (sample_count - 1.0) / sample_count) : 0.0;
  const uint64_t lookups = counters.lookups[0] + counters.lookups[1];
  const uint64_t missing = counters.missing[0] + counters.missing[1];
  const double weighted_lookups = counters.weighted_lookups[0] + counters.weighted_lookups[1];
  const double weighted_missing = counters.weighted_missing[0] + counters.weighted_missing[1];
  std::vector<double> reaches;
  reaches.reserve(counters.reach_by_info_set.size());
  for (const auto& [key, reach] : counters.reach_by_info_set) {
    (void)key;
    reaches.push_back(reach);
  }
  counters.reach_by_info_set = {};
  std::ranges::sort(reaches, std::greater<>());
  size_t rows_for_99_percent = 0;
  double covered_reach = 0.0;
  while (rows_for_99_percent < reaches.size() && covered_reach < 0.99 * weighted_lookups) {
    covered_reach += reaches[rows_for_99_percent++];
  }
  return {{mean, standard_error, lookups, missing, weighted_lookups, weighted_missing, reaches.size(),
           rows_for_99_percent},
          counters};
}

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config, const DealDistribution& deals, const HistoryTree& history,
    const PublicPosition& initial_public, const StrategyLookup& player_a,
    const StrategyLookup& player_b, uint64_t samples, uint64_t seed, bool measure_reach_coverage,
    bool sample_actions) {
  if (samples == 0) return absl::InvalidArgumentError("evaluation samples must be positive");
  return EstimateProfile(config, deals, history, initial_public, player_a, player_b, samples, seed,
                         measure_reach_coverage, sample_actions).value;
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config, const DealDistribution& deals, const HistoryTree& history,
    const PublicPosition& initial_public, ModelFingerprint model, const Policy& player_a,
    const Policy& player_b, uint64_t samples, uint64_t seed, bool measure_reach_coverage,
    bool sample_actions) {
  if (player_a.model != model || player_b.model != model) {
    return absl::FailedPreconditionError("policy model does not match game");
  }
  return EstimateExpectedValue(
      config, deals, history, initial_public, MakeStrategyLookup(player_a), MakeStrategyLookup(player_b),
      samples, seed, measure_reach_coverage, sample_actions);
}

namespace {

absl::StatusOr<BestResponseResult> TrainResponse(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    Player responder, const StrategyLookup& opponent, const StrategyLookup* responder_fallback,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError("best-response iteration counts must be positive");
  }
  InfoSetTable table(solver_config);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
  BestResponseResult result;
  for (uint64_t iteration = 0; iteration < config.training_iterations;
       ++iteration) {
    const Deal deal = deals.sample(rng);
    TerminalEvaluator terminal_utility(deal.hands);
    const ObservedPosition initial_position = ObservedPosition::Observe(initial_public, deal);
    ChanceSampler sample_chance{solver_config.card_abstraction, deal, rng};
    auto cfr = [&](auto&& self, HistoryId history_id,
                   const ObservedPosition& position,
                   std::array<double, kPlayerCount> reach) -> double {
      while (true) {
        const HistoryNode& node = history.nodes[Index(history_id)];
        if (IsTerminal(node.state)) {
          return terminal_utility(node.state, position.board(), responder);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          const HistoryId child = history.children[node.children_begin];
          const auto& child_state = history.nodes[Index(child)].state;
          double value = 0.0;
          for (int draw = 0; draw < solver_config.chance_samples; ++draw) {
            const auto next = sample_chance(position, *chance, child_state);
            value += self(self, child, next, reach);
          }
          return value / solver_config.chance_samples;
        }

        const Player player = std::get<DecisionState>(node.state).actor;
        const uint8_t action_count = node.child_count;
        const InfoSetKey key = position.info_set_key(history_id, player);
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        std::optional<uint32_t> offset;
        if (player == responder) {
          offset = table.find_or_create(key, action_count);
          if (offset || responder_fallback == nullptr) {
            table.strategy(table.regret_sum, offset, strategy);
          } else {
            LookupOrUniform(*responder_fallback, key, strategy);
          }
        } else {
          ++opponent_lookups;
          if (!LookupOrUniform(opponent, key, strategy)) ++missing_opponent_lookups;
        }

        if (config.external_sampling && player != responder) {
          float action_sample = std::uniform_real_distribution<float>{}(rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count && action_sample >= probabilities[sampled_action]) {
            action_sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history_id = history.children[node.children_begin + sampled_action];
          continue;
        }

        std::array<double, kMaxActionsPerNode> action_values;
        double node_value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          auto child_reach = reach;
          child_reach[Index(player)] *= probabilities[action];
          const HistoryId child = history.children[node.children_begin + action];
          action_values[action] = self(self, child, position, child_reach);
          node_value += probabilities[action] * action_values[action];
        }
        if (player != responder || !offset) return node_value;

        for (uint8_t action = 0; action < action_count; ++action) {
          double regret = action_values[action] - node_value;
          if (!config.external_sampling) regret *= reach[Index(Opponent(player))];
          table.add_regret(*offset, action, regret);
        }
        table.add_strategy(
            *offset, strategy,
            reach[Index(player)] * static_cast<double>(iteration + 1));
        return node_value;
      }
    };
    cfr(cfr, HistoryId{}, initial_position, {1.0, 1.0});
  }

  result.response_policy = ExtractAveragePolicy(table, history, model);
  const uint64_t evaluation_seed = config.seed ^ 0x9e3779b97f4a7c15ULL;
  const StrategyLookup response_lookup = [&result, responder_fallback](InfoSetKey key,
                                                                       std::span<float> output) {
    if (result.response_policy.strategy(key, output)) return true;
    return responder_fallback != nullptr && (*responder_fallback)(key, output);
  };
  const StrategyLookup& player_a = responder == Player::A ? response_lookup : opponent;
  const StrategyLookup& player_b = responder == Player::B ? response_lookup : opponent;
  const ProfileEstimate estimate = EstimateProfile(
      solver_config, deals, history, initial_public, player_a, player_b,
      config.evaluation_samples, evaluation_seed, false, config.external_sampling);
  result.value = responder == Player::A ? estimate.value.mean : -estimate.value.mean;
  result.standard_error = estimate.value.standard_error;
  const size_t opponent_index = Index(Opponent(responder));
  result.opponent_policy_lookups = opponent_lookups + estimate.counters.lookups[opponent_index];
  result.missing_opponent_lookups =
      missing_opponent_lookups + estimate.counters.missing[opponent_index];
  const size_t responder_index = Index(responder);
  result.response_policy_lookups = estimate.counters.lookups[responder_index];
  result.missing_response_lookups = estimate.counters.missing[responder_index];
  return result;
}

}  // namespace

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    Player responder, const StrategyLookup& opponent, const BestResponseConfig& config) {
  return TrainResponse(solver_config, deals, history, initial_public, model, responder, opponent,
                       nullptr, config);
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    Player responder, const Policy& opponent, const BestResponseConfig& config) {
  if (opponent.model != model) {
    return absl::FailedPreconditionError("opponent policy model does not match game");
  }
  return TrainApproximateBestResponse(
      solver_config, deals, history, initial_public, model, responder, MakeStrategyLookup(opponent),
      config);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    const StrategyLookup& policy, const BestResponseConfig& config) {
  auto player_a = TrainResponse(solver_config, deals, history, initial_public, model, Player::A,
                                policy, &policy, config);
  if (!player_a.ok()) return player_a.status();

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(solver_config, deals, history, initial_public, model, Player::B,
                                policy, &policy, player_b_config);
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = player_a->value + player_b->value;
  return ExploitabilityEstimate{std::move(*player_a), std::move(*player_b), nash_conv,
                                0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitabilityParallel(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    const StrategyLookupFactory& policy_factory, const BestResponseConfig& config) {
  if (!policy_factory) return absl::InvalidArgumentError("policy factory is empty");
  const std::array<StrategyLookup, kPlayerCount> policies = {policy_factory(), policy_factory()};
  if (!policies[0] || !policies[1])
    return absl::InvalidArgumentError("policy factory returned an empty lookup");
  std::optional<absl::StatusOr<BestResponseResult>> player_a;
  std::thread player_a_thread([&] {
    player_a.emplace(TrainResponse(
        solver_config, deals, history, initial_public, model, Player::A,
        policies[Index(Player::A)], &policies[Index(Player::A)], config));
  });

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(
      solver_config, deals, history, initial_public, model, Player::B, policies[Index(Player::B)],
      &policies[Index(Player::B)], player_b_config);
  player_a_thread.join();
  assert(player_a.has_value());
  if (!player_a->ok()) return player_a->status();
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = (*player_a)->value + player_b->value;
  return ExploitabilityEstimate{std::move(**player_a), std::move(*player_b), nash_conv,
                                0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    const Policy& policy, const BestResponseConfig& config) {
  if (policy.model != model) {
    return absl::FailedPreconditionError("policy model does not match game");
  }
  return EstimateExploitability(solver_config, deals, history, initial_public, model,
                                MakeStrategyLookup(policy), config);
}

}  // namespace poker
