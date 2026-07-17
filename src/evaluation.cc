#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "src/cfr_traversal.h"

namespace poker {
namespace {

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

struct PolicyBackend {
  using UpdateHandle = size_t;

  const std::array<const Policy*, kPlayerCount>& policies;
  EvaluationCounters& counters;

  std::optional<size_t> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess,
      absl::Span<float> probabilities) {
    const size_t player = Index(decision.state.actor);
    const double reach = decision.reaches[0] * decision.reaches[1];
    if (counters.measure_reach_coverage && reach > 0.0) {
      counters.reach_by_info_set[decision.key] += reach;
    }
    ++counters.lookups[player];
    counters.weighted_lookups[player] += reach;
    if (!policies[player]->strategy(decision.key, probabilities)) {
      ++counters.missing[player];
      counters.weighted_missing[player] += reach;
    }
    return std::nullopt;
  }

  void average_strategy(const internal::DecisionView& decision,
                        absl::Span<float> probabilities) {
    current_strategy(decision, internal::StrategyAccess::ReadOnly,
                     probabilities);
  }

  void record_regrets(const internal::DecisionView&,
                      size_t,
                      absl::Span<const float>) {}
  void record_strategy(const internal::DecisionView&,
                       size_t,
                       absl::Span<const float>,
                       double) {}
};

std::mt19937 MakeEvaluationRng(uint64_t seed) {
  const std::array<uint32_t, 2> words = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  std::seed_seq sequence(words.begin(), words.end());
  return std::mt19937(sequence);
}

ProfileEstimate EstimateProfile(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const std::array<const Policy*, kPlayerCount> policies = {
      &player_a, &player_b};
  EvaluationCounters counters;
  counters.measure_reach_coverage = measure_reach_coverage;
  PolicyBackend backend{policies, counters};
  SolverStats stats;
  double mean = 0.0;
  double squared_error = 0.0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Deal deal = game.deals.sample(rng);
    internal::TraversalContext context{
        .deal = deal,
        .mode = internal::TraversalMode::EvaluateCurrent,
        .update_player = Player::A,
        .iteration = 0,
        .external_sampling = false,
        .rng = rng,
        .stats = stats,
    };
    const double value = internal::Traverse(game, context, backend);
    const double delta = value - mean;
    mean += delta / (sample + 1);
    squared_error += delta * (value - mean);
  }
  const double standard_error = samples > 1
      ? std::sqrt(squared_error /
                  (samples - 1) / samples)
      : 0.0;
  const uint64_t lookups = counters.lookups[0] + counters.lookups[1];
  const uint64_t missing = counters.missing[0] + counters.missing[1];
  const double weighted_lookups =
      counters.weighted_lookups[0] + counters.weighted_lookups[1];
  const double weighted_missing =
      counters.weighted_missing[0] + counters.weighted_missing[1];
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
  while (rows_for_99_percent < reaches.size() &&
         covered_reach < 0.99 * weighted_lookups) {
    covered_reach += reaches[rows_for_99_percent++];
  }
  return {{mean, standard_error, lookups, missing,
           weighted_lookups, weighted_missing,
           reaches.size(), rows_for_99_percent},
          counters};
}

struct ResponseBackend {
  using UpdateHandle = size_t;

  Player responder;
  const Policy& opponent;
  CfrState& response;
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;

  std::optional<size_t> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess access,
      absl::Span<float> probabilities) {
    if (decision.state.actor != responder) {
      ++opponent_lookups;
      if (!opponent.strategy(decision.key, probabilities)) {
        ++missing_opponent_lookups;
      }
      return std::nullopt;
    }
    const std::optional<size_t> offset =
        access == internal::StrategyAccess::Writable
            ? response.find_or_create(decision.key, decision.action_count)
            : response.find(decision.key);
    response.strategy(response.regret_sum, offset, probabilities);
    return access == internal::StrategyAccess::Writable ? offset
                                                        : std::nullopt;
  }

  void average_strategy(const internal::DecisionView& decision,
                        absl::Span<float> probabilities) {
    current_strategy(decision, internal::StrategyAccess::ReadOnly,
                     probabilities);
  }

  void record_regrets(const internal::DecisionView&,
                      size_t offset,
                      absl::Span<const float> regrets) {
    for (size_t action = 0; action < regrets.size(); ++action) {
      response.add_regret(offset, action, regrets[action]);
    }
  }

  void record_strategy(const internal::DecisionView&,
                       size_t offset,
                       absl::Span<const float> probabilities,
                       double weight) {
    response.add_strategy(offset, probabilities, weight);
  }
};

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage) {
  if (samples == 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  if (player_a.model != game.model || player_b.model != game.model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateProfile(game, player_a, player_b, samples, seed,
                         measure_reach_coverage).value;
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError(
        "best-response iteration counts must be positive");
  }
  if (opponent.model != game.model) {
    return absl::FailedPreconditionError(
        "opponent policy model does not match game");
  }

  CfrState response_state(game.config, true);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  ResponseBackend backend{responder, opponent, response_state};
  SolverStats stats;
  BestResponseResult result;
  while (response_state.iterations < config.training_iterations) {
    const Deal deal = game.deals.sample(rng);
    internal::TraversalContext context{
        .deal = deal,
        .mode = internal::TraversalMode::Train,
        .update_player = responder,
        .iteration = response_state.iterations,
        .external_sampling = false,
        .rng = rng,
        .stats = stats,
    };
    const double value = internal::Traverse(game, context, backend);
    response_state.cumulative_root_utility += value;
    ++response_state.iterations;
  }

  auto response = ExtractAveragePolicy(
      response_state, game.history, game.model);
  if (!response.ok()) return response.status();
  result.response_policy = std::move(*response);
  const uint64_t evaluation_seed = config.seed ^ 0x9e3779b97f4a7c15ULL;
  const Policy& player_a = responder == Player::A
                               ? result.response_policy
                               : opponent;
  const Policy& player_b = responder == Player::B
                               ? result.response_policy
                               : opponent;
  const ProfileEstimate estimate = EstimateProfile(
      game, player_a, player_b, config.evaluation_samples, evaluation_seed);
  result.value = responder == Player::A
                     ? estimate.value.mean
                     : -estimate.value.mean;
  result.standard_error = estimate.value.standard_error;
  const size_t opponent_index = Index(Opponent(responder));
  result.opponent_policy_lookups =
      backend.opponent_lookups + estimate.counters.lookups[opponent_index];
  result.missing_opponent_lookups =
      backend.missing_opponent_lookups +
      estimate.counters.missing[opponent_index];
  return result;
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const Policy& policy,
    const BestResponseConfig& config) {
  auto player_a = TrainApproximateBestResponse(
      game, Player::A, policy, config);
  if (!player_a.ok()) return player_a.status();

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainApproximateBestResponse(
      game, Player::B, policy, player_b_config);
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = player_a->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(*player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

}  // namespace poker
