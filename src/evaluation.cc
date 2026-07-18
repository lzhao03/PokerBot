#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "absl/status/status.h"
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

StrategyLookup LookupPolicy(const Policy& policy) {
  return [&policy](InfoSetKey key, std::span<float> output) {
    return policy.strategy(key, output);
  };
}

StrategyLookup LookupPolicy(const CompiledGame& game,
                            const NeuralPolicy& policy) {
  constexpr size_t kCacheCapacity = 1'000'000;
  auto cache = std::make_shared<
      absl::flat_hash_map<InfoSetKey, NeuralActionVector>>();
  cache->reserve(100'000);
  return [&game, &policy, cache](InfoSetKey key, std::span<float> output) {
    const auto found = cache->find(key);
    if (found != cache->end()) {
      std::copy_n(found->second.begin(), output.size(), output.begin());
      return true;
    }
    NeuralActionVector probabilities = {};
    policy.strategy(
        game, key,
        std::span<float>(probabilities.data(), output.size()));
    std::copy_n(probabilities.begin(), output.size(), output.begin());
    if (cache->size() < kCacheCapacity) {
      cache->emplace(key, probabilities);
    }
    return true;
  };
}

struct PolicyBackend {
  using UpdateHandle = size_t;

  const std::array<const StrategyLookup*, kPlayerCount>& policies;
  EvaluationCounters& counters;

  std::optional<size_t> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess,
      std::span<float> probabilities) {
    const size_t player = Index(decision.state.actor);
    const double reach = decision.reaches[0] * decision.reaches[1];
    if (counters.measure_reach_coverage && reach > 0.0) {
      counters.reach_by_info_set[decision.key] += reach;
    }
    ++counters.lookups[player];
    counters.weighted_lookups[player] += reach;
    if (!(*policies[player])(decision.key, probabilities)) {
      ++counters.missing[player];
      counters.weighted_missing[player] += reach;
    }
    return std::nullopt;
  }

  void average_strategy(const internal::DecisionView& decision,
                        std::span<float> probabilities) {
    current_strategy(decision, internal::StrategyAccess::ReadOnly,
                     probabilities);
  }

  void record_regrets(const internal::DecisionView&,
                      size_t,
                      std::span<const float>) {}
  void record_strategy(const internal::DecisionView&,
                       size_t,
                       std::span<const float>,
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
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false,
    bool sample_actions = false) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const std::array<const StrategyLookup*, kPlayerCount> policies = {
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
        .external_sampling = sample_actions,
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
  using UpdateHandle = uint32_t;

  Player responder;
  const StrategyLookup& opponent;
  CfrState& response;
  const StrategyLookup* responder_fallback = nullptr;
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;

  std::optional<uint32_t> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess access,
      std::span<float> probabilities) {
    if (decision.state.actor != responder) {
      ++opponent_lookups;
      if (!opponent(decision.key, probabilities)) {
        ++missing_opponent_lookups;
      }
      return std::nullopt;
    }
    std::optional<uint32_t> offset = response.find(decision.key);
    if (!offset && access == internal::StrategyAccess::Writable) {
      offset = response.find_or_create(decision.key, decision.action_count);
    }
    if (offset || responder_fallback == nullptr) {
      response.strategy(response.regret_sum, offset, probabilities);
    } else {
      (*responder_fallback)(decision.key, probabilities);
    }
    return access == internal::StrategyAccess::Writable ? offset
                                                        : std::nullopt;
  }

  void average_strategy(const internal::DecisionView& decision,
                        std::span<float> probabilities) {
    current_strategy(decision, internal::StrategyAccess::ReadOnly,
                     probabilities);
  }

  void record_regrets(const internal::DecisionView&,
                      uint32_t offset,
                      std::span<const float> regrets) {
    for (size_t action = 0; action < regrets.size(); ++action) {
      response.add_regret(offset, action, regrets[action]);
    }
  }

  void record_strategy(const internal::DecisionView&,
                       uint32_t offset,
                       std::span<const float> probabilities,
                       double weight) {
    response.add_strategy(offset, probabilities, weight);
  }
};

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (samples == 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  return EstimateProfile(game, player_a, player_b, samples, seed,
                         measure_reach_coverage, sample_actions).value;
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const NeuralPolicy& player_a,
    const NeuralPolicy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (player_a.model() != game.model || player_b.model() != game.model) {
    return absl::FailedPreconditionError(
        "neural policy model does not match game");
  }
  return EstimateExpectedValue(
      game, LookupPolicy(game, player_a), LookupPolicy(game, player_b),
      samples, seed, measure_reach_coverage, sample_actions);
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (player_a.model != game.model || player_b.model != game.model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateExpectedValue(
      game, LookupPolicy(player_a), LookupPolicy(player_b),
      samples, seed, measure_reach_coverage, sample_actions);
}

namespace {

absl::StatusOr<BestResponseResult> TrainResponse(
    const CompiledGame& game,
    Player responder,
    const StrategyLookup& opponent,
    const StrategyLookup* responder_fallback,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError(
        "best-response iteration counts must be positive");
  }
  CfrState response_state(game.config, game.history.nodes.size(), true);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  ResponseBackend backend{
      responder, opponent, response_state, responder_fallback};
  SolverStats stats;
  BestResponseResult result;
  while (response_state.iterations < config.training_iterations) {
    const Deal deal = game.deals.sample(rng);
    internal::TraversalContext context{
        .deal = deal,
        .mode = internal::TraversalMode::Train,
        .update_player = responder,
        .iteration = response_state.iterations,
        .external_sampling = config.external_sampling,
        .rng = rng,
        .stats = stats,
        .accumulate_update_strategy = config.external_sampling,
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
  const StrategyLookup response_lookup = [&result, responder_fallback](
      InfoSetKey key, std::span<float> output) {
    if (result.response_policy.strategy(key, output)) return true;
    if (responder_fallback != nullptr) {
      (*responder_fallback)(key, output);
    }
    return false;
  };
  const StrategyLookup& player_a =
      responder == Player::A ? response_lookup : opponent;
  const StrategyLookup& player_b =
      responder == Player::B ? response_lookup : opponent;
  const ProfileEstimate estimate = EstimateProfile(
      game, player_a, player_b, config.evaluation_samples, evaluation_seed,
      false, config.external_sampling);
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
  const size_t responder_index = Index(responder);
  result.response_policy_lookups =
      estimate.counters.lookups[responder_index];
  result.missing_response_lookups =
      estimate.counters.missing[responder_index];
  return result;
}

}  // namespace

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const StrategyLookup& opponent,
    const BestResponseConfig& config) {
  return TrainResponse(game, responder, opponent, nullptr, config);
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config) {
  if (opponent.model != game.model) {
    return absl::FailedPreconditionError(
        "opponent policy model does not match game");
  }
  return TrainApproximateBestResponse(
      game, responder, LookupPolicy(opponent), config);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const StrategyLookup& policy,
    const BestResponseConfig& config) {
  auto player_a = TrainResponse(
      game, Player::A, policy, &policy, config);
  if (!player_a.ok()) return player_a.status();

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(
      game, Player::B, policy, &policy, player_b_config);
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = player_a->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(*player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitabilityParallel(
    const CompiledGame& game,
    const std::array<StrategyLookup, kPlayerCount>& policies,
    const BestResponseConfig& config) {
  std::optional<absl::StatusOr<BestResponseResult>> player_a;
  std::thread player_a_thread([&] {
    player_a.emplace(TrainResponse(
        game, Player::A, policies[Index(Player::A)],
        &policies[Index(Player::A)], config));
  });

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(
      game, Player::B, policies[Index(Player::B)],
      &policies[Index(Player::B)], player_b_config);
  player_a_thread.join();
  assert(player_a.has_value());
  if (!player_a->ok()) return player_a->status();
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = (*player_a)->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(**player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const Policy& policy,
    const BestResponseConfig& config) {
  if (policy.model != game.model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateExploitability(game, LookupPolicy(policy), config);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const NeuralPolicy& policy,
    const BestResponseConfig& config) {
  if (policy.model() != game.model) {
    return absl::FailedPreconditionError(
        "neural policy model does not match game");
  }
  return EstimateExploitability(game, LookupPolicy(game, policy), config);
}

}  // namespace poker
