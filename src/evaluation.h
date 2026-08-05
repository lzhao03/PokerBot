#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "absl/status/statusor.h"
#include "src/solver.h"

namespace poker {

// Returning false requests uniform fallback; callers always initialize output.
using StrategyLookup = std::function<bool(InfoSetKey, std::span<float>)>;
using StrategyLookupFactory = std::function<StrategyLookup()>;

StrategyLookup MakeStrategyLookup(const Policy& policy);
StrategyLookup MakeStrategyLookup(Policy&& policy) = delete;

struct ValueEstimate {
  double mean = 0.0;
  double standard_error = 0.0;
  uint64_t policy_lookups = 0;
  uint64_t missing_policy_lookups = 0;
  double weighted_policy_lookups = 0.0;
  double weighted_missing_policy_lookups = 0.0;
  size_t observed_info_sets = 0;
  size_t info_sets_for_99_percent_reach = 0;
};

struct BestResponseConfig {
  uint64_t training_iterations = 100'000;
  uint64_t evaluation_samples = 100'000;
  uint64_t seed = 1;
  bool external_sampling = false;
};

struct BestResponseResult {
  Policy response_policy;
  double value = 0.0;
  double standard_error = 0.0;
  uint64_t opponent_policy_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
  uint64_t response_policy_lookups = 0;
  uint64_t missing_response_lookups = 0;
};

struct ExploitabilityEstimate {
  BestResponseResult player_a_response;
  BestResponseResult player_b_response;
  double nash_conv = 0.0;
  double exploitability = 0.0;
};

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false,
    bool sample_actions = false);

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false,
    bool sample_actions = false);

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    Player responder,
    const StrategyLookup& opponent,
    const BestResponseConfig& config);

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const StrategyLookup& policy,
    const BestResponseConfig& config =
        BestResponseConfig{.external_sampling = true});

absl::StatusOr<ExploitabilityEstimate> EstimateExploitabilityParallel(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const StrategyLookupFactory& policy_factory,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const Policy& policy,
    const BestResponseConfig& config =
        BestResponseConfig{.external_sampling = true});

}  // namespace poker
