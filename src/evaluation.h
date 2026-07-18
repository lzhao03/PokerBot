#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "absl/status/statusor.h"
#include "src/solver.h"

namespace poker {

using StrategyLookup =
    std::function<bool(InfoSetKey, std::span<float>)>;

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
    const CompiledGame& game,
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false);

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false);

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const StrategyLookup& opponent,
    const BestResponseConfig& config);

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const StrategyLookup& policy,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitabilityParallel(
    const CompiledGame& game,
    const std::array<StrategyLookup, kPlayerCount>& policies,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const Policy& policy,
    const BestResponseConfig& config);

}  // namespace poker
