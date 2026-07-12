#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "src/solver.h"

namespace poker {

struct ValueEstimate {
  double mean = 0.0;
  double standard_error = 0.0;
  uint64_t samples = 0;
  uint64_t policy_lookups = 0;
  uint64_t missing_policy_lookups = 0;
};

struct BestResponseConfig {
  uint64_t training_iterations = 100'000;
  uint64_t evaluation_samples = 100'000;
  uint64_t seed = 1;
};

struct BestResponseResult {
  Player responder = Player::A;
  Policy response_policy;
  double value = 0.0;
  double standard_error = 0.0;
  uint64_t training_iterations_completed = 0;
  uint64_t opponent_policy_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
};

struct ExploitabilityEstimate {
  BestResponseResult player_a_response;
  BestResponseResult player_b_response;
  double nash_conv = 0.0;
  double exploitability = 0.0;
};

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CFRSolver& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed);

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CFRSolver& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CFRSolver& game,
    const Policy& policy,
    const BestResponseConfig& config);

}  // namespace poker
