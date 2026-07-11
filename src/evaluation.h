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

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CFRSolver& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed);

}  // namespace poker
