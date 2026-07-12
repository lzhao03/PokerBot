#pragma once

// EXPERIMENTAL: This compact policy format is unstable and is not used by
// production persistence. It exists only to measure encoding choices.

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "src/solver.h"

namespace poker {

absl::StatusOr<std::vector<uint8_t>> EncodeExperimentalPolicy(
    const Policy &policy, const HistoryTree &history);

}  // namespace poker
