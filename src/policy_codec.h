#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/solver.h"

namespace poker {

struct PolicyCodecConfig {
  uint16_t total_units = 16;
  size_t max_actions = 6;
};

size_t ActionProbabilityCodeBits(
    size_t action_count,
    PolicyCodecConfig config = {}) noexcept;

absl::StatusOr<uint64_t> EncodeActionProbabilities(
    absl::Span<const float> probabilities,
    PolicyCodecConfig config = {});

absl::Status DecodeActionProbabilities(
    uint64_t code,
    absl::Span<float> probabilities,
    PolicyCodecConfig config = {});

// Uniform rows are omitted and use Policy::strategy's existing fallback.
absl::StatusOr<std::vector<uint8_t>> EncodePolicy(
    const Policy& policy,
    PolicyCodecConfig config = {});

absl::StatusOr<Policy> DecodePolicy(absl::Span<const uint8_t> bytes);

}  // namespace poker
