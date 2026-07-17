#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
    std::span<const float> probabilities,
    PolicyCodecConfig config = {});

absl::Status DecodeActionProbabilities(
    uint64_t code,
    std::span<float> probabilities,
    PolicyCodecConfig config = {});

// Uniform rows are omitted and use Policy::strategy's existing fallback.
absl::StatusOr<std::vector<uint8_t>> EncodePolicy(
    const Policy& policy,
    PolicyCodecConfig config = {});

absl::StatusOr<Policy> DecodePolicy(std::span<const uint8_t> bytes);

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path);

}  // namespace poker
