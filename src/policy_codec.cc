#include "src/policy_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "src/bet_abstraction.h"

namespace poker {
namespace {

inline constexpr uint16_t kMaxProbabilityUnits = 256;

uint64_t Binomial(size_t n, size_t k) noexcept {
  if (k > n) return 0;
  k = std::min(k, n - k);
  uint64_t result = 1;
  for (size_t i = 1; i <= k; ++i) {
    result = result * static_cast<uint64_t>(n - k + i) / i;
  }
  return result;
}

uint64_t CompositionCount(size_t units, size_t parts) noexcept {
  return Binomial(units + parts - 1, parts - 1);
}

uint64_t DistributionCount(size_t action_count,
                           PolicyCodecConfig config) noexcept {
  if (action_count == 0 || action_count > config.max_actions ||
      config.max_actions > kMaxActionsPerNode || config.total_units == 0 ||
      config.total_units > kMaxProbabilityUnits) {
    return 0;
  }
  return CompositionCount(config.total_units, action_count);
}

}  // namespace

size_t ActionProbabilityCodeBits(size_t action_count,
                                 PolicyCodecConfig config) noexcept {
  const uint64_t count = DistributionCount(action_count, config);
  return count == 0 ? 0 : static_cast<size_t>(std::bit_width(count - 1));
}

absl::StatusOr<uint64_t> EncodeActionProbabilities(
    absl::Span<const float> probabilities,
    PolicyCodecConfig config) {
  const size_t action_count = probabilities.size();
  if (DistributionCount(action_count, config) == 0) {
    return absl::InvalidArgumentError(
        "invalid action count or probability units");
  }

  double total = 0.0;
  for (float probability : probabilities) {
    if (!std::isfinite(probability) || probability < 0.0f) {
      return absl::InvalidArgumentError("invalid action probability");
    }
    total += probability;
  }
  if (total <= 0.0) {
    return absl::InvalidArgumentError("action probabilities have no mass");
  }

  std::array<uint16_t, kMaxActionsPerNode> units = {};
  std::array<double, kMaxActionsPerNode> remainders = {};
  // Largest-remainder rounding keeps the requested total exact.
  size_t assigned = 0;
  for (size_t action = 0; action < action_count; ++action) {
    const double scaled =
        probabilities[action] * config.total_units / total;
    units[action] = static_cast<uint16_t>(scaled);
    assigned += units[action];
    remainders[action] = scaled - units[action];
  }
  for (; assigned < config.total_units; ++assigned) {
    const auto best =
        std::max_element(remainders.begin(), remainders.begin() + action_count);
    ++units[static_cast<size_t>(best - remainders.begin())];
    *best = -1.0;
  }

  // Rank lexicographically by counting the distributions skipped per action.
  uint64_t code = 0;
  size_t remaining = config.total_units;
  for (size_t action = 0; action + 1 < action_count; ++action) {
    const size_t remaining_actions = action_count - action - 1;
    for (size_t value = 0; value < units[action]; ++value) {
      code += CompositionCount(remaining - value, remaining_actions);
    }
    remaining -= units[action];
  }
  return code;
}

absl::Status DecodeActionProbabilities(
    uint64_t code,
    absl::Span<float> probabilities,
    PolicyCodecConfig config) {
  const size_t action_count = probabilities.size();
  const uint64_t count = DistributionCount(action_count, config);
  if (count == 0) {
    return absl::InvalidArgumentError(
        "invalid action count or probability units");
  }
  if (code >= count) return absl::DataLossError("invalid probability code");

  size_t remaining = config.total_units;
  for (size_t action = 0; action + 1 < action_count; ++action) {
    const size_t remaining_actions = action_count - action - 1;
    size_t units = 0;
    while (units < remaining) {
      const uint64_t skipped =
          CompositionCount(remaining - units, remaining_actions);
      if (code < skipped) break;
      code -= skipped;
      ++units;
    }
    probabilities[action] = static_cast<float>(units) / config.total_units;
    remaining -= units;
  }
  probabilities.back() = static_cast<float>(remaining) / config.total_units;
  return absl::OkStatus();
}

}  // namespace poker
