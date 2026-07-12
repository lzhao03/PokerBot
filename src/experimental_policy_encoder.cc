#include "src/experimental_policy_encoder.h"

// EXPERIMENTAL: This format is intentionally isolated from SavePolicy. Its
// layout may change or be deleted after size and quality measurements.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace poker {
namespace {

using Row = std::pair<InfoSetKey, PolicyRow>;

bool KeyLess(const Row& left, const Row& right) {
  if (left.first.history != right.first.history) {
    return left.first.history < right.first.history;
  }
  if (left.first.public_observation != right.first.public_observation) {
    return left.first.public_observation < right.first.public_observation;
  }
  return left.first.private_observation < right.first.private_observation;
}

void AppendVarint(std::vector<uint8_t>& output, uint64_t value) {
  while (value >= 0x80) {
    output.push_back(
        static_cast<uint8_t>((value & uint64_t{0x7f}) | uint64_t{0x80}));
    value >>= 7;
  }
  output.push_back(static_cast<uint8_t>(value));
}

std::array<uint8_t, kMaxActionsPerNode> Quantize(
    absl::Span<const float> probabilities) {
  std::array<uint8_t, kMaxActionsPerNode> result = {};
  std::array<double, kMaxActionsPerNode> remainders = {};
  double total = 0.0;
  for (float probability : probabilities) {
    total += std::max(0.0f, probability);
  }

  uint16_t assigned = 0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const double probability =
        total > 0.0 ? std::max(0.0f, probabilities[action]) / total
                    : 1.0 / probabilities.size();
    const double scaled = probability * 255.0;
    result[action] = static_cast<uint8_t>(std::floor(scaled));
    remainders[action] = scaled - result[action];
    assigned = static_cast<uint16_t>(assigned + result[action]);
  }
  for (; assigned < 255; ++assigned) {
    const auto best = std::max_element(
        remainders.begin(), remainders.begin() + probabilities.size());
    ++result[static_cast<size_t>(best - remainders.begin())];
    *best = -1.0;
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<uint8_t>> EncodeExperimentalPolicy(
    const Policy& policy,
    const HistoryTree& history) {
  std::vector<Row> rows(policy.rows.begin(), policy.rows.end());
  std::sort(rows.begin(), rows.end(), KeyLess);

  std::vector<uint8_t> output = {'P', 'K', 'X', '1'};
  for (std::byte byte : policy.model.bytes) {
    output.push_back(std::to_integer<uint8_t>(byte));
  }
  AppendVarint(output, rows.size());

  InfoSetKey previous;
  bool first = true;
  for (const auto& [key, row] : rows) {
    if (key.history.index() >= history.nodes.size()) {
      return absl::InvalidArgumentError("policy contains invalid history");
    }
    const auto* node =
        std::get_if<DecisionNode>(&history.nodes[key.history.index()]);
    if (node == nullptr || row.action_count != node->edges.count ||
        row.action_count == 0 || row.action_count > kMaxActionsPerNode ||
        row.action_offset + row.action_count > policy.probabilities.size()) {
      return absl::InvalidArgumentError("policy row is invalid");
    }

    if (first || key.history != previous.history) {
      output.push_back(0);
      AppendVarint(output, key.history.value() - previous.history.value());
      AppendVarint(output, key.public_observation.value());
      AppendVarint(output, key.private_observation.value());
    } else if (key.public_observation != previous.public_observation) {
      output.push_back(1);
      AppendVarint(output, key.public_observation.value() -
                               previous.public_observation.value());
      AppendVarint(output, key.private_observation.value());
    } else {
      output.push_back(2);
      AppendVarint(output, key.private_observation.value() -
                               previous.private_observation.value());
    }

    const auto quantized = Quantize(absl::Span<const float>(
        policy.probabilities.data() + row.action_offset, row.action_count));
    output.insert(output.end(), quantized.begin(),
                  quantized.begin() + row.action_count - 1);
    previous = key;
    first = false;
  }
  return output;
}

}  // namespace poker
