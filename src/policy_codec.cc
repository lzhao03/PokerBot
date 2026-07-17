#include "src/policy_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "src/bet_abstraction.h"

namespace poker {
namespace {

inline constexpr uint16_t kMaxProbabilityUnits = 256;
inline constexpr std::array<uint8_t, 8> kPolicyCodecMagic = {
    'P', 'K', 'C', 'O', 'D', 'E', 'C', '1'};

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

bool ValidConfig(PolicyCodecConfig config) noexcept {
  return config.total_units > 0 &&
         config.total_units <= kMaxProbabilityUnits && config.max_actions > 0 &&
         config.max_actions <= kMaxActionsPerNode;
}

template <typename Integer>
void AppendInteger(std::vector<uint8_t>& bytes, Integer value) {
  using Unsigned = std::make_unsigned_t<Integer>;
  const Unsigned bits = static_cast<Unsigned>(value);
  for (size_t index = 0; index < sizeof(Integer); ++index) {
    bytes.push_back(static_cast<uint8_t>(bits >> (index * 8)));
  }
}

void AppendVarint(std::vector<uint8_t>& bytes, uint64_t value) {
  while (value >= 0x80) {
    bytes.push_back(static_cast<uint8_t>(value) | 0x80);
    value >>= 7;
  }
  bytes.push_back(static_cast<uint8_t>(value));
}

class ByteReader {
 public:
  explicit ByteReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  template <typename Integer>
  std::optional<Integer> read() {
    if (sizeof(Integer) > remaining()) return std::nullopt;
    using Unsigned = std::make_unsigned_t<Integer>;
    Unsigned value = 0;
    for (size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Unsigned>(bytes_[offset_++]) << (index * 8);
    }
    return static_cast<Integer>(value);
  }

  std::optional<std::span<const uint8_t>> take(size_t size) {
    if (size > remaining()) return std::nullopt;
    const std::span<const uint8_t> result = bytes_.subspan(offset_, size);
    offset_ += size;
    return result;
  }

  std::optional<uint64_t> varint() {
    uint64_t value = 0;
    for (size_t shift = 0; shift < 64; shift += 7) {
      const auto byte = read<uint8_t>();
      if (!byte || (shift == 63 && *byte > 1)) return std::nullopt;
      value |= static_cast<uint64_t>(*byte & 0x7f) << shift;
      if ((*byte & 0x80) == 0) return value;
    }
    return std::nullopt;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

struct EncodedRow {
  InfoSetKey key;
  uint64_t code;
  uint8_t action_count;
};

absl::StatusOr<std::vector<EncodedRow>> QuantizedRows(
    const Policy& policy,
    PolicyCodecConfig config) {
  std::array<uint64_t, kMaxActionsPerNode + 1> default_codes = {};
  std::array<float, kMaxActionsPerNode> uniform;
  uniform.fill(1.0f);
  for (size_t actions = 1; actions <= config.max_actions; ++actions) {
    const auto code = EncodeActionProbabilities(
        std::span<const float>(uniform.data(), actions), config);
    if (!code.ok()) return code.status();
    default_codes[actions] = *code;
  }

  std::vector<std::pair<InfoSetKey, uint32_t>> rows(policy.rows.begin(),
                                                    policy.rows.end());
  std::ranges::sort(rows, {}, &std::pair<InfoSetKey, uint32_t>::second);
  if (rows.empty()) {
    if (policy.probabilities.empty()) return std::vector<EncodedRow>{};
    return absl::InvalidArgumentError("policy probabilities have no rows");
  }
  if (rows.front().second != 0) {
    return absl::InvalidArgumentError("policy rows are not contiguous");
  }

  std::vector<EncodedRow> encoded;
  encoded.reserve(rows.size());
  for (size_t index = 0; index < rows.size(); ++index) {
    const size_t begin = rows[index].second;
    const size_t end = index + 1 < rows.size() ? rows[index + 1].second
                                               : policy.probabilities.size();
    if (begin >= end || end > policy.probabilities.size() ||
        end - begin > config.max_actions) {
      return absl::InvalidArgumentError("invalid policy row span");
    }
    const std::span<const float> probabilities(
        policy.probabilities.data() + begin, end - begin);
    double total = 0.0;
    for (float probability : probabilities) {
      if (!std::isfinite(probability) || probability < 0.0f ||
          probability > 1.0f) {
        return absl::InvalidArgumentError("invalid action probability");
      }
      total += probability;
    }
    if (std::abs(total - 1.0) > 1e-5) {
      return absl::InvalidArgumentError("policy row is not normalized");
    }
    const auto code = EncodeActionProbabilities(probabilities, config);
    if (!code.ok()) return code.status();
    if (*code != default_codes[probabilities.size()]) {
      encoded.push_back({rows[index].first, *code,
                         static_cast<uint8_t>(probabilities.size())});
    }
  }
  return encoded;
}

absl::Status WriteBytes(const std::filesystem::path& path,
                        std::span<const uint8_t> bytes) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return absl::UnavailableError("could not open output file");
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return absl::DataLossError("could not write output file");
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return absl::UnavailableError("could not replace output file");
  }
  return absl::OkStatus();
}

}  // namespace

size_t ActionProbabilityCodeBits(size_t action_count,
                                 PolicyCodecConfig config) noexcept {
  const uint64_t count = DistributionCount(action_count, config);
  return count == 0 ? 0 : static_cast<size_t>(std::bit_width(count - 1));
}

absl::StatusOr<uint64_t> EncodeActionProbabilities(
    std::span<const float> probabilities,
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
    std::span<float> probabilities,
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

absl::StatusOr<std::vector<uint8_t>> EncodePolicy(
    const Policy& policy,
    PolicyCodecConfig config) {
  if (!ValidConfig(config)) {
    return absl::InvalidArgumentError("invalid policy codec configuration");
  }
  auto rows = QuantizedRows(policy, config);
  if (!rows.ok()) return rows.status();

  std::ranges::sort(*rows, [](const EncodedRow& left, const EncodedRow& right) {
    return std::tuple(left.action_count, std::to_underlying(left.key.history),
                      std::to_underlying(left.key.public_observation),
                      std::to_underlying(left.key.private_observation)) <
           std::tuple(right.action_count,
                      std::to_underlying(right.key.history),
                      std::to_underlying(right.key.public_observation),
                      std::to_underlying(right.key.private_observation));
  });

  std::vector<uint8_t> bytes(kPolicyCodecMagic.begin(),
                             kPolicyCodecMagic.end());
  AppendInteger(bytes, std::to_underlying(policy.model));
  AppendInteger(bytes, config.total_units);
  bytes.push_back(static_cast<uint8_t>(config.max_actions));

  // Each arity stores sorted (history, public, private, probability) varints.
  // A changed key prefix resets its lower fields; uniform rows are implicit.
  size_t row_begin = 0;
  for (size_t action_count = 2; action_count <= config.max_actions;
       ++action_count) {
    size_t row_end = row_begin;
    while (row_end < rows->size() &&
           (*rows)[row_end].action_count == action_count) {
      ++row_end;
    }
    if (row_end - row_begin > std::numeric_limits<uint32_t>::max()) {
      return absl::ResourceExhaustedError("too many compact policy rows");
    }
    AppendInteger(bytes, static_cast<uint32_t>(row_end - row_begin));
    uint32_t previous_history = 0;
    uint64_t previous_public = 0;
    uint32_t previous_private = 0;
    for (size_t index = row_begin; index < row_end; ++index) {
      const EncodedRow& row = (*rows)[index];
      const uint32_t history = std::to_underlying(row.key.history);
      const uint64_t public_observation =
          std::to_underlying(row.key.public_observation);
      const uint32_t private_observation =
          std::to_underlying(row.key.private_observation);
      const uint32_t history_delta = history - previous_history;
      if (history_delta != 0) previous_public = previous_private = 0;
      const uint64_t public_delta = public_observation - previous_public;
      if (public_delta != 0) previous_private = 0;
      AppendVarint(bytes, history_delta);
      AppendVarint(bytes, public_delta);
      AppendVarint(bytes, private_observation - previous_private);
      AppendVarint(bytes, row.code);
      previous_history = history;
      previous_public = public_observation;
      previous_private = private_observation;
    }
    row_begin = row_end;
  }
  return bytes;
}

absl::StatusOr<Policy> DecodePolicy(std::span<const uint8_t> bytes) {
  ByteReader reader(bytes);
  const auto magic = reader.take(kPolicyCodecMagic.size());
  const auto model = reader.read<uint64_t>();
  const auto total_units = reader.read<uint16_t>();
  const auto max_actions = reader.read<uint8_t>();
  if (!magic || !std::ranges::equal(*magic, kPolicyCodecMagic) || !model ||
      !total_units || !max_actions) {
    return absl::DataLossError("invalid compact policy header");
  }
  const PolicyCodecConfig config{.total_units = *total_units,
                                 .max_actions = *max_actions};
  if (!ValidConfig(config)) {
    return absl::DataLossError("invalid compact policy configuration");
  }

  Policy policy;
  policy.model = ModelFingerprint{*model};
  std::array<float, kMaxActionsPerNode> probabilities;
  for (size_t action_count = 2; action_count <= config.max_actions;
       ++action_count) {
    const auto row_count = reader.read<uint32_t>();
    if (!row_count || *row_count > reader.remaining() / 4) {
      return absl::DataLossError("invalid compact policy row count");
    }
    uint64_t history = 0;
    uint64_t public_observation = 0;
    uint64_t private_observation = 0;
    for (uint32_t index = 0; index < *row_count; ++index) {
      const auto history_delta = reader.varint();
      const auto public_delta = reader.varint();
      const auto private_delta = reader.varint();
      const auto code = reader.varint();
      if (!history_delta || !public_delta || !private_delta || !code ||
          *history_delta > std::numeric_limits<uint32_t>::max() - history) {
        return absl::DataLossError("invalid compact policy row");
      }
      history += *history_delta;
      if (*history_delta != 0) public_observation = private_observation = 0;
      if (*public_delta >
          std::numeric_limits<uint64_t>::max() - public_observation) {
        return absl::DataLossError("invalid compact policy row");
      }
      public_observation += *public_delta;
      if (*public_delta != 0) private_observation = 0;
      if (*private_delta >
              std::numeric_limits<uint32_t>::max() - private_observation ||
          (index > 0 && *history_delta == 0 && *public_delta == 0 &&
           *private_delta == 0)) {
        return absl::DataLossError("invalid compact policy row");
      }
      private_observation += *private_delta;
      const InfoSetKey key{
          PublicObservationId(public_observation),
          HistoryId(static_cast<uint32_t>(history)),
          PrivateObservationId(static_cast<uint32_t>(private_observation))};
      const std::span<float> output(probabilities.data(), action_count);
      const absl::Status decoded =
          DecodeActionProbabilities(*code, output, config);
      if (!decoded.ok()) return decoded;
      if (policy.probabilities.size() >
          std::numeric_limits<uint32_t>::max()) {
        return absl::ResourceExhaustedError("compact policy is too large");
      }
      const uint32_t offset =
          static_cast<uint32_t>(policy.probabilities.size());
      if (!policy.rows.try_emplace(key, offset).second) {
        return absl::DataLossError("duplicate compact policy row");
      }
      policy.probabilities.insert(policy.probabilities.end(), output.begin(),
                                  output.end());
    }
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError("trailing compact policy data");
  }
  return policy;
}

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path) {
  const auto bytes =
      EncodePolicy(policy, {.max_actions = kMaxActionsPerNode});
  return bytes.ok() ? WriteBytes(path, *bytes) : bytes.status();
}

absl::StatusOr<Policy> LoadPolicy(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) return absl::NotFoundError("policy file not found");
  const std::streampos end = input.tellg();
  if (end < 0 || end > std::numeric_limits<std::streamsize>::max()) {
    return absl::DataLossError("invalid policy file size");
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    return absl::DataLossError("failed to read policy file");
  }
  return DecodePolicy(bytes);
}

}  // namespace poker
