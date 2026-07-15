#include "src/policy_codec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "doctest/doctest.h"
#include "src/bet_abstraction.h"

namespace poker {
namespace {

InfoSetKey Key(uint64_t public_observation,
               uint32_t history,
               uint32_t private_observation) {
  return {PublicObservationId(public_observation), HistoryId(history),
          PrivateObservationId(private_observation)};
}

void AddRow(Policy& policy,
            InfoSetKey key,
            std::initializer_list<float> probabilities) {
  policy.rows.try_emplace(key, policy.probabilities.size());
  policy.probabilities.insert(policy.probabilities.end(), probabilities);
}

std::vector<float> Quantized(
    std::initializer_list<float> probabilities,
    PolicyCodecConfig config = {}) {
  const auto code = EncodeActionProbabilities(probabilities, config);
  REQUIRE(code.ok());
  std::vector<float> output(probabilities.size());
  REQUIRE(DecodeActionProbabilities(*code, absl::MakeSpan(output), config)
              .ok());
  return output;
}

TEST_CASE("action probabilities use compact deterministic codes") {
  constexpr std::array<uint16_t, 3> kUnitTotals = {16, 64, 256};
  constexpr std::array<std::array<size_t, 9>, 3> kExpectedBits = {{
      {0, 0, 5, 8, 10, 13, 15, 17, 18},
      {0, 0, 7, 12, 16, 20, 24, 27, 31},
      {0, 0, 9, 16, 22, 28, 34, 39, 44},
  }};
  for (size_t units = 0; units < kUnitTotals.size(); ++units) {
    for (size_t actions = 1; actions < kExpectedBits[units].size(); ++actions) {
      CHECK(ActionProbabilityCodeBits(
                actions,
                {.total_units = kUnitTotals[units],
                 .max_actions = kMaxActionsPerNode}) ==
            kExpectedBits[units][actions]);
    }
  }
  CHECK(ActionProbabilityCodeBits(6) == 15);
  CHECK(ActionProbabilityCodeBits(7) == 0);

  const PolicyCodecConfig config{.total_units = 64, .max_actions = 6};
  const std::array<float, 6> uniform = {1, 1, 1, 1, 1, 1};
  const auto code = EncodeActionProbabilities(uniform, config);
  REQUIRE(code.ok());
  std::array<float, 6> decoded = {};
  REQUIRE(
      DecodeActionProbabilities(*code, absl::MakeSpan(decoded), config).ok());
  const std::array<uint16_t, 6> expected = {11, 11, 11, 11, 10, 10};
  for (size_t action = 0; action < decoded.size(); ++action) {
    CHECK(decoded[action] * 64 == expected[action]);
  }
  const std::array<float, 7> too_many = {1, 1, 1, 1, 1, 1, 1};
  CHECK_FALSE(EncodeActionProbabilities(too_many, config).ok());
}

TEST_CASE("action probability codes round trip every supported row size") {
  for (uint16_t total_units : {16, 64, 256}) {
    const PolicyCodecConfig config{
        .total_units = total_units,
        .max_actions = kMaxActionsPerNode};
    for (size_t action_count = 1; action_count <= 8; ++action_count) {
      for (size_t sample = 0; sample < 64; ++sample) {
        std::array<float, 8> input = {};
        size_t remaining = total_units;
        for (size_t action = 0; action + 1 < action_count; ++action) {
          const size_t units =
              (sample * 37 + action * 53) % (remaining + 1);
          input[action] = static_cast<float>(units) / total_units;
          remaining -= units;
        }
        input[action_count - 1] =
            static_cast<float>(remaining) / total_units;

        const auto code = EncodeActionProbabilities(
            absl::MakeConstSpan(input).subspan(0, action_count), config);
        REQUIRE(code.ok());
        std::array<float, 8> output = {};
        REQUIRE(DecodeActionProbabilities(
                    *code, absl::MakeSpan(output).subspan(0, action_count),
                    config)
                    .ok());
        CHECK(output == input);
      }
    }
  }

  std::array<float, 8> output = {};
  CHECK_FALSE(DecodeActionProbabilities(
                  std::numeric_limits<uint64_t>::max(),
                  absl::MakeSpan(output).subspan(0, 6))
                  .ok());
}

TEST_CASE("compact policies preserve quantized strategy behavior") {
  const InfoSetKey first = Key(7, 20, 3);
  const InfoSetKey uniform = Key(7, 21, 3);
  const InfoSetKey second = Key(11, 20, 5);
  const InfoSetKey reset = Key(1, 30, 1);
  const InfoSetKey largest =
      Key(std::numeric_limits<uint64_t>::max(),
          std::numeric_limits<uint32_t>::max(),
          std::numeric_limits<uint32_t>::max());
  Policy policy;
  policy.model = ModelFingerprint{42};
  AddRow(policy, first, {0.7f, 0.3f});
  AddRow(policy, uniform, {1.0f / 3, 1.0f / 3, 1.0f / 3});
  AddRow(policy, second, {0.0f, 0.25f, 0.25f, 0.5f});
  AddRow(policy, reset, {0.2f, 0.8f});
  AddRow(policy, largest, {0.5f, 0.2f, 0.1f, 0.1f,
                           0.05f, 0.025f, 0.025f, 0.0f});

  const PolicyCodecConfig config{.max_actions = kMaxActionsPerNode};
  const auto encoded = EncodePolicy(policy, config);
  REQUIRE(encoded.ok());
  const auto decoded = DecodePolicy(*encoded);
  REQUIRE(decoded.ok());
  CHECK(decoded->model == policy.model);
  CHECK(decoded->rows.size() == 4);

  std::vector<float> output(2);
  CHECK(decoded->strategy(first, absl::MakeSpan(output)));
  CHECK(output == Quantized({0.7f, 0.3f}));
  output.resize(4);
  CHECK(decoded->strategy(second, absl::MakeSpan(output)));
  CHECK(output == Quantized({0.0f, 0.25f, 0.25f, 0.5f}));
  output.resize(2);
  CHECK(decoded->strategy(reset, absl::MakeSpan(output)));
  CHECK(output == Quantized({0.2f, 0.8f}));
  output.resize(8);
  CHECK(decoded->strategy(largest, absl::MakeSpan(output)));
  CHECK(output == Quantized({0.5f, 0.2f, 0.1f, 0.1f,
                             0.05f, 0.025f, 0.025f, 0.0f},
                            config));
  output.resize(3);
  CHECK_FALSE(decoded->strategy(uniform, absl::MakeSpan(output)));
  CHECK(output == std::vector<float>(3, 1.0f / 3));

  Policy reordered = policy;
  reordered.rows.clear();
  reordered.rows.try_emplace(second, 5);
  reordered.rows.try_emplace(largest, 11);
  reordered.rows.try_emplace(uniform, 2);
  reordered.rows.try_emplace(reset, 9);
  reordered.rows.try_emplace(first, 0);
  const auto reencoded = EncodePolicy(reordered, config);
  REQUIRE(reencoded.ok());
  CHECK(*reencoded == *encoded);
}

TEST_CASE("compact policy encoding omits uniform rows and rejects damage") {
  Policy policy;
  for (uint32_t index = 0; index < 1000; ++index) {
    AddRow(policy, Key(index, index, index), {0.5f, 0.5f});
  }
  const auto encoded = EncodePolicy(policy);
  REQUIRE(encoded.ok());
  CHECK(encoded->size() < 64);
  const auto decoded = DecodePolicy(*encoded);
  REQUIRE(decoded.ok());
  CHECK(decoded->rows.empty());

  std::vector<uint8_t> damaged = *encoded;
  damaged.front() ^= 1;
  CHECK_FALSE(DecodePolicy(damaged).ok());
  damaged = *encoded;
  damaged.pop_back();
  CHECK_FALSE(DecodePolicy(damaged).ok());
  damaged = *encoded;
  damaged.push_back(0);
  CHECK_FALSE(DecodePolicy(damaged).ok());
}

}  // namespace
}  // namespace poker
