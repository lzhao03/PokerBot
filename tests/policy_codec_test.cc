#include "src/policy_codec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/types/span.h"
#include "doctest/doctest.h"
#include "src/bet_abstraction.h"

namespace poker {
namespace {

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

}  // namespace
}  // namespace poker
