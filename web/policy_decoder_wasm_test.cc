#include "web/policy_decoder_wasm.h"

#include <array>
#include <cstdint>

#include "doctest/doctest.h"
#include "src/bet_abstraction.h"
#include "src/policy_codec.h"

namespace poker {
namespace {

TEST_CASE("standalone decoder matches the policy codec") {
  constexpr uint64_t kPublic = (uint64_t{5} << 32) | 7;
  const InfoSetKey key{PublicObservationId(kPublic), HistoryId(20),
                       PrivateObservationId(3)};
  Policy policy;
  policy.model = ModelFingerprint{0x123456789abcdef0};
  policy.rows.try_emplace(key, 0);
  policy.probabilities = {0.7f, 0.3f};
  const auto encoded = EncodePolicy(
      policy, {.max_actions = kMaxActionsPerNode});
  REQUIRE(encoded.ok());
  REQUIRE(poker_load_policy(encoded->data(), encoded->size()) == 1);

  std::array<float, 2> output;
  CHECK(poker_strategy(7, 5, 20, 3, output.size(), output.data()) == 1);
  const std::array<float, 2> expected = {11.0f / 16, 5.0f / 16};
  CHECK(output == expected);
  CHECK(poker_strategy(0, 0, 0, 0, output.size(), output.data()) == 0);
  const std::array<float, 2> uniform = {0.5f, 0.5f};
  CHECK(output == uniform);
  CHECK(poker_model_low() == 0x9abcdef0);
  CHECK(poker_model_high() == 0x12345678);

  poker_unload_policy();
  CHECK(poker_strategy(0, 0, 0, 0, output.size(), output.data()) == -1);
}

}  // namespace
}  // namespace poker
