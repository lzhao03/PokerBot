#include "src/experimental_policy_encoder.h"

#include <cstddef>
#include <cstdint>

#include "doctest/doctest.h"

namespace poker {
namespace {

HistoryTree ThreeActionHistory() {
  HistoryTree history;
  history.nodes.push_back(DecisionNode{DecisionState{}, EdgeRange{0, 3}});
  history.edges.resize(3);
  return history;
}

Policy TestPolicy(bool reverse) {
  Policy policy;
  for (size_t index = 0; index < policy.model.bytes.size(); ++index) {
    policy.model.bytes[index] = static_cast<std::byte>(index);
  }
  policy.probabilities = {
      0.1f, 0.2f, 0.7f, 0.0f, 0.0f, 0.0f, 0.4f, 0.4f, 0.2f,
  };
  const std::array<std::pair<InfoSetKey, PolicyRow>, 3> rows = {{
      {{HistoryId(0), PublicObservationId(4), PrivateObservationId(7)}, {0, 3}},
      {{HistoryId(0), PublicObservationId(4), PrivateObservationId(9)}, {3, 3}},
      {{HistoryId(0), PublicObservationId(8), PrivateObservationId(2)}, {6, 3}},
  }};
  if (reverse) {
    for (auto row = rows.rbegin(); row != rows.rend(); ++row) {
      policy.rows.emplace(row->first, row->second);
    }
  } else {
    for (const auto &[key, row] : rows) policy.rows.emplace(key, row);
  }
  return policy;
}

TEST_CASE("experimental policy encoding is compact and deterministic") {
  const HistoryTree history = ThreeActionHistory();
  const auto forward = EncodeExperimentalPolicy(TestPolicy(false), history);
  const auto reverse = EncodeExperimentalPolicy(TestPolicy(true), history);
  REQUIRE(forward.ok());
  REQUIRE(reverse.ok());
  CHECK(*forward == *reverse);

  constexpr size_t kCurrentFormatBytes = 60 + 3 * (29 + 3 * sizeof(float));
  CHECK(forward->size() < kCurrentFormatBytes);
}

TEST_CASE("experimental policy encoding rejects invalid rows") {
  const HistoryTree history = ThreeActionHistory();
  Policy policy = TestPolicy(false);
  policy.rows.begin()->second.action_offset = policy.probabilities.size();
  CHECK_FALSE(EncodeExperimentalPolicy(policy, history).ok());
}

}  // namespace
}  // namespace poker
