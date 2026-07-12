#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "src/poker.h"

namespace poker {

inline constexpr size_t kMaxActionsPerNode = 8;

struct BetAbstractionConfig {
  std::array<std::vector<double>, 4> pot_fractions = {{
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
  }};
};

using AbstractActions =
    absl::InlinedVector<GameAction, kMaxActionsPerNode>;

AbstractActions SelectAbstractActions(const BetAbstractionConfig& config,
                                      const DecisionState& state);

}  // namespace poker
