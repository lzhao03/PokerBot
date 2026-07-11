#pragma once

#include <array>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "src/poker.h"

namespace poker {

struct BetAbstractionConfig {
  std::array<std::vector<double>, 4> bet_sizes = {{
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
  }};
};

using AbstractActions = absl::InlinedVector<GameAction, 8>;

AbstractActions SelectAbstractActions(const BetAbstractionConfig& config,
                                      const DecisionState& state,
                                      const LegalActionSpace& legal);

}  // namespace poker
