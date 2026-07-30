#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "src/bet_abstraction.h"

namespace poker {

// Bump when history IDs change for an otherwise identical game.
inline constexpr uint8_t kHistorySchemaVersion = 0;

enum class HistoryId : uint32_t {};

constexpr size_t Index(HistoryId history) noexcept {
  return std::to_underlying(history);
}

struct HistoryNode {
  BettingState state;
  uint32_t children_begin = 0;
  uint8_t child_count = 0;
};

struct HistoryTree {
  std::vector<HistoryNode> nodes;
  std::vector<HistoryId> children;
};

HistoryTree BuildHistoryTree(const BettingState& root,
                             const BettingRules& rules,
                             const BetAbstractionConfig& abstraction);

}  // namespace poker
