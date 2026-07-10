#pragma once

#include <array>

#include "absl/types/span.h"
#include "src/combo.h"
#include "src/game_state.h"

namespace poker {

struct ActionMenu {
  uint8_t count = 0;
  std::array<GameAction, kMaxActionsPerNode> actions = {};
};

ActionMenu LegalActions(const BettingState& state,
                        absl::Span<const double> bet_sizes);
BettingState ApplyAction(const BettingState& state,
                         const GameAction& action);
ExactPublicState ApplyChance(const ExactPublicState& state,
                             absl::Span<const CardId> cards,
                             const BettingRules& rules);
double TerminalUtility(const ExactPublicState& state,
                       ComboId player0_hand,
                       ComboId player1_hand);
bool IsTerminal(const ExactPublicState& state);
bool IsBettingRoundOver(const BettingState& state) noexcept;

}  // namespace poker
