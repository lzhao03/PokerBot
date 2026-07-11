#pragma once

#include <array>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "src/game_state.h"

namespace poker {

using SolverActions = absl::InlinedVector<GameAction, 8>;

ExactPublicState MakeInitialState(
    const BettingRules& rules,
    std::array<Chips, kPlayerCount> stacks,
    std::array<Chips, kPlayerCount> blinds);
SolverActions GetSolverActions(const SolverConfig& config,
                               const BettingState& state);
BettingState ApplyAction(const BettingState& state,
                         const GameAction& action);
BettingState AdvanceBettingStreet(const BettingState& state,
                                  const BettingRules& rules);
ExactPublicState ApplyChance(const ExactPublicState& state,
                             absl::Span<const CardId> cards,
                             const BettingRules& rules);
double TerminalUtility(const ExactPublicState& state,
                       ComboId player0_hand,
                       ComboId player1_hand);
bool IsTerminal(const ExactPublicState& state);
bool IsBettingRoundOver(const BettingState& state) noexcept;

}  // namespace poker
