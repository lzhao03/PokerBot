#pragma once

#include "absl/types/span.h"
#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

BettingState ApplyAction(const BettingState& state,
                         const GameAction& action);
CompactPublicState ApplyAction(const CompactPublicState& state,
                               const GameAction& action);
ExactGameState ApplyChance(const ExactGameState& state,
                           absl::Span<const CardId> cards);
CompactPublicState ApplyChance(const CompactPublicState& state,
                               absl::Span<const CardId> cards);
double GetUtility(const ExactGameState& state,
                  ComboId player_a_hand,
                  ComboId player_b_hand);
bool IsTerminal(const BettingState& state, const Board& board);
int GetPlayerToAct(const BettingState& state, const Board& board);
bool IsBettingRoundOver(const BettingState& state);

}  // namespace poker
