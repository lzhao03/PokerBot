#pragma once

#include "absl/types/span.h"
#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

CompactPublicState ApplyAction(const CompactPublicState& state,
                               const GameAction& action);
CompactPublicState ApplyChance(const CompactPublicState& state,
                               absl::Span<const CardId> cards);
double GetUtility(const CompactPublicState& state,
                  ComboId player_a_hand,
                  ComboId player_b_hand);
bool IsTerminal(const CompactPublicState& state);
int GetPlayerToAct(const CompactPublicState& state);
bool IsBettingRoundOver(const CompactPublicState& state);

}  // namespace poker
