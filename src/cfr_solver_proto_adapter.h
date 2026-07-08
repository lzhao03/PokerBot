#pragma once

#include "src/combo.h"
#include "src/poker.pb.h"
#include "src/poker_types.h"

namespace poker {

SolverConfig SolverConfigFromProto(const PokerConfig& config);
PokerConfig SolverConfigToProto(const SolverConfig& config);
GameState GameStateFromProto(const BoardState& state);
CompactPublicState CompactPublicStateFromProto(const BoardState& state);
GameAction GameActionFromProto(const Action& action);
ComboId ComboIdFromProtoHand(const Hand& hand);
CardId CardIdFromProto(const Card& card);

}  // namespace poker
