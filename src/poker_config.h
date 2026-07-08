#pragma once

#include <string>

#include "src/poker.pb.h"
#include "src/poker_types.h"

namespace poker {

PokerConfig DefaultPokerConfig();
SolverConfig SolverConfigFromProto(const PokerConfig& config);
void LoadPokerConfig(const std::string& path, PokerConfig* config);
void AddBetSize(PokerConfig* config, Street street, double size);

}  // namespace poker
