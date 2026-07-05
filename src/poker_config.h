#pragma once

#include <string>

#include "src/poker.pb.h"

namespace poker {

PokerConfig DefaultPokerConfig();
void LoadPokerConfig(const std::string& path, PokerConfig* config);
void AddBetSize(PokerConfig* config, Street street, double size);

}  // namespace poker
