#pragma once

#include <random>
#include <vector>

#include "src/poker_types.h"

namespace poker {

std::vector<CardId> BuildDeck();
int CardsForNextStreet(StreetKind street);
std::vector<CardId> SampleStreetCards(const GameState& state,
                                      CardMask known_private_cards,
                                      std::mt19937& rng);

}  // namespace poker
