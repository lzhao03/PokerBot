#pragma once

#include <random>
#include <vector>

#include "src/card.h"
#include "src/poker.pb.h"

namespace poker {

bool SameCard(const Card& left, const Card& right);
std::vector<Card> BuildDeck();
Hand DealHand(std::vector<Card>& deck);
int CardsForNextStreet(Street street);
std::vector<Card> SampleStreetCards(const BoardState& state,
                                    const Hand& player_a_hand,
                                    const Hand& player_b_hand,
                                    std::mt19937& rng);

}  // namespace poker
