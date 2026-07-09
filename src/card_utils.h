#pragma once

#include <random>

#include "absl/container/inlined_vector.h"
#include "src/poker_types.h"

namespace poker {

int CardsForNextStreet(StreetKind street);
StreetKind StreetAfterChance(StreetKind street);
int BoardCardsForStreet(StreetKind street);
absl::InlinedVector<CardId, 5> SampleStreetCards(
    StreetKind street,
    int board_count,
    CardMask board_mask,
    CardMask known_private_cards,
    std::mt19937& rng);
absl::InlinedVector<CardId, 5> SampleStreetCards(
    const CompactPublicState& state,
    CardMask known_private_cards,
    std::mt19937& rng);

}  // namespace poker
