#include "src/card_utils.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace poker {

int CardsForNextStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 3;
    case StreetKind::kFlop:
    case StreetKind::kTurn:
      return 1;
    case StreetKind::kRiver:
      return 0;
  }
}

StreetKind StreetAfterChance(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return StreetKind::kFlop;
    case StreetKind::kFlop:
      return StreetKind::kTurn;
    case StreetKind::kTurn:
    case StreetKind::kRiver:
      return StreetKind::kRiver;
  }
}

int BoardCardsForStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 0;
    case StreetKind::kFlop:
      return 3;
    case StreetKind::kTurn:
      return 4;
    case StreetKind::kRiver:
      return 5;
  }
}

absl::InlinedVector<CardId, 5> SampleStreetCards(
    StreetKind street,
    int board_count,
    CardMask board_mask,
    CardMask known_private_cards,
    std::mt19937& rng) {
  const int remaining_board_slots =
      std::max(0, kMaxBoardCards - board_count);
  const int count = std::min(CardsForNextStreet(street),
                             remaining_board_slots);
  if (count <= 0) {
    return {};
  }

  const CardMask blocked_mask = known_private_cards | board_mask;
  if (count == 1) {
    std::uniform_int_distribution<int> card_distribution(
        0, kDeckCardCount - 1);
    for (int attempt = 0; attempt < kDeckCardCount; ++attempt) {
      const CardId candidate =
          static_cast<CardId>(card_distribution(rng));
      if ((blocked_mask & CardBit(candidate)) == 0) {
        return {candidate};
      }
    }
  }

  std::array<CardId, kDeckCardCount> candidates = {};
  int candidate_count = 0;
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    const CardId candidate = static_cast<CardId>(card_id);
    if ((blocked_mask & CardBit(candidate)) == 0) {
      candidates[candidate_count] = candidate;
      ++candidate_count;
    }
  }

  if (candidate_count < count) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  absl::InlinedVector<CardId, 5> sampled;
  sampled.reserve(count);
  for (int i = 0; i < count; ++i) {
    std::uniform_int_distribution<int> card_distribution(
        i, candidate_count - 1);
    const int chosen = card_distribution(rng);
    std::swap(candidates[i], candidates[chosen]);
    sampled.push_back(candidates[i]);
  }
  return sampled;
}

absl::InlinedVector<CardId, 5> SampleStreetCards(
    const CompactPublicState& state,
    CardMask known_private_cards,
    std::mt19937& rng) {
  return SampleStreetCards(
      state.street, state.board_count, state.board_mask,
      known_private_cards, rng);
}

}  // namespace poker
