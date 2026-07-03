#pragma once

#include <cstdint>

#include "src/poker.pb.h"

namespace poker {

using CardMask = uint64_t;

inline Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

inline bool SameCard(const Card& left, const Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

inline CardMask CardBit(const Card& card) {
  const int rank_index = card.rank() - 2;
  const int suit_index = static_cast<int>(card.suit()) - 1;
  if (rank_index < 0 || rank_index >= 13 || suit_index < 0 ||
      suit_index >= 4) {
    return 0;
  }
  return CardMask{1} << (suit_index * 13 + rank_index);
}

inline CardMask HandMask(const Hand& hand) {
  CardMask mask = 0;
  for (const Card& card : hand.cards()) {
    mask |= CardBit(card);
  }
  return mask;
}

inline CardMask BoardMask(const BoardState& state) {
  CardMask mask = 0;
  for (const Card& card : state.cards()) {
    mask |= CardBit(card);
  }
  return mask;
}

}  // namespace poker
