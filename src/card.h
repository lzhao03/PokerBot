#pragma once

#include "src/poker.pb.h"

namespace poker {

inline Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

inline bool SameCard(const Card& left, const Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

}  // namespace poker
