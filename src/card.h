#pragma once

#include "src/poker.pb.h"

namespace poker {

inline Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

}  // namespace poker
