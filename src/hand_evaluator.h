#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "src/poker.h"

namespace poker {

enum class HandRank {
  HighCard = 0,
  Pair = 1,
  TwoPair = 2,
  ThreeOfAKind = 3,
  Straight = 4,
  Flush = 5,
  FullHouse = 6,
  FourOfAKind = 7,
  StraightFlush = 8,
  RoyalFlush = 9
};

struct HandEvaluation {
  HandRank rank = HandRank::HighCard;
  std::array<int, 5> kickers = {};
  size_t kicker_count = 0;

  bool operator<(const HandEvaluation& other) const {
    if (rank != other.rank) {
      return static_cast<int>(rank) < static_cast<int>(other.rank);
    }

    for (size_t i = 0; i < std::min(kicker_count, other.kicker_count); ++i) {
      if (kickers[i] != other.kickers[i]) {
        return kickers[i] < other.kickers[i];
      }
    }

    return kicker_count < other.kicker_count;
  }

  bool operator>(const HandEvaluation& other) const {
    return other < *this;
  }

  bool operator==(const HandEvaluation& other) const {
    if (rank != other.rank || kicker_count != other.kicker_count) {
      return false;
    }
    for (size_t i = 0; i < kicker_count; ++i) {
      if (kickers[i] != other.kickers[i]) {
        return false;
      }
    }
    return true;
  }
};

HandEvaluation EvaluateFiveCards(const std::array<Card, 5>& cards);
uint16_t HandValue(ComboId hand, const RiverBoard& board) noexcept;
int CompareHands(ComboId first,
                 ComboId second,
                 const RiverBoard& board);

}  // namespace poker
