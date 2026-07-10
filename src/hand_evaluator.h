#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

enum class HandRank {
  HIGH_CARD = 0,
  PAIR = 1,
  TWO_PAIR = 2,
  THREE_OF_A_KIND = 3,
  STRAIGHT = 4,
  FLUSH = 5,
  FULL_HOUSE = 6,
  FOUR_OF_A_KIND = 7,
  STRAIGHT_FLUSH = 8,
  ROYAL_FLUSH = 9
};

struct HandEvaluation {
  HandRank rank = HandRank::HIGH_CARD;
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

class HandEvaluator {
 public:
  HandEvaluator() = default;

  HandEvaluation evaluate(const std::array<CardId, 5>& cards) const;
  HandEvaluation evaluate_hand(ComboId hole_cards,
                               const Board& board) const;

  uint16_t hand_value(ComboId hand, const Board& board) const;
  uint16_t hand_value(
      ComboId hand,
      const std::array<CardId, kMaxBoardCards>& board_cards,
      uint8_t board_count) const;

  int compare_hands(ComboId hand1, ComboId hand2,
                    const Board& board) const;
  int compare_hands(ComboId hand1, ComboId hand2,
                    const std::array<CardId, kMaxBoardCards>& board_cards,
                    uint8_t board_count) const;

 private:
  HandEvaluation find_best_hand(absl::Span<const CardId> cards) const;
};

}  // namespace poker
