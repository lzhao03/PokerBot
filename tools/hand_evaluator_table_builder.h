#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "src/poker.h"

namespace poker::hand_evaluator_generation {

enum class HandRank {
  HighCard,
  Pair,
  TwoPair,
  ThreeOfAKind,
  Straight,
  Flush,
  FullHouse,
  FourOfAKind,
  StraightFlush,
  RoyalFlush,
};

struct EvaluationScore {
  HandRank rank = HandRank::HighCard;
  std::array<int, 5> kickers = {};

  friend auto operator<=>(const EvaluationScore&,
                          const EvaluationScore&) = default;
};

struct TableData {
  std::array<uint16_t, 8192> flushes = {};
  std::array<uint16_t, 8192> unique5 = {};
  std::vector<std::pair<int, uint16_t>> products;
};

EvaluationScore EvaluateFiveCardScore(const std::array<Card, 5>& cards);
TableData BuildCactusTables();

}  // namespace poker::hand_evaluator_generation
