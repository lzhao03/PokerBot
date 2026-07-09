#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "src/hand_evaluator.h"

namespace poker::hand_evaluator_generation {

constexpr size_t kCactusScoreCount = 7463;

struct EvaluationScore {
  HandRank rank = HandRank::HIGH_CARD;
  std::array<int, 5> kickers = {};
  size_t kicker_count = 0;

  bool operator<(const EvaluationScore& other) const;
};

struct TableData {
  std::array<uint16_t, 8192> flushes = {};
  std::array<uint16_t, 8192> unique5 = {};
  std::array<EvaluationScore, kCactusScoreCount> scores = {};
  std::vector<std::pair<int, uint16_t>> products;
};

EvaluationScore EvaluateFiveCardScore(const std::array<CardId, 5>& cards);
int CompareScores(const EvaluationScore& first, const EvaluationScore& second);
TableData BuildCactusTables();

}  // namespace poker::hand_evaluator_generation
