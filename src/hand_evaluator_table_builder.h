#pragma once

#include <algorithm>
#include <array>
#include <bit>
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

  bool operator<(const EvaluationScore& other) const {
    if (rank != other.rank) {
      return static_cast<int>(rank) < static_cast<int>(other.rank);
    }
    return kickers < other.kickers;
  }
};

template <typename... Kickers>
EvaluationScore MakeScore(HandRank rank, Kickers... kickers) {
  static_assert(sizeof...(Kickers) <= 5);

  EvaluationScore score;
  score.rank = rank;
  score.kickers = {static_cast<int>(kickers)...};
  score.kicker_count = sizeof...(Kickers);
  return score;
}

inline EvaluationScore EvaluateFiveCardScore(
    const std::array<CardId, 5>& cards) {
  std::array<int, 5> ranks;
  size_t rank_count = 0;
  for (CardId card : cards) {
    ranks[rank_count] = RankFromCardId(card);
    ++rank_count;
  }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  bool flush = true;
  for (CardId card : cards) {
    if (SuitFromCardId(card) != SuitFromCardId(cards[0])) {
      flush = false;
      break;
    }
  }

  const bool wheel_straight = ranks[0] == 14 && ranks[1] == 5 &&
                              ranks[2] == 4 && ranks[3] == 3 &&
                              ranks[4] == 2;
  bool straight = wheel_straight;
  if (!straight) {
    straight = true;
    for (size_t i = 1; i < ranks.size(); ++i) {
      if (ranks[i - 1] != ranks[i] + 1) {
        straight = false;
        break;
      }
    }
  }
  const int straight_high_rank = wheel_straight ? 5 : ranks[0];

  if (flush && ranks[0] == 14 && ranks[1] == 13 && ranks[2] == 12 &&
      ranks[3] == 11 && ranks[4] == 10) {
    return MakeScore(HandRank::ROYAL_FLUSH);
  }

  if (flush && straight) {
    return MakeScore(HandRank::STRAIGHT_FLUSH, straight_high_rank);
  }

  std::array<int, 15> rank_counts = {};
  for (int rank : ranks) {
    ++rank_counts[rank];
  }

  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 4) {
      int kicker = -1;
      for (int rank : ranks) {
        if (rank != static_cast<int>(i)) {
          kicker = rank;
          break;
        }
      }
      return MakeScore(HandRank::FOUR_OF_A_KIND, static_cast<int>(i),
                       kicker);
    }
  }

  int three_of_a_kind_rank = -1;
  int pair_rank = -1;
  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 3) {
      three_of_a_kind_rank = static_cast<int>(i);
    } else if (rank_counts[i] == 2) {
      pair_rank = static_cast<int>(i);
    }
  }

  if (three_of_a_kind_rank != -1 && pair_rank != -1) {
    return MakeScore(HandRank::FULL_HOUSE, three_of_a_kind_rank, pair_rank);
  }

  if (flush) {
    return MakeScore(HandRank::FLUSH, ranks[0], ranks[1], ranks[2], ranks[3],
                     ranks[4]);
  }

  if (straight) {
    return MakeScore(HandRank::STRAIGHT, straight_high_rank);
  }

  if (three_of_a_kind_rank != -1) {
    std::array<int, 3> kickers = {three_of_a_kind_rank, 0, 0};
    size_t kicker_count = 1;
    for (int rank : ranks) {
      if (rank != three_of_a_kind_rank) {
        kickers[kicker_count] = rank;
        ++kicker_count;
      }
    }
    return MakeScore(HandRank::THREE_OF_A_KIND, kickers[0], kickers[1],
                     kickers[2]);
  }

  std::array<int, 2> pairs = {};
  size_t pair_count = 0;
  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 2) {
      pairs[pair_count] = static_cast<int>(i);
      ++pair_count;
    }
  }

  if (pair_count >= 2) {
    const int high_pair = pairs[1];
    const int low_pair = pairs[0];
    int kicker = -1;
    for (int rank : ranks) {
      if (rank != high_pair && rank != low_pair) {
        kicker = rank;
        break;
      }
    }
    return MakeScore(HandRank::TWO_PAIR, high_pair, low_pair, kicker);
  }

  if (pair_count > 0) {
    std::array<int, 4> kickers = {pairs[0], 0, 0, 0};
    size_t kicker_count = 1;
    for (int rank : ranks) {
      if (rank != pairs[0]) {
        kickers[kicker_count] = rank;
        ++kicker_count;
      }
    }
    return MakeScore(HandRank::PAIR, kickers[0], kickers[1], kickers[2],
                     kickers[3]);
  }

  return MakeScore(HandRank::HIGH_CARD, ranks[0], ranks[1], ranks[2],
                   ranks[3], ranks[4]);
}

inline int CompareScores(const EvaluationScore& first,
                         const EvaluationScore& second) {
  if (second < first) {
    return 1;
  }
  if (first < second) {
    return -1;
  }
  return 0;
}

inline EvaluationScore ScoreForDistinctRanks(int rank_mask, bool flush) {
  static constexpr std::array<SuitKind, 5> kNonFlushSuits = {
      SuitKind::kHearts, SuitKind::kDiamonds, SuitKind::kClubs,
      SuitKind::kSpades, SuitKind::kHearts};
  std::array<CardId, 5> cards = {};
  size_t index = 0;
  for (int rank_index = 0; rank_index < 13; ++rank_index) {
    if ((rank_mask & (1 << rank_index)) == 0) {
      continue;
    }
    const SuitKind suit = flush ? SuitKind::kHearts : kNonFlushSuits[index];
    cards[index] = MakeCardId(rank_index + 2, suit);
    ++index;
  }
  return EvaluateFiveCardScore(cards);
}

inline EvaluationScore ScoreForRankMultiset(
    const std::array<int, 5>& rank_indices) {
  static constexpr std::array<SuitKind, 4> kSuits = {
      SuitKind::kHearts, SuitKind::kDiamonds,
      SuitKind::kClubs, SuitKind::kSpades};
  std::array<CardId, 5> cards = {};
  std::array<int, 13> rank_counts = {};
  for (size_t i = 0; i < rank_indices.size(); ++i) {
    const int rank_index = rank_indices[i];
    const int suit_index = rank_counts[rank_index];
    cards[i] = MakeCardId(rank_index + 2, kSuits[suit_index]);
    ++rank_counts[rank_index];
  }
  return EvaluateFiveCardScore(cards);
}

inline int PrimeProductForRanks(const std::array<int, 5>& rank_indices) {
  static constexpr std::array<int, 13> kRankPrimes = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
  int product = 1;
  for (int rank_index : rank_indices) {
    product *= kRankPrimes[rank_index];
  }
  return product;
}

struct CactusLookupRecord {
  enum class Kind { kFlush, kUnique, kProduct };

  EvaluationScore score;
  Kind kind = Kind::kUnique;
  int key = 0;
};

struct TableData {
  std::array<uint16_t, 8192> flushes = {};
  std::array<uint16_t, 8192> unique5 = {};
  std::array<EvaluationScore, kCactusScoreCount> scores = {};
  std::vector<std::pair<int, uint16_t>> products;
};

inline TableData BuildCactusTables() {
  std::vector<CactusLookupRecord> records;
  records.reserve(7462);

  for (int rank_mask = 0; rank_mask < 8192; ++rank_mask) {
    if (std::popcount(static_cast<unsigned int>(rank_mask)) != 5) {
      continue;
    }
    records.push_back({ScoreForDistinctRanks(rank_mask, true),
                       CactusLookupRecord::Kind::kFlush, rank_mask});
    records.push_back({ScoreForDistinctRanks(rank_mask, false),
                       CactusLookupRecord::Kind::kUnique, rank_mask});
  }

  for (int a = 0; a < 13; ++a) {
    for (int b = a; b < 13; ++b) {
      for (int c = b; c < 13; ++c) {
        for (int d = c; d < 13; ++d) {
          for (int e = d; e < 13; ++e) {
            const std::array<int, 5> ranks = {a, b, c, d, e};
            std::array<int, 13> counts = {};
            int distinct_count = 0;
            bool too_many = false;
            for (int rank_index : ranks) {
              if (counts[rank_index] == 0) {
                ++distinct_count;
              }
              ++counts[rank_index];
              if (counts[rank_index] > 4) {
                too_many = true;
                break;
              }
            }
            if (too_many || distinct_count == 5) {
              continue;
            }
            records.push_back({ScoreForRankMultiset(ranks),
                               CactusLookupRecord::Kind::kProduct,
                               PrimeProductForRanks(ranks)});
          }
        }
      }
    }
  }

  std::sort(records.begin(), records.end(),
            [](const CactusLookupRecord& first,
               const CactusLookupRecord& second) {
              return CompareScores(first.score, second.score) > 0;
            });

  TableData tables;
  tables.products.reserve(records.size());
  for (size_t i = 0; i < records.size(); ++i) {
    const uint16_t value = static_cast<uint16_t>(i + 1);
    const CactusLookupRecord& record = records[i];
    tables.scores[value] = record.score;
    switch (record.kind) {
      case CactusLookupRecord::Kind::kFlush:
        tables.flushes[record.key] = value;
        break;
      case CactusLookupRecord::Kind::kUnique:
        tables.unique5[record.key] = value;
        break;
      case CactusLookupRecord::Kind::kProduct:
        tables.products.push_back({record.key, value});
        break;
    }
  }
  std::sort(tables.products.begin(), tables.products.end());
  return tables;
}

}  // namespace poker::hand_evaluator_generation
