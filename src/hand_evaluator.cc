#include "src/hand_evaluator.h"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace poker {
namespace {

struct EvaluationScore {
  HandRank rank = HandRank::HIGH_CARD;
  std::array<int, 5> kickers = {};
  size_t kicker_count = 0;

  bool operator<(const EvaluationScore& other) const {
    if (rank != other.rank) {
      return static_cast<int>(rank) < static_cast<int>(other.rank);
    }

    const size_t limit = std::min(kicker_count, other.kicker_count);
    for (size_t i = 0; i < limit; ++i) {
      if (kickers[i] != other.kickers[i]) {
        return kickers[i] < other.kickers[i];
      }
    }
    return kicker_count < other.kicker_count;
  }
};

EvaluationScore MakeScore(HandRank rank, std::initializer_list<int> kickers) {
  EvaluationScore score;
  score.rank = rank;
  score.kicker_count = kickers.size();
  size_t index = 0;
  for (int kicker : kickers) {
    score.kickers[index] = kicker;
    ++index;
  }
  return score;
}

HandEvaluation ToHandEvaluation(const EvaluationScore& score) {
  HandEvaluation evaluation;
  evaluation.rank = score.rank;
  evaluation.kickers = score.kickers;
  evaluation.kicker_count = score.kicker_count;
  return evaluation;
}

class SevenCardHand {
 public:
  static SevenCardHand FromHoleAndBoard(ComboId hole_cards,
                                        const GameState& board_state) {
    SevenCardHand hand(hole_cards);
    for (CardId card : board_state.board_cards) {
      hand.append(card);
    }
    return hand;
  }

  static SevenCardHand FromHoleAndBoard(
      ComboId hole_cards,
      const CompactPublicState& board_state) {
    SevenCardHand hand(hole_cards);
    for (uint8_t i = 0; i < board_state.board_count; ++i) {
      hand.append(board_state.board_cards[static_cast<size_t>(i)]);
    }
    return hand;
  }

  static SevenCardHand FromHoleAndBoard(
      ComboId hole_cards,
      const std::array<CardId, kMaxBoardCards>& board_cards,
      uint8_t board_count) {
    SevenCardHand hand(hole_cards);
    for (uint8_t i = 0; i < board_count; ++i) {
      hand.append(board_cards[static_cast<size_t>(i)]);
    }
    return hand;
  }

  absl::Span<const CardId> cards() const {
    return absl::Span<const CardId>(cards_.data(), count_);
  }

 private:
  explicit SevenCardHand(ComboId hole_cards) {
    const ComboInfo& combo = GetComboInfo(hole_cards);
    cards_[0] = combo.card0;
    cards_[1] = combo.card1;
    count_ = 2;
  }

  void append(CardId card) {
    cards_[count_] = card;
    ++count_;
  }

  std::array<CardId, 7> cards_ = {};
  size_t count_ = 0;
};

EvaluationScore EvaluateFiveCardScore(const std::array<CardId, 5>& cards) {
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
    return MakeScore(HandRank::ROYAL_FLUSH, {});
  }

  if (flush && straight) {
    return MakeScore(HandRank::STRAIGHT_FLUSH, {straight_high_rank});
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
      return MakeScore(HandRank::FOUR_OF_A_KIND,
                       {static_cast<int>(i), kicker});
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
    return MakeScore(HandRank::FULL_HOUSE,
                     {three_of_a_kind_rank, pair_rank});
  }

  if (flush) {
    return MakeScore(HandRank::FLUSH,
                     {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
  }

  if (straight) {
    return MakeScore(HandRank::STRAIGHT, {straight_high_rank});
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
    return MakeScore(HandRank::THREE_OF_A_KIND,
                     {kickers[0], kickers[1], kickers[2]});
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
    return MakeScore(HandRank::TWO_PAIR, {high_pair, low_pair, kicker});
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
    return MakeScore(HandRank::PAIR,
                     {kickers[0], kickers[1], kickers[2], kickers[3]});
  }

  return MakeScore(HandRank::HIGH_CARD,
                   {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
}

int CompareScores(const EvaluationScore& first, const EvaluationScore& second) {
  if (second < first) {
    return 1;
  }
  if (first < second) {
    return -1;
  }
  return 0;
}

int CountBits(int value) {
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

int CactusCard(CardId card) {
  static constexpr std::array<int, 13> kRankPrimes = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
  static constexpr std::array<int, 4> kSuitBits = {
      0x8000, 0x4000, 0x2000, 0x1000};
  const int rank_index = RankFromCardId(card) - 2;
  const int suit_index = SuitIndex(SuitFromCardId(card));
  return kRankPrimes[rank_index] | (rank_index << 8) |
         kSuitBits[suit_index] | (1 << (16 + rank_index));
}

EvaluationScore ScoreForDistinctRanks(int rank_mask, bool flush) {
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

EvaluationScore ScoreForRankMultiset(
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

int PrimeProductForRanks(const std::array<int, 5>& rank_indices) {
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

// Tables are generated once from the slow five-card scorer. Lower lookup
// value means stronger hand, matching Cactus Kev convention.
struct CactusTables {
  CactusTables() {
    std::vector<CactusLookupRecord> records;
    records.reserve(7462);

    for (int rank_mask = 0; rank_mask < 8192; ++rank_mask) {
      if (CountBits(rank_mask) != 5) {
        continue;
      }
      records.push_back(
          {ScoreForDistinctRanks(rank_mask, true),
           CactusLookupRecord::Kind::kFlush, rank_mask});
      records.push_back(
          {ScoreForDistinctRanks(rank_mask, false),
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
              records.push_back(
                  {ScoreForRankMultiset(ranks),
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

    products.reserve(records.size());
    for (size_t i = 0; i < records.size(); ++i) {
      const uint16_t value = static_cast<uint16_t>(i + 1);
      const CactusLookupRecord& record = records[i];
      switch (record.kind) {
        case CactusLookupRecord::Kind::kFlush:
          flushes[record.key] = value;
          break;
        case CactusLookupRecord::Kind::kUnique:
          unique5[record.key] = value;
          break;
        case CactusLookupRecord::Kind::kProduct:
          products.push_back({record.key, value});
          break;
      }
    }
    std::sort(products.begin(), products.end());
  }

  std::array<uint16_t, 8192> flushes = {};
  std::array<uint16_t, 8192> unique5 = {};
  std::vector<std::pair<int, uint16_t>> products;
};

const CactusTables& GetCactusTables() {
  static const CactusTables tables;
  return tables;
}

uint16_t ProductRank(const CactusTables& tables, int product) {
  const auto it = std::lower_bound(
      tables.products.begin(), tables.products.end(), product,
      [](const std::pair<int, uint16_t>& entry, int value) {
        return entry.first < value;
      });
  if (it == tables.products.end() || it->first != product) {
    throw std::logic_error("Missing Cactus-Kev product lookup");
  }
  return it->second;
}

uint16_t EvalFiveCactus(const std::array<int, 5>& cards) {
  const CactusTables& tables = GetCactusTables();
  const int rank_mask =
      (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16;
  if ((cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000) != 0) {
    return tables.flushes[rank_mask];
  }
  if (CountBits(rank_mask) == 5) {
    return tables.unique5[rank_mask];
  }
  const int product = (cards[0] & 0xFF) * (cards[1] & 0xFF) *
                      (cards[2] & 0xFF) * (cards[3] & 0xFF) *
                      (cards[4] & 0xFF);
  return ProductRank(tables, product);
}

uint16_t EvalBestCactus(absl::Span<const CardId> cards) {
  if (cards.size() < 5) {
    throw std::invalid_argument("Need at least 5 cards to find best hand");
  }
  if (cards.size() > 7) {
    throw std::invalid_argument("Cannot evaluate more than seven cards");
  }

  std::array<int, 7> cactus_cards = {};
  for (size_t i = 0; i < cards.size(); ++i) {
    cactus_cards[i] = CactusCard(cards[i]);
  }

  uint16_t best = std::numeric_limits<uint16_t>::max();
  std::array<int, 5> combo;
  for (size_t a = 0; a + 4 < cards.size(); ++a) {
    combo[0] = cactus_cards[a];
    for (size_t b = a + 1; b + 3 < cards.size(); ++b) {
      combo[1] = cactus_cards[b];
      for (size_t c = b + 1; c + 2 < cards.size(); ++c) {
        combo[2] = cactus_cards[c];
        for (size_t d = c + 1; d + 1 < cards.size(); ++d) {
          combo[3] = cactus_cards[d];
          for (size_t e = d + 1; e < cards.size(); ++e) {
            combo[4] = cactus_cards[e];
            best = std::min(best, EvalFiveCactus(combo));
          }
        }
      }
    }
  }
  return best;
}

int CompareCactusHands(absl::Span<const CardId> first_cards,
                       absl::Span<const CardId> second_cards) {
  const uint16_t first = EvalBestCactus(first_cards);
  const uint16_t second = EvalBestCactus(second_cards);
  if (first < second) {
    return 1;
  }
  if (first > second) {
    return -1;
  }
  return 0;
}

EvaluationScore FindBestHandScore(absl::Span<const CardId> cards) {
  if (cards.size() < 5) {
    throw std::invalid_argument("Need at least 5 cards to find best hand");
  }

  EvaluationScore bestEval;
  std::array<CardId, 5> combo;
  for (size_t a = 0; a + 4 < cards.size(); ++a) {
    combo[0] = cards[a];
    for (size_t b = a + 1; b + 3 < cards.size(); ++b) {
      combo[1] = cards[b];
      for (size_t c = b + 1; c + 2 < cards.size(); ++c) {
        combo[2] = cards[c];
        for (size_t d = c + 1; d + 1 < cards.size(); ++d) {
          combo[3] = cards[d];
          for (size_t e = d + 1; e < cards.size(); ++e) {
            combo[4] = cards[e];
            EvaluationScore currentEval = EvaluateFiveCardScore(combo);
            if (bestEval < currentEval) {
              bestEval = currentEval;
            }
          }
        }
      }
    }
  }

  return bestEval;
}

}  // namespace

HandEvaluation HandEvaluator::evaluate(
    const std::array<CardId, 5>& cards) const {
  return ToHandEvaluation(EvaluateFiveCardScore(cards));
}

HandEvaluation HandEvaluator::evaluate_hand(
    ComboId hole_cards,
    const GameState& board_state) const {
  const SevenCardHand cards =
      SevenCardHand::FromHoleAndBoard(hole_cards, board_state);
  return find_best_hand(cards.cards());
}

HandEvaluation HandEvaluator::evaluate_hand(
    ComboId hole_cards,
    const CompactPublicState& board_state) const {
  const SevenCardHand cards =
      SevenCardHand::FromHoleAndBoard(hole_cards, board_state);
  return find_best_hand(cards.cards());
}

int HandEvaluator::compare_hands(
    ComboId hand1,
    ComboId hand2,
    const GameState& board_state) const {
  const SevenCardHand first =
      SevenCardHand::FromHoleAndBoard(hand1, board_state);
  const SevenCardHand second =
      SevenCardHand::FromHoleAndBoard(hand2, board_state);
  return CompareCactusHands(first.cards(), second.cards());
}

int HandEvaluator::compare_hands(
    ComboId hand1,
    ComboId hand2,
    const CompactPublicState& board_state) const {
  const SevenCardHand first =
      SevenCardHand::FromHoleAndBoard(hand1, board_state);
  const SevenCardHand second =
      SevenCardHand::FromHoleAndBoard(hand2, board_state);
  return CompareCactusHands(first.cards(), second.cards());
}

int HandEvaluator::compare_hands(
    ComboId hand1,
    ComboId hand2,
    const std::array<CardId, kMaxBoardCards>& board_cards,
    uint8_t board_count) const {
  const SevenCardHand first =
      SevenCardHand::FromHoleAndBoard(hand1, board_cards, board_count);
  const SevenCardHand second =
      SevenCardHand::FromHoleAndBoard(hand2, board_cards, board_count);
  return CompareCactusHands(first.cards(), second.cards());
}

HandEvaluation HandEvaluator::find_best_hand(
    absl::Span<const CardId> cards) const {
  return ToHandEvaluation(FindBestHandScore(cards));
}

} // namespace poker
