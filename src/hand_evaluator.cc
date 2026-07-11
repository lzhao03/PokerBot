#include "src/hand_evaluator.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "absl/types/span.h"
#include "generated/hand_evaluator_tables.h"

namespace poker {
namespace {

HandEvaluation ToHandEvaluation(
    const hand_evaluator_tables::ScoreRecord& score) {
  HandEvaluation evaluation;
  evaluation.rank = score.rank;
  evaluation.kickers = score.kickers;
  evaluation.kicker_count = score.kicker_count;
  return evaluation;
}

class SevenCardHand {
 public:
  static SevenCardHand FromHoleAndBoard(ComboId hole_cards,
                                        absl::Span<const CardId> board_cards) {
    SevenCardHand hand(hole_cards);
    for (CardId card : board_cards) {
      hand.append(card);
    }
    return hand;
  }

  absl::Span<const CardId> cards() const {
    return {cards_.data(), count_};
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

int BuildCactusCard(CardId card) {
  static constexpr std::array<int, 13> kRankPrimes = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
  static constexpr std::array<int, 4> kSuitBits = {
      0x8000, 0x4000, 0x2000, 0x1000};
  const int rank_index = RankFromCardId(card) - 2;
  const int suit_index = SuitIndex(SuitFromCardId(card));
  return kRankPrimes[rank_index] | (rank_index << 8) |
         kSuitBits[suit_index] | (1 << (16 + rank_index));
}

const std::array<int, kDeckCardCount>& CactusCardTable() {
  static const std::array<int, kDeckCardCount> table = [] {
    std::array<int, kDeckCardCount> out = {};
    for (int id = 0; id < kDeckCardCount; ++id) {
      out[static_cast<size_t>(id)] = BuildCactusCard(static_cast<CardId>(id));
    }
    return out;
  }();
  return table;
}

int CactusCard(CardId card) {
  return CactusCardTable()[static_cast<size_t>(card)];
}

uint16_t ProductRank(int product) {
  const auto& products = hand_evaluator_tables::kCactusProducts;
  const auto it = std::lower_bound(
      products.begin(), products.end(), product,
      [](const auto& entry, int value) { return entry.first < value; });
  if (it == products.end() || it->first != product) {
    throw std::logic_error("Missing Cactus-Kev product lookup");
  }
  return it->second;
}

uint16_t EvalFiveCactus(const std::array<int, 5>& cards) {
  const int rank_mask =
      (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16;
  if ((cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000) != 0) {
    return hand_evaluator_tables::kCactusFlushes[rank_mask];
  }
  if (std::popcount(static_cast<unsigned int>(rank_mask)) == 5) {
    return hand_evaluator_tables::kCactusUnique5[rank_mask];
  }
  const int product = (cards[0] & 0xFF) * (cards[1] & 0xFF) *
                      (cards[2] & 0xFF) * (cards[3] & 0xFF) *
                      (cards[4] & 0xFF);
  return ProductRank(product);
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

int CompareCactusValues(uint16_t first, uint16_t second) {
  if (first < second) {
    return 1;
  }
  if (first > second) {
    return -1;
  }
  return 0;
}

uint16_t HandValue(ComboId hand, const BoardRunout& board) {
  const SevenCardHand cards =
      SevenCardHand::FromHoleAndBoard(hand, board.cards());
  return EvalBestCactus(cards.cards());
}

const hand_evaluator_tables::ScoreRecord& ScoreForCards(
    absl::Span<const CardId> cards) {
  return hand_evaluator_tables::kCactusScores[EvalBestCactus(cards)];
}

}  // namespace

HandEvaluation EvaluateFiveCards(const std::array<CardId, 5>& cards) {
  return ToHandEvaluation(ScoreForCards(absl::MakeConstSpan(cards)));
}

int CompareHands(ComboId first,
                 ComboId second,
                 const BoardRunout& board) {
  return CompareCactusValues(HandValue(first, board),
                             HandValue(second, board));
}

}  // namespace poker
