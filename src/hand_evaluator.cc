#include "src/hand_evaluator.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include "generated/hand_evaluator_tables.h"

namespace poker {
namespace {

static constexpr std::array<int, 13> kRankPrimes = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

HandEvaluation ToHandEvaluation(
    const hand_evaluator_tables::ScoreRecord& score) {
  HandEvaluation evaluation;
  evaluation.rank = score.rank;
  evaluation.kickers = score.kickers;
  evaluation.kicker_count = score.kicker_count;
  return evaluation;
}

int BuildCactusCard(Card card) {
  static constexpr std::array<int, 4> kSuitBits = {
      0x8000, 0x4000, 0x2000, 0x1000};
  const int rank_index = PokerRank(card) - 2;
  const int suit_index = SuitIndex(CardSuit(card));
  return kRankPrimes[static_cast<size_t>(rank_index)] | (rank_index << 8) |
         kSuitBits[static_cast<size_t>(suit_index)] |
         (1 << (16 + rank_index));
}

uint16_t HighestStraightMask(uint16_t rank_mask) noexcept {
  for (int start = 8; start >= 0; --start) {
    const uint16_t straight = static_cast<uint16_t>(0x1F << start);
    if ((rank_mask & straight) == straight) return straight;
  }
  constexpr uint16_t kWheel = (1U << 12) | 0x0F;
  return (rank_mask & kWheel) == kWheel ? kWheel : 0;
}

uint16_t HighestRanks(uint16_t rank_mask, int count) noexcept {
  uint16_t selected = 0;
  for (int rank = 12; rank >= 0 && count > 0; --rank) {
    const uint16_t bit = static_cast<uint16_t>(1U << rank);
    if ((rank_mask & bit) != 0) {
      selected |= bit;
      --count;
    }
  }
  return selected;
}

int PrimePower(int rank, int count) noexcept {
  int product = 1;
  while (count-- > 0) product *= kRankPrimes[static_cast<size_t>(rank)];
  return product;
}

uint16_t ProductRank(int product);

uint16_t EvalSevenCactus(const std::array<Card, 7>& cards) noexcept {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint16_t, 4> suit_masks = {};
  uint16_t rank_mask = 0;
  for (Card card : cards) {
    const int rank = PokerRank(card) - 2;
    const uint16_t bit = static_cast<uint16_t>(1U << rank);
    ++rank_counts[static_cast<size_t>(rank)];
    suit_masks[static_cast<size_t>(SuitIndex(CardSuit(card)))] |= bit;
    rank_mask |= bit;
  }

  for (uint16_t suit_mask : suit_masks) {
    if (std::popcount(suit_mask) >= 5) {
      if (const uint16_t straight = HighestStraightMask(suit_mask)) {
        return hand_evaluator_tables::kCactusFlushes[straight];
      }
    }
  }

  int four = -1;
  int first_three = -1;
  int second_three = -1;
  std::array<int, 3> pairs = {-1, -1, -1};
  int pair_count = 0;
  for (int rank = 12; rank >= 0; --rank) {
    const int count = rank_counts[static_cast<size_t>(rank)];
    if (count == 4) {
      four = rank;
    } else if (count == 3) {
      if (first_three < 0) {
        first_three = rank;
      } else {
        second_three = rank;
      }
    } else if (count == 2) {
      pairs[static_cast<size_t>(pair_count++)] = rank;
    }
  }

  if (four >= 0) {
    const uint16_t remaining = static_cast<uint16_t>(
        rank_mask & ~(1U << four));
    const int kicker = 31 - std::countl_zero(
        static_cast<unsigned int>(remaining));
    return ProductRank(
        PrimePower(four, 4) * kRankPrimes[static_cast<size_t>(kicker)]);
  }

  if (first_three >= 0 && (second_three >= 0 || pair_count > 0)) {
    const int pair = second_three >= 0 ? second_three : pairs[0];
    return ProductRank(
        PrimePower(first_three, 3) * PrimePower(pair, 2));
  }

  for (uint16_t suit_mask : suit_masks) {
    if (std::popcount(suit_mask) >= 5) {
      return hand_evaluator_tables::kCactusFlushes[
          HighestRanks(suit_mask, 5)];
    }
  }

  if (const uint16_t straight = HighestStraightMask(rank_mask)) {
    return hand_evaluator_tables::kCactusUnique5[straight];
  }

  auto append_highest = [&](int product, uint16_t excluded, int count) {
    for (int rank = 12; rank >= 0 && count > 0; --rank) {
      if ((excluded & (1U << rank)) == 0 &&
          rank_counts[static_cast<size_t>(rank)] > 0) {
        product *= kRankPrimes[static_cast<size_t>(rank)];
        --count;
      }
    }
    return product;
  };

  if (first_three >= 0) {
    const int product = append_highest(
        PrimePower(first_three, 3),
        static_cast<uint16_t>(1U << first_three), 2);
    return ProductRank(product);
  }
  if (pair_count >= 2) {
    int product = PrimePower(pairs[0], 2) * PrimePower(pairs[1], 2);
    product = append_highest(
        product,
        static_cast<uint16_t>((1U << pairs[0]) | (1U << pairs[1])), 1);
    return ProductRank(product);
  }
  if (pair_count == 1) {
    const int product = append_highest(
        PrimePower(pairs[0], 2),
        static_cast<uint16_t>(1U << pairs[0]), 3);
    return ProductRank(product);
  }
  return hand_evaluator_tables::kCactusUnique5[
      HighestRanks(rank_mask, 5)];
}

const std::array<int, kDeckCardCount>& CactusCardTable() {
  static const std::array<int, kDeckCardCount> table = [] {
    std::array<int, kDeckCardCount> out = {};
    for (int id = 0; id < kDeckCardCount; ++id) {
      out[static_cast<size_t>(id)] =
          BuildCactusCard(kDeck[static_cast<size_t>(id)]);
    }
    return out;
  }();
  return table;
}

int CactusCard(Card card) {
  return CactusCardTable()[card.index()];
}

uint16_t ProductRank(int product) {
  const auto& products = hand_evaluator_tables::kCactusProducts;
  const auto it = std::lower_bound(
      products.begin(), products.end(), product,
      [](const auto& entry, int value) { return entry.first < value; });
  assert(it != products.end() && it->first == product);
  return it->second;
}

consteval bool ProductsAreStrictlySorted() {
  const auto& products = hand_evaluator_tables::kCactusProducts;
  for (size_t index = 1; index < products.size(); ++index) {
    if (products[index - 1].first >= products[index].first) {
      return false;
    }
  }
  return true;
}

static_assert(ProductsAreStrictlySorted());

uint16_t EvalFiveCactus(const std::array<int, 5>& cards) {
  const int rank_mask =
      (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16;
  if ((cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000) != 0) {
    return hand_evaluator_tables::kCactusFlushes[
        static_cast<size_t>(rank_mask)];
  }
  if (std::popcount(static_cast<unsigned int>(rank_mask)) == 5) {
    return hand_evaluator_tables::kCactusUnique5[
        static_cast<size_t>(rank_mask)];
  }
  const int product = (cards[0] & 0xFF) * (cards[1] & 0xFF) *
                      (cards[2] & 0xFF) * (cards[3] & 0xFF) *
                      (cards[4] & 0xFF);
  return ProductRank(product);
}

template <size_t N>
  requires(N >= 5 && N <= 7)
uint16_t EvalBestCactus(const std::array<Card, N>& cards) noexcept {
  std::array<int, 7> cactus_cards = {};
  for (size_t i = 0; i < N; ++i) {
    cactus_cards[i] = CactusCard(cards[i]);
  }

  uint16_t best = std::numeric_limits<uint16_t>::max();
  std::array<int, 5> combo;
  for (size_t a = 0; a + 4 < N; ++a) {
    combo[0] = cactus_cards[a];
    for (size_t b = a + 1; b + 3 < N; ++b) {
      combo[1] = cactus_cards[b];
      for (size_t c = b + 1; c + 2 < N; ++c) {
        combo[2] = cactus_cards[c];
        for (size_t d = c + 1; d + 1 < N; ++d) {
          combo[3] = cactus_cards[d];
          for (size_t e = d + 1; e < N; ++e) {
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

uint16_t HandValueImpl(ComboId hand, const RiverBoard& board) noexcept {
  std::array<Card, 7> cards;
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  std::copy(board.cards().begin(), board.cards().end(), cards.begin() + 2);
  return EvalSevenCactus(cards);
}

template <size_t N>
const hand_evaluator_tables::ScoreRecord& ScoreForCards(
    const std::array<Card, N>& cards) {
  return hand_evaluator_tables::kCactusScores[EvalBestCactus(cards)];
}

}  // namespace

HandEvaluation EvaluateFiveCards(const std::array<Card, 5>& cards) {
  return ToHandEvaluation(ScoreForCards(cards));
}

uint16_t HandValue(ComboId hand, const RiverBoard& board) noexcept {
  return HandValueImpl(hand, board);
}

int CompareHands(ComboId first,
                 ComboId second,
                 const RiverBoard& board) {
  return CompareCactusValues(HandValueImpl(first, board),
                             HandValueImpl(second, board));
}

}  // namespace poker
