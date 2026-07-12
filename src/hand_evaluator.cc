#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

#include "generated/hand_evaluator_tables.h"

namespace poker {
namespace {

static constexpr std::array<int, 13> kRankPrimes = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

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
    suit_masks[std::to_underlying(card.suit())] |= bit;
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

uint16_t EvaluateHand(ComboId hand, const Board& board) noexcept {
  assert(board.count() == kMaxBoardCards);
  std::array<Card, 7> cards;
  const auto hole_cards = hand.cards();
  cards[0] = hole_cards[0];
  cards[1] = hole_cards[1];
  std::copy(board.cards().begin(), board.cards().end(), cards.begin() + 2);
  return EvalSevenCactus(cards);
}

}  // namespace

int CompareHands(ComboId first,
                 ComboId second,
                 const Board& board) {
  const uint16_t first_value = EvaluateHand(first, board);
  const uint16_t second_value = EvaluateHand(second, board);
  return first_value < second_value ? 1 : first_value > second_value ? -1 : 0;
}

}  // namespace poker
