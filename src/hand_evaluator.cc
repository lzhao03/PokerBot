#include "src/hand_evaluator.h"

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

struct HandFeatures {
  std::array<uint16_t, 4> suit_masks = {};
  uint16_t rank_mask = 0;
  uint16_t pair_mask = 0;
  uint16_t trip_mask = 0;
  uint16_t quad_mask = 0;
};

void AddCard(HandFeatures& features, Card card) noexcept {
  const size_t rank = std::to_underlying(card.rank());
  const uint16_t bit = static_cast<uint16_t>(1U << rank);
  features.suit_masks[std::to_underlying(card.suit())] |= bit;
  features.quad_mask |= features.trip_mask & bit;
  features.trip_mask |= features.pair_mask & bit;
  features.pair_mask |= features.rank_mask & bit;
  features.rank_mask |= bit;
}

uint16_t HighestStraightMask(uint16_t rank_mask) noexcept {
  uint16_t starts = rank_mask;
  for (int shift = 1; shift < 5; ++shift) {
    starts &= static_cast<uint16_t>(rank_mask >> shift);
  }
  if (starts != 0) {
    return static_cast<uint16_t>(0x1F << (std::bit_width(starts) - 1));
  }
  constexpr uint16_t kWheel = (1U << 12) | 0x0F;
  return (rank_mask & kWheel) == kWheel ? kWheel : 0;
}

uint16_t HighestFiveRanks(uint16_t rank_mask) noexcept {
  for (int excess = std::popcount(rank_mask) - 5;
       excess > 0; --excess) {
    rank_mask &= static_cast<uint16_t>(rank_mask - 1);
  }
  return rank_mask;
}

int PrimePower(int rank, int count) noexcept {
  int product = 1;
  while (count-- > 0) product *= kRankPrimes[static_cast<size_t>(rank)];
  return product;
}

constexpr int kProductTableBits = 13;
constexpr size_t kProductTableSize = 1 << kProductTableBits;
constexpr uint32_t kProductHashMultiplier = 0x9E3779B1;

constexpr size_t ProductSlot(int product) noexcept {
  return (static_cast<uint32_t>(product) * kProductHashMultiplier) >>
      (32 - kProductTableBits);
}

constexpr auto kProductRanks = [] {
  std::array<std::pair<int, uint16_t>, kProductTableSize> ranks = {};
  for (const auto& entry : hand_evaluator_tables::kCactusProducts) {
    size_t slot = ProductSlot(entry.first);
    while (ranks[slot].first != 0) {
      slot = (slot + 1) & (kProductTableSize - 1);
    }
    ranks[slot] = entry;
  }
  return ranks;
}();

[[gnu::noinline]] uint16_t ProductRank(int product) {
  size_t slot = ProductSlot(product);
  while (kProductRanks[slot].first != product) {
    assert(kProductRanks[slot].first != 0);
    slot = (slot + 1) & (kProductTableSize - 1);
  }
  return kProductRanks[slot].second;
}

uint16_t EvalSevenCactus(const HandFeatures& features) noexcept {
  const auto& suit_masks = features.suit_masks;
  const uint16_t rank_mask = features.rank_mask;

  uint16_t flush_mask = 0;
  for (uint16_t suit_mask : suit_masks) {
    if (std::popcount(suit_mask) >= 5) {
      flush_mask = suit_mask;
      break;
    }
  }
  if (flush_mask != 0) {
    if (const uint16_t straight = HighestStraightMask(flush_mask)) {
      return hand_evaluator_tables::kCactusFlushes[straight];
    }
  }

  if (features.quad_mask != 0) {
    const int four = std::bit_width(features.quad_mask) - 1;
    const uint16_t remaining = static_cast<uint16_t>(
        rank_mask & ~(1U << four));
    const int kicker = std::bit_width(remaining) - 1;
    return ProductRank(
        PrimePower(four, 4) * kRankPrimes[static_cast<size_t>(kicker)]);
  }

  if (features.trip_mask != 0 && std::popcount(features.pair_mask) > 1) {
    const int three = std::bit_width(features.trip_mask) - 1;
    const int pair = std::bit_width(static_cast<uint16_t>(
        features.pair_mask & ~(1U << three))) - 1;
    return ProductRank(
        PrimePower(three, 3) * PrimePower(pair, 2));
  }

  if (flush_mask != 0) {
    return hand_evaluator_tables::kCactusFlushes[
        HighestFiveRanks(flush_mask)];
  }

  if (const uint16_t straight = HighestStraightMask(rank_mask)) {
    return hand_evaluator_tables::kCactusUnique5[straight];
  }

  auto append_highest = [](int product, uint16_t ranks, int count) {
    while (count-- > 0) {
      const int rank = std::bit_width(ranks) - 1;
      product *= kRankPrimes[static_cast<size_t>(rank)];
      ranks &= static_cast<uint16_t>(~(1U << rank));
    }
    return product;
  };

  if (features.trip_mask != 0) {
    const int three = std::bit_width(features.trip_mask) - 1;
    const int product = append_highest(
        PrimePower(three, 3),
        static_cast<uint16_t>(rank_mask & ~(1U << three)), 2);
    return ProductRank(product);
  }
  if (std::popcount(features.pair_mask) >= 2) {
    const int first = std::bit_width(features.pair_mask) - 1;
    const int second = std::bit_width(static_cast<uint16_t>(
        features.pair_mask & ~(1U << first))) - 1;
    int product = PrimePower(first, 2) * PrimePower(second, 2);
    product = append_highest(product, static_cast<uint16_t>(
        rank_mask & ~(1U << first) & ~(1U << second)), 1);
    return ProductRank(product);
  }
  if (features.pair_mask != 0) {
    const int pair = std::bit_width(features.pair_mask) - 1;
    const int product = append_highest(
        PrimePower(pair, 2),
        static_cast<uint16_t>(rank_mask & ~(1U << pair)), 3);
    return ProductRank(product);
  }
  return hand_evaluator_tables::kCactusUnique5[
      HighestFiveRanks(rank_mask)];
}

uint16_t EvaluateHand(ComboId hand, HandFeatures features) noexcept {
  for (Card card : hand.cards()) AddCard(features, card);
  return EvalSevenCactus(features);
}

}  // namespace

int CompareHands(ComboId first,
                 ComboId second,
                 const Board& board) {
  assert(board.count() == kMaxBoardCards);
  HandFeatures board_features;
  for (Card card : board.cards()) AddCard(board_features, card);
  const uint16_t first_value = EvaluateHand(first, board_features);
  const uint16_t second_value = EvaluateHand(second, board_features);
  return first_value < second_value ? 1 : first_value > second_value ? -1 : 0;
}

}  // namespace poker
