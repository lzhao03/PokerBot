#include "src/hand_evaluator.h"

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

namespace poker {
namespace {

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

int HighestStraight(uint16_t rank_mask) noexcept {
  uint16_t starts = rank_mask;
  for (int shift = 1; shift < 5; ++shift) {
    starts &= static_cast<uint16_t>(rank_mask >> shift);
  }
  if (starts != 0) {
    return std::bit_width(starts) + 3;
  }
  constexpr uint16_t kWheel = (1U << 12) | 0x0F;
  return (rank_mask & kWheel) == kWheel ? 3 : -1;
}

uint16_t HighestRanks(uint16_t rank_mask, int count) noexcept {
  for (int excess = std::popcount(rank_mask) - count;
       excess > 0; --excess) {
    rank_mask &= static_cast<uint16_t>(rank_mask - 1);
  }
  return rank_mask;
}

enum class HandCategory : uint32_t {
  HighCard,
  Pair,
  TwoPair,
  ThreeOfAKind,
  Straight,
  Flush,
  FullHouse,
  FourOfAKind,
  StraightFlush,
};

constexpr uint32_t PackHand(HandCategory category,
                            uint16_t primary,
                            uint16_t kickers = 0) noexcept {
  return (std::to_underlying(category) << 26) |
      (static_cast<uint32_t>(primary) << 13) | kickers;
}

uint32_t EvaluateSeven(const HandFeatures& features) noexcept {
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
    if (const int straight = HighestStraight(flush_mask); straight >= 0) {
      return PackHand(HandCategory::StraightFlush,
                      static_cast<uint16_t>(straight));
    }
  }

  if (features.quad_mask != 0) {
    const int four = std::bit_width(features.quad_mask) - 1;
    const uint16_t remaining = static_cast<uint16_t>(
        rank_mask & ~(1U << four));
    const int kicker = std::bit_width(remaining) - 1;
    return PackHand(HandCategory::FourOfAKind,
                    static_cast<uint16_t>(four),
                    static_cast<uint16_t>(kicker));
  }

  if (features.trip_mask != 0 && std::popcount(features.pair_mask) > 1) {
    const int three = std::bit_width(features.trip_mask) - 1;
    const int pair = std::bit_width(static_cast<uint16_t>(
        features.pair_mask & ~(1U << three))) - 1;
    return PackHand(HandCategory::FullHouse,
                    static_cast<uint16_t>(three),
                    static_cast<uint16_t>(pair));
  }

  if (flush_mask != 0) {
    return PackHand(HandCategory::Flush, HighestRanks(flush_mask, 5));
  }

  if (const int straight = HighestStraight(rank_mask); straight >= 0) {
    return PackHand(HandCategory::Straight,
                    static_cast<uint16_t>(straight));
  }

  if (features.trip_mask != 0) {
    const int three = std::bit_width(features.trip_mask) - 1;
    return PackHand(HandCategory::ThreeOfAKind,
                    static_cast<uint16_t>(three),
                    HighestRanks(static_cast<uint16_t>(
                        rank_mask & ~(1U << three)), 2));
  }
  if (std::popcount(features.pair_mask) >= 2) {
    const uint16_t pairs = HighestRanks(features.pair_mask, 2);
    return PackHand(HandCategory::TwoPair, pairs,
                    HighestRanks(static_cast<uint16_t>(rank_mask & ~pairs), 1));
  }
  if (features.pair_mask != 0) {
    const int pair = std::bit_width(features.pair_mask) - 1;
    return PackHand(HandCategory::Pair,
                    static_cast<uint16_t>(pair),
                    HighestRanks(static_cast<uint16_t>(
                        rank_mask & ~(1U << pair)), 3));
  }
  return PackHand(HandCategory::HighCard, HighestRanks(rank_mask, 5));
}

uint32_t EvaluateHand(ComboId hand, HandFeatures features) noexcept {
  for (Card card : hand.cards()) AddCard(features, card);
  return EvaluateSeven(features);
}

}  // namespace

int CompareHands(ComboId first,
                 ComboId second,
                 const Board& board) {
  assert(board.count() == kMaxBoardCards);
  HandFeatures board_features;
  for (Card card : board.cards()) AddCard(board_features, card);
  const uint32_t first_value = EvaluateHand(first, board_features);
  const uint32_t second_value = EvaluateHand(second, board_features);
  return first_value > second_value ? 1 : first_value < second_value ? -1 : 0;
}

}  // namespace poker
