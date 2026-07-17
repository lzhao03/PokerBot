#include "src/card_abstraction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "src/card_canonicalization.h"

namespace poker {
namespace {

inline constexpr int kPublicObservationBitsPerStreet = 7;
inline constexpr int kSuitBucketCount = 4;
inline constexpr int kStraightBucketCount = 3;
inline constexpr int kHighBucketCount = 3;
inline constexpr uint64_t kTextureBucketCount =
    3 * kSuitBucketCount * kStraightBucketCount * kHighBucketCount;
inline constexpr std::array<uint64_t, 3> kTextureBuckets = {108, 108, 108};
inline constexpr std::array<uint64_t, 3> kCompactTextureBuckets = {16, 16, 64};
inline constexpr std::array<uint32_t, 4> kPrivateObservationPlaces = {
    1, 37, 37 * 37, 37 * 37 * 37};

struct BoardFeatures {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint8_t, 4> suit_counts = {};
  uint16_t rank_mask = 0;
  uint8_t max_rank_count = 0;
  uint8_t max_suit_count = 0;
};

const auto kStraightCardsByRankMask = [] {
  std::array<uint8_t, 1 << 13> values = {};
  for (size_t rank_mask = 0; rank_mask < values.size(); ++rank_mask) {
    constexpr uint16_t kWheel = (uint16_t{1} << 12) | 0x0F;
    int best = std::popcount(static_cast<uint16_t>(rank_mask & kWheel));
    for (int start = 0; start <= 8; ++start) {
      best = std::max(best, std::popcount(static_cast<uint16_t>(
                                (rank_mask >> start) & 0x1F)));
    }
    values[rank_mask] = static_cast<uint8_t>(best);
  }
  return values;
}();

int MaxStraightCards(uint16_t rank_mask) {
  return kStraightCardsByRankMask[rank_mask];
}

void AddCard(BoardFeatures& features, Card card) noexcept {
  const size_t rank = std::to_underlying(card.rank());
  const size_t suit = std::to_underlying(card.suit());
  ++features.rank_counts[rank];
  ++features.suit_counts[suit];
  features.max_rank_count =
      std::max(features.max_rank_count, features.rank_counts[rank]);
  features.max_suit_count =
      std::max(features.max_suit_count, features.suit_counts[suit]);
  features.rank_mask |= static_cast<uint16_t>(1u << rank);
}

BoardFeatures BoardFeaturesFor(const Board& board) noexcept {
  BoardFeatures features;
  for (Card card : board.cards()) AddCard(features, card);
  return features;
}

uint64_t BoardTextureBucket(const BoardFeatures& features) noexcept {
  if (features.rank_mask == 0) {
    return 0;
  }
  const int pair_bucket = features.max_rank_count >= 3
                              ? 2
                              : (features.max_rank_count == 2 ? 1 : 0);
  const int suit_bucket = features.max_suit_count >= 4
                              ? 3
                              : (features.max_suit_count >= 3
                                     ? 2
                                     : (features.max_suit_count == 2 ? 1 : 0));
  const int straight_cards = MaxStraightCards(features.rank_mask);
  const int straight_bucket =
      straight_cards >= 4 ? 2 : (straight_cards >= 3 ? 1 : 0);
  const int high_rank = std::bit_width(features.rank_mask) + 1;
  const int high_bucket = high_rank >= 14 ? 0 : (high_rank >= 11 ? 1 : 2);
  int bucket = pair_bucket;
  bucket = bucket * kSuitBucketCount + suit_bucket;
  bucket = bucket * kStraightBucketCount + straight_bucket;
  return static_cast<uint64_t>(bucket * kHighBucketCount + high_bucket);
}

uint16_t Handcrafted36Bucket(
    ComboId hand,
    const BoardFeatures& features) noexcept {
  const auto hole_cards = hand.cards();
  const size_t rank_index0 = std::to_underlying(hole_cards[0].rank());
  const size_t rank_index1 = std::to_underlying(hole_cards[1].rank());
  const int rank0 = static_cast<int>(rank_index0) + 2;
  const int rank1 = static_cast<int>(rank_index1) + 2;
  const int high = std::max(rank0, rank1);
  const int low = std::min(rank0, rank1);
  const bool pair = rank0 == rank1;
  const bool suited = hole_cards[0].suit() == hole_cards[1].suit();
  if (features.rank_mask == 0) {
    const int shape_bucket = pair ? 0 : (suited ? 1 : 2);
    const int high_bucket =
        high >= 14 ? 0 : (high >= 12 ? 1 : (high >= 9 ? 2 : 3));
    const int low_bucket = low >= 10 ? 0 : (low >= 7 ? 1 : 2);
    return static_cast<uint16_t>(
        shape_bucket * 12 + high_bucket * 3 + low_bucket);
  }

  int pairs = static_cast<int>(std::ranges::count_if(
      features.rank_counts,
      [](uint8_t count) { return count >= 2; }));
  uint8_t max_count = features.max_rank_count;
  auto add_hole_rank = [&](size_t rank, uint8_t count) {
    const uint8_t board_count = features.rank_counts[rank];
    if (board_count < 2 && board_count + count >= 2) ++pairs;
    max_count = std::max(
        max_count, static_cast<uint8_t>(board_count + count));
  };
  if (pair) {
    add_hole_rank(rank_index0, 2);
  } else {
    add_hole_rank(rank_index0, 1);
    add_hole_rank(rank_index1, 1);
  }
  const int made_bucket =
      max_count >= 3 ? 3 : (pairs >= 2 ? 2 : (pairs == 1 ? 1 : 0));

  const size_t suit0 = std::to_underlying(hole_cards[0].suit());
  const size_t suit1 = std::to_underlying(hole_cards[1].suit());
  const bool flush_draw = features.max_suit_count >= 4 ||
      features.suit_counts[suit0] + 1 + (suit0 == suit1) >= 4 ||
      features.suit_counts[suit1] + 1 >= 4;
  const uint16_t rank_mask = static_cast<uint16_t>(
      features.rank_mask | (1u << rank_index0) | (1u << rank_index1));
  const int draw_bucket =
      flush_draw ? 2 : (MaxStraightCards(rank_mask) >= 4 ? 1 : 0);

  const int gap = high - low;
  const int strength_bucket =
      pair || high == 14 || (high >= 13 && low >= 10)
          ? 0
          : ((high >= 11 && low >= 8) || (suited && gap <= 2) ? 1 : 2);
  return static_cast<uint16_t>(
      made_bucket * 9 + draw_bucket * 3 + strength_bucket);
}

PublicObservationId EncodeBoardTextureHistory(
    const Board& board,
    const std::array<uint64_t, 3>& buckets_per_street,
    BoardFeatures* final_features) noexcept {
  uint64_t observation = 0;
  BoardFeatures features;
  const auto cards = board.cards();
  for (size_t index = 0; index < cards.size(); ++index) {
    AddCard(features, cards[index]);
    if (index >= 2) {
      const uint64_t bucket = BoardTextureBucket(features) *
                              buckets_per_street[index - 2] /
                              kTextureBucketCount;
      observation |= (bucket + 1)
                     << ((index - 2) * kPublicObservationBitsPerStreet);
    }
  }
  if (final_features != nullptr) *final_features = features;
  return PublicObservationId(observation);
}

PublicObservationId ObservePublicImpl(const CardAbstractionConfig& config,
                                      const Board& board,
                                      BoardFeatures* features) noexcept {
  switch (config.public_mode) {
    case PublicCardMode::ExactCanonical:
      if (features != nullptr) *features = BoardFeaturesFor(board);
      return CanonicalPublicObservation(board);
    case PublicCardMode::Texture:
      return EncodeBoardTextureHistory(board, kTextureBuckets, features);
    case PublicCardMode::CompactTexture:
      return EncodeBoardTextureHistory(board, kCompactTextureBuckets,
                                       features);
  }
}

PrivateObservationId HandcraftedObservation(
    RecallMode recall_mode,
    ComboId hand,
    const Board& board,
    PrivateObservationId previous,
    const BoardFeatures& board_features) noexcept {
  if (recall_mode == RecallMode::CurrentBucketOnly) {
    return PrivateObservationId(
        Handcrafted36Bucket(hand, board_features) + 1);
  }
  if (previous != PrivateObservationId{} && board.count() >= 3) {
    return PrivateObservationId(
        std::to_underlying(previous) +
        (uint32_t{Handcrafted36Bucket(hand, board_features)} + 1)
            * kPrivateObservationPlaces[board.count() - 2]);
  }

  uint32_t observation =
      uint32_t{Handcrafted36Bucket(hand, BoardFeatures{})} + 1;
  BoardFeatures features;
  const auto cards = board.cards();
  for (size_t index = 0; index < cards.size(); ++index) {
    AddCard(features, cards[index]);
    if (index >= 2) {
      observation +=
          (uint32_t{Handcrafted36Bucket(hand, features)} + 1)
          * kPrivateObservationPlaces[index - 1];
    }
  }
  return PrivateObservationId(observation);
}

}  // namespace

PrivateObservationId ObservePrivate(ComboId hand,
                                    const PublicPosition& position,
                                    PrivateObservationId previous) noexcept {
  switch (position.private_kind_) {
    case PrivateAbstractionKind::ExactCanonical:
      return CanonicalPrivateObservation(hand, position.board_);
    case PrivateAbstractionKind::Handcrafted36:
      return HandcraftedObservation(
          position.recall_mode_, hand, position.board_, previous,
          {position.rank_counts_, position.suit_counts_, position.rank_mask_,
           position.max_rank_count_, position.max_suit_count_});
  }
}

PublicPosition::PublicPosition(const CardAbstractionConfig& config,
                               const Board& board)
    : board_(board),
      private_kind_(config.private_kind),
      recall_mode_(config.recall_mode) {
  BoardFeatures features;
  observation_ = ObservePublicImpl(config, board_, &features);
  rank_counts_ = features.rank_counts;
  suit_counts_ = features.suit_counts;
  rank_mask_ = features.rank_mask;
  max_rank_count_ = features.max_rank_count;
  max_suit_count_ = features.max_suit_count;
}

}  // namespace poker
