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
inline constexpr int kPrivateObservationBitsPerStreet = 6;
inline constexpr int kSuitBucketCount = 4;
inline constexpr int kStraightBucketCount = 3;
inline constexpr int kHighBucketCount = 3;

struct BoardFeatures {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint8_t, 4> suit_counts = {};
  uint16_t rank_mask = 0;
  uint8_t max_rank_count = 0;
  uint8_t max_suit_count = 0;
  uint8_t max_rank = 0;
};

int MaxStraightCards(uint16_t rank_mask) {
  constexpr uint16_t wheel = (uint16_t{1} << 12) | 0x0F;
  int best = std::popcount(static_cast<uint16_t>(rank_mask & wheel));
  for (int start = 0; start <= 8; ++start) {
    best = std::max(
        best, std::popcount(
                  static_cast<uint16_t>((rank_mask >> start) & 0x1F)));
  }
  return best;
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
  features.max_rank =
      std::max(features.max_rank, static_cast<uint8_t>(PokerRank(card)));
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
  const int high_bucket = features.max_rank >= 14
                              ? 0
                              : (features.max_rank >= 11 ? 1 : 2);
  int bucket = pair_bucket;
  bucket = bucket * kSuitBucketCount + suit_bucket;
  bucket = bucket * kStraightBucketCount + straight_bucket;
  return static_cast<uint64_t>(bucket * kHighBucketCount + high_bucket);
}

uint16_t Handcrafted36Bucket(
    ComboId hand,
    const BoardFeatures& features) noexcept {
  const auto hole_cards = hand.cards();
  const int rank0 = PokerRank(hole_cards[0]);
  const int rank1 = PokerRank(hole_cards[1]);
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

  std::array<uint8_t, 13> rank_counts = features.rank_counts;
  std::array<uint8_t, 4> suit_counts = features.suit_counts;
  uint16_t rank_mask = features.rank_mask;
  for (Card card : hole_cards) {
    ++rank_counts[static_cast<size_t>(PokerRank(card) - 2)];
    ++suit_counts[std::to_underlying(card.suit())];
    rank_mask |= static_cast<uint16_t>(1u << (PokerRank(card) - 2));
  }

  int pairs = 0;
  uint8_t max_count = 0;
  for (uint8_t count : rank_counts) {
    pairs += count >= 2 ? 1 : 0;
    max_count = std::max(max_count, count);
  }
  const int made_bucket =
      max_count >= 3 ? 3 : (pairs >= 2 ? 2 : (pairs == 1 ? 1 : 0));

  const bool flush_draw = std::ranges::any_of(
      suit_counts, [](uint8_t count) { return count >= 4; });
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

PublicObservationId EncodeBoardTextureHistory(const Board& board) noexcept {
  uint64_t observation = 0;
  BoardFeatures features;
  const auto cards = board.cards();
  for (size_t index = 0; index < cards.size(); ++index) {
    AddCard(features, cards[index]);
    if (index >= 2) {
      observation |= (BoardTextureBucket(features) + 1)
                     << ((index - 2) * kPublicObservationBitsPerStreet);
    }
  }
  return PublicObservationId(observation);
}

PrivateObservationId HandcraftedObservation(
    const CardAbstractionConfig& config,
    ComboId hand,
    const Board& board,
    PrivateObservationId previous) noexcept {
  if (config.recall_mode == RecallMode::CurrentBucketOnly) {
    return PrivateObservationId(
        Handcrafted36Bucket(hand, BoardFeaturesFor(board)) + 1);
  }
  if (previous != PrivateObservationId{} && board.count() >= 3) {
    return PrivateObservationId(
        std::to_underlying(previous) |
        (uint64_t{Handcrafted36Bucket(hand, BoardFeaturesFor(board))} + 1)
            << (kPrivateObservationBitsPerStreet * (board.count() - 2)));
  }

  uint64_t observation =
      uint64_t{Handcrafted36Bucket(hand, BoardFeatures{})} + 1;
  BoardFeatures features;
  const auto cards = board.cards();
  for (size_t index = 0; index < cards.size(); ++index) {
    AddCard(features, cards[index]);
    if (index >= 2) {
      observation |=
          (uint64_t{Handcrafted36Bucket(hand, features)} + 1)
          << (kPrivateObservationBitsPerStreet * (index - 1));
    }
  }
  return PrivateObservationId(observation);
}

}  // namespace

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  const Board& board) noexcept {
  switch (config.public_mode) {
    case PublicCardMode::ExactCanonical:
      return CanonicalPublicObservation(board);
    case PublicCardMode::Texture:
      return EncodeBoardTextureHistory(board);
  }
}

PrivateObservationId ObservePrivate(const CardAbstractionConfig& config,
                                    ComboId hand,
                                    const Board& board,
                                    PrivateObservationId previous) noexcept {
  switch (config.private_kind) {
    case PrivateAbstractionKind::ExactCanonical:
      return CanonicalPrivateObservation(hand, board);
    case PrivateAbstractionKind::Handcrafted36:
      return HandcraftedObservation(config, hand, board, previous);
  }
}

PublicPosition::PublicPosition(const CardAbstractionConfig& config,
                               const Board& board)
    : board_(board), observation_(ObservePublic(config, board_)) {}

}  // namespace poker
