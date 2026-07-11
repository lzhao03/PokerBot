#include "src/card_abstraction.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace poker {
namespace {

inline constexpr int kPrivateObservationBitsPerStreet = 6;
inline constexpr int kPublicObservationBitsPerStreet = 7;
inline constexpr int kSuitBuckets = 4;
inline constexpr int kStraightBuckets = 3;
inline constexpr int kHighBuckets = 3;

static_assert(kCoarsePrivateStreetObservationCount <
              (uint32_t{1} << kPrivateObservationBitsPerStreet));
static_assert(kCoarsePublicStreetObservationCount <
              (uint32_t{1} << kPublicObservationBitsPerStreet));

constexpr uint8_t StraightWindowDensity(uint16_t rank_mask) {
  uint8_t best = 0;
  for (int start = 0; start <= 8; ++start) {
    const uint8_t count = static_cast<uint8_t>(__builtin_popcount(
        static_cast<unsigned int>((rank_mask >> start) & 0x1F)));
    best = std::max(best, count);
  }
  const uint16_t wheel_mask = static_cast<uint16_t>((1u << 12) | 0x0F);
  const uint8_t wheel_count = static_cast<uint8_t>(__builtin_popcount(
      static_cast<unsigned int>(rank_mask & wheel_mask)));
  return std::max(best, wheel_count);
}

constexpr std::array<uint8_t, 8192> BuildStraightDensityTable() {
  std::array<uint8_t, 8192> table = {};
  for (size_t mask = 0; mask < table.size(); ++mask) {
    table[mask] = StraightWindowDensity(static_cast<uint16_t>(mask));
  }
  return table;
}

inline constexpr auto kStraightDensity = BuildStraightDensityTable();

int HighRankGroup(int rank) noexcept {
  if (rank >= 14) {
    return 0;
  }
  if (rank >= 12) {
    return 1;
  }
  return rank >= 9 ? 2 : 3;
}

int LowRankGroup(int rank) noexcept {
  if (rank >= 10) {
    return 0;
  }
  return rank >= 7 ? 1 : 2;
}

int HoleStrengthBucket(int high, int low, bool pair, bool suited) noexcept {
  if (pair || high == 14 || (high >= 13 && low >= 10)) {
    return 0;
  }
  const int gap = high - low;
  return (high >= 11 && low >= 8) || (suited && gap <= 2) ? 1 : 2;
}

int MadeBucket(const ComboInfo& combo,
               const BoardFeatures& features) noexcept {
  std::array<uint8_t, 13> rank_counts = features.rank_counts;
  ++rank_counts[static_cast<size_t>(PokerRank(combo.card0) - 2)];
  ++rank_counts[static_cast<size_t>(PokerRank(combo.card1) - 2)];

  int pairs = 0;
  int max_count = 0;
  for (uint8_t count : rank_counts) {
    pairs += count >= 2 ? 1 : 0;
    max_count = std::max(max_count, static_cast<int>(count));
  }
  if (max_count >= 3) {
    return 3;
  }
  if (pairs >= 2) {
    return 2;
  }
  return pairs == 1 ? 1 : 0;
}

int DrawBucket(const ComboInfo& combo,
               const BoardFeatures& features) noexcept {
  std::array<uint8_t, 4> suit_counts = features.suit_counts;
  uint16_t rank_mask = features.rank_mask;
  auto add_card = [&](Card card) {
    ++suit_counts[static_cast<size_t>(card.suit())];
    rank_mask |= static_cast<uint16_t>(1u << (PokerRank(card) - 2));
  };
  add_card(combo.card0);
  add_card(combo.card1);

  for (uint8_t count : suit_counts) {
    if (count >= 4) {
      return 2;
    }
  }
  return kStraightDensity[rank_mask] >= 4 ? 1 : 0;
}

constexpr int PublicShift(StreetKind street) noexcept {
  assert(street != StreetKind::kPreflop);
  return (static_cast<int>(street) - 1) * kPublicObservationBitsPerStreet;
}

PublicObservationId AdvanceCoarsePublic(PublicObservationId previous,
                                        StreetKind street,
                                        BoardBucketId bucket) noexcept {
  assert(bucket < kCoarsePublicStreetObservationCount);
  const int shift = PublicShift(street);
  constexpr uint64_t kSlotMask =
      (uint64_t{1} << kPublicObservationBitsPerStreet) - 1;
  const uint64_t slot_mask = kSlotMask << shift;
  assert((previous.value() & slot_mask) == 0);
  return PublicObservationId(
      previous.value() | ((static_cast<uint64_t>(bucket) + 1) << shift));
}

PublicObservationId ObserveCoarsePublic(StreetKind street,
                                        const Board& board) noexcept {
  PublicObservationId observation;
  if (street == StreetKind::kPreflop) {
    return observation;
  }

  const auto cards = BoardCards(board);
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation = AdvanceCoarsePublic(
      observation, StreetKind::kFlop,
      BoardTextureBucket(StreetKind::kFlop, BoardFeaturesFor(Board{flop})));
  if (street == StreetKind::kFlop) {
    return observation;
  }

  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation = AdvanceCoarsePublic(
      observation, StreetKind::kTurn,
      BoardTextureBucket(StreetKind::kTurn, BoardFeaturesFor(Board{turn})));
  if (street == StreetKind::kTurn) {
    return observation;
  }

  const RiverBoard river = DealRiver(turn, cards[4]);
  return AdvanceCoarsePublic(
      observation, StreetKind::kRiver,
      BoardTextureBucket(StreetKind::kRiver, BoardFeaturesFor(Board{river})));
}

PrivateObservationId AdvanceCoarsePrivate(
    PrivateObservationId previous,
    ComboId hand,
    const PublicPosition& child) noexcept {
  const StreetKind street = child.street();
  assert(street != StreetKind::kPreflop);
  const PrivateBucketId bucket =
      CoarsePrivateBucket(hand, street, child.features());
  assert(bucket < kCoarsePrivateStreetObservationCount);
  const int shift = static_cast<int>(street) *
                    kPrivateObservationBitsPerStreet;
  constexpr uint64_t kSlotMask =
      (uint64_t{1} << kPrivateObservationBitsPerStreet) - 1;
  const uint64_t slot_mask = kSlotMask << shift;
  assert((previous.value() & slot_mask) == 0);
  return PrivateObservationId(
      previous.value() | ((static_cast<uint64_t>(bucket) + 1) << shift));
}

}  // namespace

BoardFeatures BoardFeaturesFor(const Board& board) noexcept {
  BoardFeatures features;
  features.card_count = BoardCount(board);
  for (Card card : BoardCards(board)) {
    const size_t rank = static_cast<size_t>(card.rank());
    const size_t suit = static_cast<size_t>(card.suit());
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
  return features;
}

BoardBucketId BoardTextureBucket(StreetKind,
                                 const BoardFeatures& features) noexcept {
  if (features.card_count == 0) {
    return 0;
  }
  const int paired = features.max_rank_count >= 3
                         ? 2
                         : (features.max_rank_count == 2 ? 1 : 0);
  const int suited = features.max_suit_count >= 4
                         ? 3
                         : (features.max_suit_count >= 3
                                ? 2
                                : (features.max_suit_count == 2 ? 1 : 0));
  const int density = kStraightDensity[features.rank_mask & 0x1FFF];
  const int straight = density >= 4 ? 2 : (density >= 3 ? 1 : 0);
  const int high = features.max_rank >= 14
                       ? 0
                       : (features.max_rank >= 11 ? 1 : 2);
  return static_cast<BoardBucketId>(
      (((paired * kSuitBuckets) + suited) * kStraightBuckets + straight) *
          kHighBuckets +
      high);
}

PrivateBucketId CoarsePrivateBucket(
    ComboId hand,
    StreetKind street,
    const BoardFeatures& features) noexcept {
  const ComboInfo& combo = GetComboInfo(hand);
  const int rank0 = PokerRank(combo.card0);
  const int rank1 = PokerRank(combo.card1);
  const int high = std::max(rank0, rank1);
  const int low = std::min(rank0, rank1);
  const bool pair = rank0 == rank1;
  const bool suited = combo.card0.suit() == combo.card1.suit();
  if (street == StreetKind::kPreflop || features.card_count == 0) {
    const int shape = pair ? 0 : (suited ? 1 : 2);
    return static_cast<PrivateBucketId>(
        shape * 12 + HighRankGroup(high) * 3 + LowRankGroup(low));
  }
  return static_cast<PrivateBucketId>(
      MadeBucket(combo, features) * 9 + DrawBucket(combo, features) * 3 +
      HoleStrengthBucket(high, low, pair, suited));
}

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  StreetKind street,
                                  const Board& board) noexcept {
  switch (config.public_mode) {
    case PublicCardMode::kExactCanonical:
      return CanonicalPublicObservation(board);
    case PublicCardMode::kTexture:
      return ObserveCoarsePublic(street, board);
  }
}

PrivateObservationId ObservePrivate(const CardAbstractionConfig& config,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept {
  if (config.private_mode == PrivateCardMode::kExactCanonical) {
    return CanonicalizeObservation(hand, position.board()).private_observation;
  }

  PrivateObservationId observation =
      InitialPrivateObservation(config, hand);
  if (position.street() == StreetKind::kPreflop) {
    return observation;
  }
  const auto cards = BoardCards(position.board());
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation = AdvanceCoarsePrivate(
      observation, hand,
      PublicPosition::Root(config, StreetKind::kFlop, Board{flop}));
  if (position.street() == StreetKind::kFlop) {
    return observation;
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation = AdvanceCoarsePrivate(
      observation, hand,
      PublicPosition::Root(config, StreetKind::kTurn, Board{turn}));
  if (position.street() == StreetKind::kTurn) {
    return observation;
  }
  const RiverBoard river = DealRiver(turn, cards[4]);
  return AdvanceCoarsePrivate(
      observation, hand,
      PublicPosition::Root(config, StreetKind::kRiver, Board{river}));
}

PrivateObservationId InitialPrivateObservation(
    const CardAbstractionConfig& config,
    ComboId hand) noexcept {
  if (config.private_mode == PrivateCardMode::kExactCanonical) {
    return CanonicalizeObservation(hand, Board{PreflopBoard{}})
        .private_observation;
  }
  const PrivateBucketId bucket = CoarsePrivateBucket(
      hand, StreetKind::kPreflop, BoardFeatures{});
  return PrivateObservationId(static_cast<uint64_t>(bucket) + 1);
}

PrivateObservationId AdvancePrivateObservation(
    const CardAbstractionConfig& config,
    PrivateObservationId previous,
    ComboId hand,
    const PublicPosition& child) noexcept {
  if (config.private_mode == PrivateCardMode::kExactCanonical) {
    return CanonicalizeObservation(hand, child.board()).private_observation;
  }
  return AdvanceCoarsePrivate(previous, hand, child);
}

PublicPosition PublicPosition::Root(const CardAbstractionConfig& config,
                                    StreetKind street,
                                    Board board) {
  const BoardFeatures features = BoardFeaturesFor(board);
  const PublicObservationId observation = ObservePublic(config, street, board);
  return PublicPosition(street, std::move(board), observation, features);
}

PublicPosition PublicPosition::after_chance(
    const CardAbstractionConfig& config,
    StreetKind street,
    Board board) const {
  const BoardFeatures features = BoardFeaturesFor(board);
  PublicObservationId observation;
  if (config.public_mode == PublicCardMode::kExactCanonical) {
    observation = CanonicalPublicObservation(board);
  } else {
    observation = AdvanceCoarsePublic(
        observation_, street, BoardTextureBucket(street, features));
  }
  return PublicPosition(street, std::move(board), observation, features);
}

}  // namespace poker
