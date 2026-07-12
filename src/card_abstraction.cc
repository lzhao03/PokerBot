#include "src/card_abstraction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace poker {
namespace {

inline constexpr int kPublicObservationBitsPerStreet = 7;
inline constexpr int kSuitBuckets = 4;
inline constexpr int kStraightBuckets = 3;
inline constexpr int kHighBuckets = 3;

static_assert(kCoarsePublicStreetObservationCount <
              (uint32_t{1} << kPublicObservationBitsPerStreet));

constexpr std::array<uint8_t, 8192> BuildStraightDensityTable() {
  const auto straight_window_density = [](size_t rank_mask) {
    int best = 0;
    for (int start = 0; start <= 8; ++start) {
      best = std::max(
          best, std::popcount((rank_mask >> start) & size_t{0x1F}));
    }
    constexpr size_t wheel_mask = (size_t{1} << 12) | 0x0F;
    return std::max(best, std::popcount(rank_mask & wheel_mask));
  };

  std::array<uint8_t, 8192> table = {};
  for (size_t mask = 0; mask < table.size(); ++mask) {
    table[mask] = static_cast<uint8_t>(straight_window_density(mask));
  }
  return table;
}

inline const auto kStraightDensity = BuildStraightDensityTable();

PublicObservationId AppendBoardTextureBucket(PublicObservationId previous,
                                        StreetKind street,
                                        BoardBucketId bucket) noexcept {
  assert(street != StreetKind::Preflop);
  assert(bucket < kCoarsePublicStreetObservationCount);
  const int shift = (std::to_underlying(street) - 1) * kPublicObservationBitsPerStreet;
  return PublicObservationId(previous.value() | ((bucket + 1) << shift));
}

PublicObservationId EncodeBoardTextureHistory(const Board& board) noexcept {
  PublicObservationId observation;
  const auto cards = BoardCards(board);
  if (cards.empty()) return observation;

  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation = AppendBoardTextureBucket(
      observation, StreetKind::Flop,
      BoardTextureBucket(StreetKind::Flop, BoardFeaturesFor(Board{flop})));
  if (cards.size() == 3) return observation;

  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation = AppendBoardTextureBucket(
      observation, StreetKind::Turn,
      BoardTextureBucket(StreetKind::Turn, BoardFeaturesFor(Board{turn})));
  if (cards.size() == 4) return observation;

  const RiverBoard river = DealRiver(turn, cards[4]);
  return AppendBoardTextureBucket(
      observation, StreetKind::River,
      BoardTextureBucket(StreetKind::River, BoardFeaturesFor(Board{river})));
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

PrivateBucketId Handcrafted36Bucket(
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
  if (street == StreetKind::Preflop || features.card_count == 0) {
    const int shape = pair ? 0 : (suited ? 1 : 2);
    const int high_group =
        high >= 14 ? 0 : (high >= 12 ? 1 : (high >= 9 ? 2 : 3));
    const int low_group = low >= 10 ? 0 : (low >= 7 ? 1 : 2);
    return static_cast<PrivateBucketId>(
        shape * 12 + high_group * 3 + low_group);
  }

  std::array<uint8_t, 13> rank_counts = features.rank_counts;
  std::array<uint8_t, 4> suit_counts = features.suit_counts;
  uint16_t rank_mask = features.rank_mask;
  for (Card card : {combo.card0, combo.card1}) {
    ++rank_counts[static_cast<size_t>(PokerRank(card) - 2)];
    ++suit_counts[static_cast<size_t>(card.suit())];
    rank_mask |= static_cast<uint16_t>(1u << (PokerRank(card) - 2));
  }

  int pairs = 0;
  int max_count = 0;
  for (uint8_t count : rank_counts) {
    pairs += count >= 2 ? 1 : 0;
    max_count = std::max(max_count, static_cast<int>(count));
  }
  const int made =
      max_count >= 3 ? 3 : (pairs >= 2 ? 2 : (pairs == 1 ? 1 : 0));

  const bool flush_draw = std::ranges::any_of(
      suit_counts, [](uint8_t count) { return count >= 4; });
  const int draw =
      flush_draw ? 2 : (kStraightDensity[rank_mask] >= 4 ? 1 : 0);

  const int gap = high - low;
  const int hole_strength =
      pair || high == 14 || (high >= 13 && low >= 10)
          ? 0
          : ((high >= 11 && low >= 8) || (suited && gap <= 2) ? 1 : 2);
  return static_cast<PrivateBucketId>(
      made * 9 + draw * 3 + hole_strength);
}

namespace {

PrivateObservationId HandcraftedObservation(
    const CardAbstractionConfig& config,
    ComboId hand,
    const PublicPosition& position) noexcept {
  const PrivateBucketId current = Handcrafted36Bucket(
      hand, position.street(), position.features());
  if (config.recall_mode == RecallMode::CurrentBucketOnly) {
    return PrivateObservationId(static_cast<uint64_t>(current) + 1);
  }

  uint64_t observation = static_cast<uint64_t>(Handcrafted36Bucket(
      hand, StreetKind::Preflop, BoardFeatures{})) + 1;
  if (position.street() == StreetKind::Preflop) {
    return PrivateObservationId(observation);
  }
  const auto cards = BoardCards(position.board());
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation |=
      (static_cast<uint64_t>(Handcrafted36Bucket(
           hand, StreetKind::Flop, BoardFeaturesFor(Board{flop}))) + 1)
      << 6;
  if (position.street() == StreetKind::Flop) {
    return PrivateObservationId(observation);
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation |=
      (static_cast<uint64_t>(Handcrafted36Bucket(
           hand, StreetKind::Turn, BoardFeaturesFor(Board{turn}))) + 1)
      << 12;
  if (position.street() == StreetKind::Turn) {
    return PrivateObservationId(observation);
  }
  const RiverBoard river = DealRiver(turn, cards[4]);
  observation |=
      (static_cast<uint64_t>(Handcrafted36Bucket(
           hand, StreetKind::River, BoardFeaturesFor(Board{river}))) + 1)
      << 18;
  return PrivateObservationId(observation);
}

}  // namespace

PublicObservationId ObservePublic(const CardAbstraction& abstraction,
                                  const Board& board) noexcept {
  switch (abstraction.config().public_mode) {
    case PublicCardMode::ExactCanonical:
      return CanonicalPublicObservation(board);
    case PublicCardMode::Texture:
      return EncodeBoardTextureHistory(board);
  }
}

PrivateObservationId ObservePrivate(const CardAbstraction& abstraction,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept {
  switch (abstraction.config().private_kind) {
    case PrivateAbstractionKind::ExactCanonical:
      return CanonicalizeObservation(hand, position.board())
          .private_observation;
    case PrivateAbstractionKind::Handcrafted36:
      return HandcraftedObservation(abstraction.config(), hand, position);
  }
}

PublicPosition PublicPosition::Root(const CardAbstraction& abstraction,
                                    StreetKind street,
                                    Board board) {
  const BoardFeatures features = BoardFeaturesFor(board);
  const PublicObservationId observation = ObservePublic(abstraction, board);
  return PublicPosition(street, std::move(board), observation, features);
}

PublicPosition PublicPosition::after_chance(
    const CardAbstraction& abstraction,
    StreetKind street,
    Board board) const {
  const BoardFeatures features = BoardFeaturesFor(board);
  PublicObservationId observation;
  if (abstraction.config().public_mode == PublicCardMode::ExactCanonical) {
    observation = CanonicalPublicObservation(board);
  } else {
    observation = AppendBoardTextureBucket(
        observation_, street, BoardTextureBucket(street, features));
  }
  return PublicPosition(street, std::move(board), observation, features);
}

}  // namespace poker
