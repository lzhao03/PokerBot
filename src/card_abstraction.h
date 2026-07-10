#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "src/build_flags.h"
#include "src/combo.h"
#include "src/game_state.h"
#include "src/poker_types.h"

namespace poker {

using PrivateBucketId = uint16_t;

inline constexpr uint32_t kCoarsePrivateStreetObservationCount = 36;
inline constexpr uint32_t kCoarsePublicStreetObservationCount = 108;

struct ExactChanceObservation {
  std::array<CardId, 3> cards = {};
  uint8_t count = 0;

  friend bool operator==(const ExactChanceObservation&,
                         const ExactChanceObservation&) = default;
};

struct PublicStreetObservation {
  // Transitional graph key. Exact-mode histories should use exact_cards.
  BoardBucketId value = 0;
  ExactChanceObservation exact_cards;

  friend bool operator==(const PublicStreetObservation&,
                         const PublicStreetObservation&) = default;
};

struct PrivateStreetObservation {
  PrivateBucketId value = 0;

  friend bool operator==(const PrivateStreetObservation&,
                         const PrivateStreetObservation&) = default;
};

struct BoardFeatures {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint8_t, 4> suit_counts = {};
  uint16_t rank_mask = 0;
  uint8_t card_count = 0;
  uint8_t max_rank_count = 0;
  uint8_t max_suit_count = 0;
  uint8_t max_rank = 0;

  friend bool operator==(const BoardFeatures&, const BoardFeatures&) = default;
};

namespace card_abstraction_detail {

inline constexpr int kSuitBuckets = 4;
inline constexpr int kStraightBuckets = 3;
inline constexpr int kHighBuckets = 3;
constexpr uint8_t StraightWindowDensity(uint16_t rank_mask) {
  uint8_t best = 0;
  for (int start = 0; start <= 8; ++start) {
    const uint8_t count = static_cast<uint8_t>(
        __builtin_popcount(static_cast<unsigned int>((rank_mask >> start) &
                                                     0x1F)));
    best = std::max(best, count);
  }
  const uint16_t wheel_mask = static_cast<uint16_t>((1u << 12) | 0x0F);
  const uint8_t wheel_count = static_cast<uint8_t>(
      __builtin_popcount(static_cast<unsigned int>(rank_mask & wheel_mask)));
  return std::max(best, wheel_count);
}

constexpr std::array<uint8_t, 8192> BuildStraightDensityTable() {
  std::array<uint8_t, 8192> table = {};
  for (size_t mask = 0; mask < table.size(); ++mask) {
    table[mask] = StraightWindowDensity(static_cast<uint16_t>(mask));
  }
  return table;
}

inline constexpr std::array<uint8_t, 8192> kStraightDensity =
    BuildStraightDensityTable();

inline int HighRankGroup(int rank) {
  if (rank >= 14) {
    return 0;
  }
  if (rank >= 12) {
    return 1;
  }
  return rank >= 9 ? 2 : 3;
}

inline int LowRankGroup(int rank) {
  if (rank >= 10) {
    return 0;
  }
  return rank >= 7 ? 1 : 2;
}

inline int HoleStrengthBucket(int high, int low, bool pair, bool suited) {
  if (pair || high == 14 || (high >= 13 && low >= 10)) {
    return 0;
  }
  const int gap = high - low;
  if ((high >= 11 && low >= 8) || (suited && gap <= 2)) {
    return 1;
  }
  return 2;
}

inline int MadeBucket(const ComboInfo& combo, const BoardFeatures& features) {
  std::array<uint8_t, 13> rank_counts = features.rank_counts;
  ++rank_counts[static_cast<size_t>(RankFromCardId(combo.card0) - 2)];
  ++rank_counts[static_cast<size_t>(RankFromCardId(combo.card1) - 2)];

  int pairs = 0;
  int max_count = 0;
  for (uint8_t count : rank_counts) {
    if (count >= 2) {
      ++pairs;
    }
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

inline int DrawBucket(const ComboInfo& combo, const BoardFeatures& features) {
  std::array<uint8_t, 4> suit_counts = features.suit_counts;
  uint16_t rank_mask = features.rank_mask;
  auto add_card = [&](CardId card) {
    ++suit_counts[static_cast<size_t>(SuitIndex(SuitFromCardId(card)))];
    rank_mask |= static_cast<uint16_t>(1u << (RankFromCardId(card) - 2));
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

}  // namespace card_abstraction_detail

inline BoardFeatures board_features(const BoardRunout& board) {
  BoardFeatures features;
  features.card_count = board.count();
  for (CardId card : board.cards()) {
    const int rank = RankFromCardId(card);
    const size_t rank_index = static_cast<size_t>(rank - 2);
    const size_t suit_index =
        static_cast<size_t>(SuitIndex(SuitFromCardId(card)));
    ++features.rank_counts[rank_index];
    ++features.suit_counts[suit_index];
    features.max_rank_count =
        std::max(features.max_rank_count, features.rank_counts[rank_index]);
    features.max_suit_count =
        std::max(features.max_suit_count, features.suit_counts[suit_index]);
    features.max_rank = std::max(features.max_rank, static_cast<uint8_t>(rank));
    features.rank_mask |= static_cast<uint16_t>(1u << rank_index);
  }
  return features;
}

inline uint8_t straight_density(uint16_t rank_mask) {
  return card_abstraction_detail::kStraightDensity[rank_mask & 0x1FFF];
}

inline BoardBucketId exact_board_bucket(const BoardRunout& board) {
  return board.mask();
}

inline BoardBucketId board_texture_bucket(
    StreetKind street,
    const BoardFeatures& features) {
  if (features.card_count == 0) {
    return 0;
  }

  const int paired_bucket = features.max_rank_count >= 3
                                ? 2
                                : (features.max_rank_count == 2 ? 1 : 0);
  int suit_bucket = 0;
  if (features.max_suit_count >= 4) {
    suit_bucket = 3;
  } else if (features.max_suit_count >= 3) {
    suit_bucket = 2;
  } else if (features.max_suit_count == 2) {
    suit_bucket = 1;
  }

  const int density = straight_density(features.rank_mask);
  const int straight_bucket = density >= 4 ? 2 : (density >= 3 ? 1 : 0);
  const int high_bucket = features.max_rank >= 14
                              ? 0
                              : (features.max_rank >= 11 ? 1 : 2);
  const int texture =
      (((paired_bucket * card_abstraction_detail::kSuitBuckets) +
        suit_bucket) *
           card_abstraction_detail::kStraightBuckets +
       straight_bucket) *
          card_abstraction_detail::kHighBuckets +
      high_bucket;
  return 1 + static_cast<BoardBucketId>(street) *
                 kCoarsePublicStreetObservationCount +
         static_cast<BoardBucketId>(texture);
}

inline ExactChanceObservation exact_chance_observation(
    StreetKind street,
    const BoardRunout& board) {
  constexpr std::array<uint8_t, 4> kOffsets = {0, 0, 3, 4};
  constexpr std::array<uint8_t, 4> kCounts = {0, 3, 1, 1};
  ExactChanceObservation observation;
  const size_t street_index = static_cast<size_t>(street);
  if (street_index >= kCounts.size()) {
    return observation;
  }
  const size_t offset = kOffsets[street_index];
  const size_t count = kCounts[street_index];
  const auto cards = board.cards();
  if (cards.size() < offset + count) {
    return observation;
  }
  std::copy_n(cards.begin() + offset, count, observation.cards.begin());
  observation.count = static_cast<uint8_t>(count);
  return observation;
}

inline PublicStreetObservation observe_public_street(
    StreetKind street,
    const BoardRunout& board,
    const BoardFeatures& features) {
  if constexpr (kCoarsePublicBuckets) {
    return {board_texture_bucket(street, features), {}};
  } else {
    return {exact_board_bucket(board),
            exact_chance_observation(street, board)};
  }
}

inline PublicStreetObservation observe_public_street(
    StreetKind street,
    const BoardRunout& board) {
  return observe_public_street(street, board, board_features(board));
}

inline BoardBucketId board_bucket(StreetKind street,
                                    const BoardRunout& board,
                                    const BoardFeatures& features) {
  return observe_public_street(street, board, features).value;
}

inline BoardBucketId board_bucket(StreetKind street, const BoardRunout& board) {
  return board_bucket(street, board, board_features(board));
}

inline PrivateBucketId exact_private_bucket(ComboId combo_id) {
  return combo_id;
}

inline PrivateBucketId coarse_private_bucket(
    ComboId combo_id,
    StreetKind street,
    const BoardFeatures& features) {
  const ComboInfo& combo = GetComboInfo(combo_id);
  const int rank0 = RankFromCardId(combo.card0);
  const int rank1 = RankFromCardId(combo.card1);
  const int high = rank0 > rank1 ? rank0 : rank1;
  const int low = rank0 > rank1 ? rank1 : rank0;
  const bool pair = rank0 == rank1;
  const bool suited = SuitFromCardId(combo.card0) ==
                      SuitFromCardId(combo.card1);

  if (street == StreetKind::kPreflop || features.card_count == 0) {
    const int shape = pair ? 0 : (suited ? 1 : 2);
    return static_cast<PrivateBucketId>(
        shape * 12 + card_abstraction_detail::HighRankGroup(high) * 3 +
        card_abstraction_detail::LowRankGroup(low));
  }

  const int local_bucket =
      card_abstraction_detail::MadeBucket(combo, features) * 9 +
      card_abstraction_detail::DrawBucket(combo, features) * 3 +
      card_abstraction_detail::HoleStrengthBucket(high, low, pair, suited);
  return static_cast<PrivateBucketId>(local_bucket);
}

inline PrivateStreetObservation observe_private_street(
    ComboId combo_id,
    StreetKind street,
    const BoardFeatures& features) {
  if constexpr (kCoarsePrivateBuckets) {
    return {coarse_private_bucket(combo_id, street, features)};
  } else {
    return {exact_private_bucket(combo_id)};
  }
}

inline PrivateStreetObservation observe_private_street(
    ComboId combo_id,
    StreetKind street,
    const BoardRunout& board) {
  return observe_private_street(combo_id, street, board_features(board));
}

inline PrivateBucketId private_bucket(ComboId combo_id,
                                      StreetKind street,
                                      const BoardFeatures& features) {
  return observe_private_street(combo_id, street, features).value;
}

inline uint32_t private_bucket_count(StreetKind) {
  return kCoarsePrivateBuckets ? kCoarsePrivateStreetObservationCount
                               : kComboCount;
}

}  // namespace poker
