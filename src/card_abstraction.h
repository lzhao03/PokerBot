#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/build_flags.h"
#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

using PublicBucketId = uint64_t;
using PrivateBucketId = uint16_t;

namespace card_abstraction_detail {

inline int BestStraightWindowDensity(uint16_t rank_mask) {
  int best = 0;
  for (int start = 0; start <= 8; ++start) {
    const int count =
        std::popcount(static_cast<unsigned int>((rank_mask >> start) & 0x1F));
    best = std::max(best, count);
  }
  const uint16_t wheel_mask = static_cast<uint16_t>((1u << 12) | 0x0F);
  const int wheel_count =
      std::popcount(static_cast<unsigned int>(rank_mask & wheel_mask));
  return std::max(best, wheel_count);
}

}  // namespace card_abstraction_detail

struct ExactPublicCardBuckets {
  PublicBucketId bucket(const CompactPublicState& state) const {
    return state.board_mask;
  }
};

struct BoardTexturePublicCardBuckets {
  PublicBucketId bucket(const CompactPublicState& state) const {
    const int board_count = state.board_count;
    if (board_count == 0) {
      return 0;
    }

    int rank_counts[13] = {};
    int suit_counts[4] = {};
    uint16_t rank_mask = 0;
    int max_rank_count = 0;
    int max_suit_count = 0;
    int max_rank = 0;
    for (int i = 0; i < board_count; ++i) {
      const CardId card = state.board_cards[static_cast<size_t>(i)];
      const int rank = RankFromCardId(card);
      const int rank_index = rank - 2;
      const int suit_index = SuitIndex(SuitFromCardId(card));
      ++rank_counts[rank_index];
      ++suit_counts[suit_index];
      max_rank_count = std::max(max_rank_count, rank_counts[rank_index]);
      max_suit_count = std::max(max_suit_count, suit_counts[suit_index]);
      max_rank = std::max(max_rank, rank);
      rank_mask |= static_cast<uint16_t>(1u << rank_index);
    }

    const int paired_bucket =
        max_rank_count >= 3 ? 2 : (max_rank_count == 2 ? 1 : 0);
    int suit_bucket = 0;
    if (max_suit_count >= 4) {
      suit_bucket = 3;
    } else if (max_suit_count >= 3) {
      suit_bucket = 2;
    } else if (max_suit_count == 2) {
      suit_bucket = 1;
    }
    const int straight_density =
        card_abstraction_detail::BestStraightWindowDensity(rank_mask);
    const int straight_bucket =
        straight_density >= 4 ? 2 : (straight_density >= 3 ? 1 : 0);
    const int high_bucket = max_rank >= 14 ? 0 : (max_rank >= 11 ? 1 : 2);
    const int texture =
        (((paired_bucket * kSuitBuckets) + suit_bucket) * kStraightBuckets +
         straight_bucket) *
            kHighBuckets +
        high_bucket;
    return 1 + static_cast<PublicBucketId>(state.street) *
                   kTextureBucketsPerStreet +
           static_cast<PublicBucketId>(texture);
  }

 private:
  static constexpr int kSuitBuckets = 4;
  static constexpr int kStraightBuckets = 3;
  static constexpr int kHighBuckets = 3;
  static constexpr int kTextureBucketsPerStreet =
      3 * kSuitBuckets * kStraightBuckets * kHighBuckets;
};

using DefaultPublicCardBuckets =
    std::conditional_t<kCoarsePublicBuckets,
                       BoardTexturePublicCardBuckets,
                       ExactPublicCardBuckets>;

struct ExactPrivateBuckets {
  PrivateBucketId bucket(ComboId combo_id, const CompactPublicState&) const {
    return combo_id;
  }

  uint32_t bucket_count(const CompactPublicState&) const {
    return kComboCount;
  }
};

struct CoarsePrivateBuckets {
  static constexpr uint32_t kPreflopBucketCount = 36;
  static constexpr uint32_t kPostflopBucketsPerStreet = 36;
  static constexpr uint32_t kBucketCount =
      kPreflopBucketCount + 3 * kPostflopBucketsPerStreet;

  PrivateBucketId bucket(ComboId combo_id,
                         const CompactPublicState& state) const {
    const ComboInfo& combo = GetComboInfo(combo_id);
    const int rank0 = RankFromCardId(combo.card0);
    const int rank1 = RankFromCardId(combo.card1);
    const int high = rank0 > rank1 ? rank0 : rank1;
    const int low = rank0 > rank1 ? rank1 : rank0;
    const bool pair = rank0 == rank1;
    const bool suited = SuitFromCardId(combo.card0) ==
                        SuitFromCardId(combo.card1);

    if (state.street == StreetKind::kPreflop || state.board_count == 0) {
      const int shape = pair ? 0 : (suited ? 1 : 2);
      return static_cast<PrivateBucketId>(
          shape * 12 + HighRankGroup(high) * 3 + LowRankGroup(low));
    }

    const int street = static_cast<int>(state.street) - 1;
    const int local_bucket =
        MadeBucket(combo, state) * 9 + DrawBucket(combo, state) * 3 +
        HoleStrengthBucket(high, low, pair, suited);
    return static_cast<PrivateBucketId>(
        kPreflopBucketCount + street * kPostflopBucketsPerStreet +
        local_bucket);
  }

  uint32_t bucket_count(const CompactPublicState&) const {
    return kBucketCount;
  }

 private:
  static int HighRankGroup(int rank) {
    if (rank >= 14) {
      return 0;
    }
    if (rank >= 12) {
      return 1;
    }
    return rank >= 9 ? 2 : 3;
  }

  static int LowRankGroup(int rank) {
    if (rank >= 10) {
      return 0;
    }
    return rank >= 7 ? 1 : 2;
  }

  static int HoleStrengthBucket(int high, int low, bool pair, bool suited) {
    if (pair || high == 14 || (high >= 13 && low >= 10)) {
      return 0;
    }
    const int gap = high - low;
    if ((high >= 11 && low >= 8) || (suited && gap <= 2)) {
      return 1;
    }
    return 2;
  }

  static int MadeBucket(const ComboInfo& combo,
                        const CompactPublicState& state) {
    int rank_counts[15] = {};
    ++rank_counts[RankFromCardId(combo.card0)];
    ++rank_counts[RankFromCardId(combo.card1)];
    for (int i = 0; i < state.board_count; ++i) {
      ++rank_counts[
          RankFromCardId(state.board_cards[static_cast<size_t>(i)])];
    }

    int pairs = 0;
    int max_count = 0;
    for (int rank = 2; rank <= 14; ++rank) {
      if (rank_counts[rank] >= 2) {
        ++pairs;
      }
      if (rank_counts[rank] > max_count) {
        max_count = rank_counts[rank];
      }
    }
    if (max_count >= 3) {
      return 3;
    }
    if (pairs >= 2) {
      return 2;
    }
    return pairs == 1 ? 1 : 0;
  }

  static int DrawBucket(const ComboInfo& combo,
                        const CompactPublicState& state) {
    int suit_counts[4] = {};
    uint16_t rank_mask = 0;
    auto add_card = [&](CardId card) {
      ++suit_counts[SuitIndex(SuitFromCardId(card))];
      rank_mask |=
          static_cast<uint16_t>(1u << (RankFromCardId(card) - 2));
    };
    add_card(combo.card0);
    add_card(combo.card1);
    for (int i = 0; i < state.board_count; ++i) {
      add_card(state.board_cards[static_cast<size_t>(i)]);
    }

    for (int count : suit_counts) {
      if (count >= 4) {
        return 2;
      }
    }
    return card_abstraction_detail::BestStraightWindowDensity(rank_mask) >= 4
               ? 1
               : 0;
  }
};

using DefaultPrivateBuckets =
    std::conditional_t<kCoarsePublicBuckets,
                       CoarsePrivateBuckets,
                       ExactPrivateBuckets>;

struct CardAbstraction {
  DefaultPublicCardBuckets public_buckets;
  DefaultPrivateBuckets private_buckets;

  PublicBucketId public_bucket(const CompactPublicState& state) const {
    return public_buckets.bucket(state);
  }

  PrivateBucketId private_bucket(ComboId combo_id,
                                 const CompactPublicState& state) const {
    return private_buckets.bucket(combo_id, state);
  }

  uint32_t private_bucket_count(const CompactPublicState& state) const {
    return private_buckets.bucket_count(state);
  }
};

}  // namespace poker
