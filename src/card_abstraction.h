#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "src/card_canonicalization.h"
#include "src/poker.h"

namespace poker {

using PrivateBucketId = uint16_t;

inline constexpr uint32_t kCoarsePrivateStreetObservationCount = 36;
inline constexpr uint32_t kCoarsePublicStreetObservationCount = 108;

enum class PublicCardMode : uint8_t {
  kExactCanonical,
  kTexture,
};

enum class PrivateAbstractionKind : uint8_t {
  kExactCanonical,
  kHandcrafted36,
};

enum class RecallMode : uint8_t {
  kCurrentBucketOnly,
  kBucketHistory,
};

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::kTexture;
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::kHandcrafted36;
  RecallMode recall_mode = RecallMode::kBucketHistory;
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

class PublicPosition {
 public:
  static PublicPosition Root(const CardAbstractionConfig& config,
                             StreetKind street,
                             Board board);

  PublicPosition after_chance(const CardAbstractionConfig& config,
                              StreetKind street,
                              Board board) const;

  StreetKind street() const noexcept { return street_; }
  const Board& board() const noexcept { return board_; }
  PublicObservationId observation() const noexcept { return observation_; }
  const BoardFeatures& features() const noexcept { return features_; }

 private:
  PublicPosition(StreetKind street,
                 Board board,
                 PublicObservationId observation,
                 BoardFeatures features)
      : street_(street),
        board_(std::move(board)),
        observation_(observation),
        features_(features) {}

  StreetKind street_ = StreetKind::kPreflop;
  Board board_ = PreflopBoard{};
  PublicObservationId observation_;
  BoardFeatures features_;
};

BoardFeatures BoardFeaturesFor(const Board& board) noexcept;
BoardBucketId BoardTextureBucket(StreetKind street,
                                 const BoardFeatures& features) noexcept;
PrivateBucketId CoarsePrivateBucket(ComboId hand,
                                    StreetKind street,
                                    const BoardFeatures& features) noexcept;

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  StreetKind street,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(const CardAbstractionConfig& config,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept;

PrivateObservationId InitialPrivateObservation(
    const CardAbstractionConfig& config,
    ComboId hand) noexcept;
PrivateObservationId AdvancePrivateObservation(
    const CardAbstractionConfig& config,
    PrivateObservationId previous,
    ComboId hand,
    const PublicPosition& child) noexcept;

}  // namespace poker
