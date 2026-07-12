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
  ExactCanonical,
  Texture,
};

enum class PrivateAbstractionKind : uint8_t {
  ExactCanonical,
  Handcrafted36,
};

enum class RecallMode : uint8_t {
  CurrentBucketOnly,
  BucketHistory,
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

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::Texture;
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode = RecallMode::BucketHistory;
};

class CardAbstraction;

class PublicPosition {
 public:
  static PublicPosition Root(const CardAbstraction& abstraction,
                             StreetKind street,
                             Board board);

  PublicPosition after_chance(const CardAbstraction& abstraction,
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

  StreetKind street_ = StreetKind::Preflop;
  Board board_ = PreflopBoard{};
  PublicObservationId observation_;
  BoardFeatures features_;
};

class CardAbstraction {
 public:
  explicit CardAbstraction(CardAbstractionConfig config)
      : config_(std::move(config)) {}

  const CardAbstractionConfig& config() const noexcept { return config_; }

 private:
  CardAbstractionConfig config_;

  friend PublicObservationId ObservePublic(
      const CardAbstraction&, const Board&) noexcept;
  friend PrivateObservationId ObservePrivate(
      const CardAbstraction&, ComboId, const PublicPosition&) noexcept;
};

BoardFeatures BoardFeaturesFor(const Board& board) noexcept;
BoardBucketId BoardTextureBucket(StreetKind street,
                                 const BoardFeatures& features) noexcept;
PrivateBucketId Handcrafted36Bucket(ComboId hand,
                                    StreetKind street,
                                    const BoardFeatures& features) noexcept;

PublicObservationId ObservePublic(const CardAbstraction& abstraction,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(const CardAbstraction& abstraction,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept;

}  // namespace poker
