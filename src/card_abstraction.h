#pragma once

#include <array>
#include <cstdint>

#include "src/poker.h"

namespace poker {

// Bump when a public or private observation ID changes meaning.
inline constexpr uint8_t kCardAbstractionSchemaVersion = 0;
inline constexpr size_t kPublicObservationBitsPerStreet = 7;
inline constexpr std::array<uint64_t, 3> kCompactPublicBuckets = {16, 16, 64};
inline constexpr size_t kPrivateBucketCount = 36;
inline constexpr uint32_t kPrivateObservationRadix = kPrivateBucketCount + 1;
inline constexpr std::array<uint32_t, 4> kPrivateObservationPlaces = {
    1,
    kPrivateObservationRadix,
    kPrivateObservationRadix * kPrivateObservationRadix,
    kPrivateObservationRadix * kPrivateObservationRadix *
        kPrivateObservationRadix};

enum class PublicCardMode : uint8_t {
  ExactCanonical,
  Texture,
  CompactTexture,
};

enum class PrivateAbstractionKind : uint8_t {
  ExactCanonical,
  Handcrafted36,
};

enum class RecallMode : uint8_t {
  CurrentBucketOnly,
  BucketHistory,
};

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::Texture;
  PrivateAbstractionKind private_kind = PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode = RecallMode::BucketHistory;
};

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept;
PrivateObservationId CanonicalPrivateObservation(
    ComboId hand,
    const Board& board) noexcept;

class PublicPosition {
 public:
  PublicPosition(const CardAbstractionConfig& config, const Board& board);

  const Board& board() const noexcept { return board_; }
  PublicObservationId observation() const noexcept { return observation_; }

 private:
  Board board_;
  PublicObservationId observation_;
  std::array<uint8_t, 13> rank_counts_ = {};
  std::array<uint8_t, 4> suit_counts_ = {};
  uint16_t rank_mask_ = 0;
  uint8_t max_rank_count_ = 0;
  uint8_t max_suit_count_ = 0;
  PrivateAbstractionKind private_kind_ = PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode_ = RecallMode::BucketHistory;

  friend PrivateObservationId ObservePrivate(
      ComboId hand,
      const PublicPosition& position,
      PrivateObservationId previous) noexcept;
};

PrivateObservationId ObservePrivate(ComboId hand,
                                    const PublicPosition& position,
                                    PrivateObservationId previous = {}) noexcept;

}  // namespace poker
