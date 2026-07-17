#pragma once

#include <array>
#include <cstdint>

#include "src/poker.h"

namespace poker {

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
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode = RecallMode::BucketHistory;
};

struct BoardFeatures {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint8_t, 4> suit_counts = {};
  uint16_t rank_mask = 0;
  uint8_t max_rank_count = 0;
  uint8_t max_suit_count = 0;
};

class PublicPosition {
 public:
  PublicPosition(const CardAbstractionConfig& config, const Board& board);

  const Board& board() const noexcept { return board_; }
  PublicObservationId observation() const noexcept { return observation_; }

 private:
  Board board_;
  PublicObservationId observation_;
  BoardFeatures features_;
  PrivateAbstractionKind private_kind_ =
      PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode_ = RecallMode::BucketHistory;

  friend PrivateObservationId ObservePrivate(
      ComboId hand,
      const PublicPosition& position,
      PrivateObservationId previous) noexcept;
};

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(ComboId hand,
                                    const PublicPosition& position,
                                    PrivateObservationId previous = {}) noexcept;

}  // namespace poker
