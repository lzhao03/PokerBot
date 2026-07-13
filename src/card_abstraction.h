#pragma once

#include <cstdint>
#include "src/poker.h"

namespace poker {

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

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::Texture;
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode = RecallMode::BucketHistory;
};

class PublicPosition {
 public:
  PublicPosition(const CardAbstractionConfig& config, const Board& board);

  const Board& board() const noexcept { return board_; }
  PublicObservationId observation() const noexcept { return observation_; }

 private:
  Board board_;
  PublicObservationId observation_;
};

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(const CardAbstractionConfig& config,
                                    ComboId hand,
                                    const Board& board,
                                    PrivateObservationId previous = {}) noexcept;

}  // namespace poker
