#pragma once

#include "src/poker.h"

namespace poker {

struct CanonicalCardObservation {
  PublicObservationId public_observation;
  PrivateObservationId private_observation;

  friend bool operator==(const CanonicalCardObservation&,
                         const CanonicalCardObservation&) = default;
};

struct CanonicalCardState {
  ComboId hand;
  Board board;
  CanonicalCardObservation observation;
};

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept;
CanonicalCardState CanonicalizeCardState(ComboId hand,
                                         const Board& board) noexcept;
CanonicalCardObservation CanonicalizeObservation(ComboId hand,
                                                 const Board& board) noexcept;

}  // namespace poker
