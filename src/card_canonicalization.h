#pragma once

#include "src/poker.h"

namespace poker {

struct CanonicalCardObservation {
  PublicObservationId public_observation;
  PrivateObservationId private_observation;

  friend bool operator==(const CanonicalCardObservation&,
                         const CanonicalCardObservation&) = default;
};

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept;
CanonicalCardObservation CanonicalizeObservation(ComboId hand,
                                                 const Board& board) noexcept;

}  // namespace poker
