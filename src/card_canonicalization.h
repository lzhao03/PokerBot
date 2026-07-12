#pragma once

#include "src/poker.h"

namespace poker {

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept;
PrivateObservationId CanonicalPrivateObservation(
    ComboId hand,
    const Board& board) noexcept;

}  // namespace poker
