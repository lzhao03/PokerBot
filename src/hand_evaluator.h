#pragma once

#include "src/poker.h"

namespace poker {

int CompareHands(ComboId first,
                 ComboId second,
                 const Board& board);

}  // namespace poker
