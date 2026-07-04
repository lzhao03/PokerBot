#pragma once

#include <array>
#include <cstdint>

#include "src/combo.h"
#include "src/hand_range.h"

namespace poker {

struct TrainingRange {
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  bool empty() const { return active_count == 0; }
  float weight(ComboId combo_id) const { return weights[combo_id]; }
};

TrainingRange BuildTrainingRange(const HandRange& range);

}  // namespace poker
