#pragma once

#include <array>
#include <string_view>

#include "src/combo.h"

namespace poker {

struct ComboRange {
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  void add(ComboId combo, float weight = 1.0f) {
    if (combo < kComboCount && weight > 0.0f) {
      if (weights[combo] == 0.0f) {
        active[active_count++] = combo;
      }
      weights[combo] += weight;
    }
  }

  bool empty() const;
  size_t count() const { return active_count; }
  float weight(ComboId combo) const { return weights[combo]; }
};

ComboRange ParseRange(std::string_view text);
ComboRange UniformRange();
ComboRange SingleComboRange(ComboId combo, float weight = 1.0f);

}  // namespace poker
