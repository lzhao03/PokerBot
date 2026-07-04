#pragma once

#include <array>
#include <cstddef>
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

struct TrainingRangeView {
  const TrainingRange* source = nullptr;
  bool all_source_combos = false;
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  TrainingRangeView() = default;

  explicit TrainingRangeView(const TrainingRange& source_range) {
    reset_to_all(source_range);
  }

  void reset_to_all(const TrainingRange& source_range) {
    clear_active_weights();
    source = &source_range;
    all_source_combos = true;
  }

  void reset_to_filtered() {
    clear_active_weights();
    source = nullptr;
    all_source_combos = false;
  }

  void clear() {
    clear_active_weights();
    source = nullptr;
    all_source_combos = false;
  }

  void add(ComboId combo_id, float weight) {
    all_source_combos = false;
    if (weights[combo_id] == 0.0f) {
      active[active_count++] = combo_id;
    }
    weights[combo_id] += weight;
  }

  bool empty() const { return size() == 0; }

  size_t size() const {
    return all_source_combos && source != nullptr
               ? source->active_count
               : active_count;
  }

  ComboId combo(size_t index) const {
    return all_source_combos ? source->active[index] : active[index];
  }

  Hand hand(size_t index) const { return ComboIdToHand(combo(index)); }

  float weight(size_t index) const {
    const ComboId combo_id = combo(index);
    return all_source_combos ? source->weights[combo_id] : weights[combo_id];
  }

  CardMask mask(size_t index) const { return ComboMask(combo(index)); }

 private:
  void clear_active_weights() {
    for (uint16_t i = 0; i < active_count; ++i) {
      weights[active[i]] = 0.0f;
    }
    active_count = 0;
  }
};

TrainingRange BuildTrainingRange(const HandRange& range);
TrainingRange BuildTrainingRange(const WeightedHandRange& combos);

}  // namespace poker
