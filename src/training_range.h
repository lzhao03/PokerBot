#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "src/combo.h"

namespace poker {

struct TrainingRange {
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  void add(ComboId combo_id, float weight) {
    if (combo_id >= kComboCount || weight <= 0.0f) {
      return;
    }
    if (weights[combo_id] == 0.0f) {
      active[active_count++] = combo_id;
    }
    weights[combo_id] += weight;
  }

  bool empty() const { return active_count == 0; }
  float weight(ComboId combo_id) const { return weights[combo_id]; }
};

struct RangeDeal {
  RangeDeal(ComboId player_a_combo, ComboId player_b_combo)
      : player_a_combo(player_a_combo),
        player_b_combo(player_b_combo) {}

  ComboId player_a_combo = 0;
  ComboId player_b_combo = 0;
};

struct RangeSampler {
  RangeSampler(const TrainingRange& player_a_range,
               const TrainingRange& player_b_range);

  RangeDeal sample(std::mt19937& rng) const;

  const TrainingRange& player_a_range;
  const TrainingRange& player_b_range;
  std::vector<float> compatible_player_b_weight;
  std::vector<float> player_a_sample_weights;
  std::vector<float> player_a_cumulative_weights;
  float total_player_a_weight = 0.0f;
  std::vector<uint32_t> compatible_player_b_offsets;
  std::vector<uint16_t> compatible_player_b_counts;
  std::vector<ComboId> compatible_player_b_combos;
  std::vector<float> compatible_player_b_cumulative_weights;
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

  const TrainingRangeView& without_mask(CardMask blocked_mask,
                                        TrainingRangeView& scratch) const;

  bool empty() const { return size() == 0; }

  size_t size() const {
    return all_source_combos && source != nullptr
               ? source->active_count
               : active_count;
  }

  ComboId combo(size_t index) const {
    return all_source_combos ? source->active[index] : active[index];
  }

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

}  // namespace poker
