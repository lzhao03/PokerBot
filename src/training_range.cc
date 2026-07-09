#include "src/training_range.h"

#include <algorithm>
#include <stdexcept>

namespace poker {

RangeSampler::RangeSampler(const TrainingRange& a_range,
                           const TrainingRange& b_range)
    : player_a_range(a_range),
      player_b_range(b_range),
      compatible_player_b_weight(kComboCount, 0.0f),
      compatible_player_b_offsets(kComboCount, 0),
      compatible_player_b_counts(kComboCount, 0) {
  float total_weight = 0.0f;
  const size_t a_count = a_range.active_count;
  const size_t b_count = b_range.active_count;
  const size_t max_pairs = a_count * b_count;

  player_a_sample_weights.reserve(a_count);
  compatible_player_b_combos.reserve(max_pairs);
  compatible_player_b_cumulative_weights.reserve(max_pairs);

  for (uint16_t a = 0; a < a_range.active_count; ++a) {
    const ComboId a_combo = a_range.active[a];
    const float a_weight = a_range.weights[a_combo];
    if (a_weight <= 0.0f) {
      player_a_sample_weights.push_back(0.0f);
      player_a_cumulative_weights.push_back(total_weight);
      continue;
    }

    const size_t offset = compatible_player_b_combos.size();
    compatible_player_b_offsets[a_combo] = static_cast<uint32_t>(offset);

    float b_cumulative = 0.0f;
    for (uint16_t b = 0; b < b_range.active_count; ++b) {
      const ComboId b_combo = b_range.active[b];
      const float b_weight = b_range.weights[b_combo];
      if (b_weight <= 0.0f || (ComboMask(a_combo) & ComboMask(b_combo)) != 0) {
        continue;
      }
      b_cumulative += b_weight;
      compatible_player_b_combos.push_back(b_combo);
      compatible_player_b_cumulative_weights.push_back(b_cumulative);
    }

    compatible_player_b_weight[a_combo] = b_cumulative;
    compatible_player_b_counts[a_combo] =
        static_cast<uint16_t>(compatible_player_b_combos.size() - offset);
    const float sample_weight = a_weight * b_cumulative;
    player_a_sample_weights.push_back(sample_weight);
    total_weight += sample_weight;
    player_a_cumulative_weights.push_back(total_weight);
  }

  if (total_weight <= 0.0f) {
    throw std::invalid_argument(
        "Could not sample non-overlapping hands from ranges");
  }
  total_player_a_weight = total_weight;
}

RangeDeal RangeSampler::sample(std::mt19937& rng) const {
  std::uniform_real_distribution<float> a_dist(0.0f, total_player_a_weight);
  const float a_sample = a_dist(rng);
  const auto a_begin = player_a_cumulative_weights.begin();
  const auto a_end = player_a_cumulative_weights.end();
  auto a_it = std::upper_bound(a_begin, a_end, a_sample);
  if (a_it == a_end) {
    a_it = a_end - 1;
  }

  const size_t a_index = static_cast<size_t>(a_it - a_begin);
  const ComboId a_combo = player_a_range.active[a_index];
  const float b_total = compatible_player_b_weight[a_combo];
  const uint16_t b_count = compatible_player_b_counts[a_combo];
  if (b_total <= 0.0f || b_count == 0) {
    throw std::logic_error("Range sampler selected an incompatible hand");
  }

  std::uniform_real_distribution<float> b_dist(0.0f, b_total);
  const float b_sample = b_dist(rng);
  const size_t offset = compatible_player_b_offsets[a_combo];
  const auto b_begin = compatible_player_b_cumulative_weights.begin() + offset;
  const auto b_end = b_begin + b_count;
  auto b_it = std::upper_bound(b_begin, b_end, b_sample);
  if (b_it == b_end) {
    b_it = b_end - 1;
  }

  const size_t b_index = offset + static_cast<size_t>(b_it - b_begin);
  return RangeDeal{a_combo, compatible_player_b_combos[b_index]};
}

const TrainingRangeView& TrainingRangeView::copy_without_mask_into(
    CardMask blocked_mask,
    TrainingRangeView& scratch) const {
  scratch.reset_to_filtered();
  for (size_t i = 0; i < size(); ++i) {
    if (weight(i) > 0.0f && (mask(i) & blocked_mask) == 0) {
      scratch.add(combo(i), weight(i));
    }
  }
  return scratch;
}

}  // namespace poker
