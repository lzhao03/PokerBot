#include "src/training_range.h"

#include <algorithm>
#include <stdexcept>

namespace poker {

RangeSampler::RangeSampler(const TrainingRange& player_a_range,
                           const TrainingRange& player_b_range)
    : player_a_range(player_a_range),
      player_b_range(player_b_range),
      compatible_player_b_weight(kComboCount, 0.0f),
      compatible_player_b_offsets(kComboCount, 0),
      compatible_player_b_counts(kComboCount, 0) {
  float total_weight = 0.0f;
  player_a_sample_weights.reserve(player_a_range.active_count);
  const size_t max_compatible_pairs =
      static_cast<size_t>(player_a_range.active_count) *
      static_cast<size_t>(player_b_range.active_count);
  compatible_player_b_combos.reserve(max_compatible_pairs);
  compatible_player_b_cumulative_weights.reserve(max_compatible_pairs);
  for (uint16_t a = 0; a < player_a_range.active_count; ++a) {
    const ComboId player_a_combo = player_a_range.active[a];
    const float player_a_weight = player_a_range.weights[player_a_combo];
    if (player_a_weight <= 0.0f) {
      player_a_sample_weights.push_back(0.0f);
      player_a_cumulative_weights.push_back(total_weight);
      continue;
    }
    const size_t offset = compatible_player_b_combos.size();
    compatible_player_b_offsets[player_a_combo] =
        static_cast<uint32_t>(offset);
    float cumulative_player_b_weight = 0.0f;
    for (uint16_t b = 0; b < player_b_range.active_count; ++b) {
      const ComboId player_b_combo = player_b_range.active[b];
      const float player_b_weight = player_b_range.weights[player_b_combo];
      if (player_b_weight <= 0.0f ||
          (ComboMask(player_a_combo) & ComboMask(player_b_combo)) != 0) {
        continue;
      }
      cumulative_player_b_weight += player_b_weight;
      compatible_player_b_combos.push_back(player_b_combo);
      compatible_player_b_cumulative_weights.push_back(
          cumulative_player_b_weight);
    }
    compatible_player_b_weight[player_a_combo] = cumulative_player_b_weight;
    compatible_player_b_counts[player_a_combo] =
        static_cast<uint16_t>(compatible_player_b_combos.size() - offset);
    const float sample_weight =
        player_a_weight * compatible_player_b_weight[player_a_combo];
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
  std::uniform_real_distribution<float> player_a_distribution(
      0.0f, total_player_a_weight);
  const float player_a_sample = player_a_distribution(rng);
  auto player_a_sampled = std::upper_bound(
      player_a_cumulative_weights.begin(), player_a_cumulative_weights.end(),
      player_a_sample);
  if (player_a_sampled == player_a_cumulative_weights.end()) {
    player_a_sampled = player_a_cumulative_weights.end() - 1;
  }
  const size_t player_a_active_index = static_cast<size_t>(
      player_a_sampled - player_a_cumulative_weights.begin());
  const ComboId player_a_combo = player_a_range.active[player_a_active_index];
  const float total_player_b_weight =
      compatible_player_b_weight[player_a_combo];
  const uint16_t compatible_count =
      compatible_player_b_counts[player_a_combo];
  if (total_player_b_weight <= 0.0f || compatible_count == 0) {
    throw std::logic_error("Range sampler selected an incompatible hand");
  }

  std::uniform_real_distribution<float> distribution(
      0.0f, total_player_b_weight);
  const float sample = distribution(rng);
  const size_t offset = compatible_player_b_offsets[player_a_combo];
  const auto first = compatible_player_b_cumulative_weights.begin() + offset;
  const auto last = first + compatible_count;
  auto sampled = std::upper_bound(first, last, sample);
  if (sampled == last) {
    sampled = last - 1;
  }

  const size_t sampled_index = offset + static_cast<size_t>(sampled - first);
  return RangeDeal(player_a_combo, compatible_player_b_combos[sampled_index]);
}

TrainingRange BuildTrainingRange(const HandRange& range) {
  return BuildTrainingRange(range.get_all_weighted_combos());
}

TrainingRange BuildTrainingRange(const WeightedHandRange& combos) {
  TrainingRange training_range;

  for (size_t i = 0; i < combos.size(); ++i) {
    if (combos.weights[i] <= 0.0) {
      continue;
    }

    const ComboId combo_id = combos.combos[i];
    if (training_range.weights[combo_id] == 0.0f) {
      training_range.active[training_range.active_count++] = combo_id;
    }
    training_range.weights[combo_id] += static_cast<float>(combos.weights[i]);
  }

  return training_range;
}

}  // namespace poker
