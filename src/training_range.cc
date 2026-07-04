#include "src/training_range.h"

namespace poker {

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
