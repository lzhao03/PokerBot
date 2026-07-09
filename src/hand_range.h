#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/combo.h"
#include "src/training_range.h"

namespace poker {

class HandRange {
 public:
  HandRange();

  void add_combo(ComboId combo_id, double weight);
  void add_hand_by_index(int index, double weight);
  double get_probability(ComboId combo_id) const;

  const std::vector<std::pair<int, double>>& hand_type_weights() const {
    return hand_weights_;
  }

  const std::vector<std::pair<ComboId, double>>& exact_combo_weights() const {
    return exact_hand_weights_;
  }

  void clear();
  void set_uniform_range();
  void set_from_string(const std::string& range_str);
  std::string to_string() const;
  double get_total_weight() const;
  void normalize();

  static int combo_to_index(ComboId combo_id);
  static std::optional<ComboId> index_to_combo(int index);
  static std::string combo_to_string(ComboId combo_id);
  static int string_to_index(const std::string& hand_str);

 private:
  std::vector<std::pair<int, double>> hand_weights_;
  std::vector<std::pair<ComboId, double>> exact_hand_weights_;
  double total_weight_;

  void parse_range_component(const std::string& component);
};

TrainingRange BuildTrainingRange(const HandRange& range);

}  // namespace poker
