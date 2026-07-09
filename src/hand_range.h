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
  
  // Add an exact two-card combo to the range with a weight.
  void add_combo(ComboId combo_id, double weight);
  
  // Add a hand by index with a weight
  void add_hand_by_index(int index, double weight);
  
  // Get the probability of a specific hand
  double get_probability(ComboId combo_id) const;

  const std::vector<std::pair<int, double>>& hand_type_weights() const {
    return hand_weights_;
  }

  const std::vector<std::pair<ComboId, double>>& exact_combo_weights() const {
    return exact_hand_weights_;
  }
  
  // Clear the range
  void clear();
  
  // Set a uniform range (all hands with equal weight)
  void set_uniform_range();
  
  // Set a range from a string representation (e.g., "AK,QQ+,89s")
  void set_from_string(const std::string& range_str);
  
  // Get a string representation of the range
  std::string to_string() const;
  
  // Get the total weight of all hands in the range
  double get_total_weight() const;
  
  // Normalize the weights to sum to 1
  void normalize();
  
  // Convert a combo to its canonical hand-class index (0-168)
  static int combo_to_index(ComboId combo_id);
  
  // Convert an index to a representative combo, or nullopt for invalid input.
  static std::optional<ComboId> index_to_combo(int index);
  
  // Convert a combo to a string representation (for display only)
  static std::string combo_to_string(ComboId combo_id);
  
  // Convert a string to a hand index (for parsing only)
  static int string_to_index(const std::string& hand_str);

private:
  // Direct mapping from hand index to weight
  // Using a sparse representation for efficiency
  std::vector<std::pair<int, double>> hand_weights_;

  // Exact two-card combos added directly with add_combo().
  std::vector<std::pair<ComboId, double>> exact_hand_weights_;
  
  // Total weight of all hands
  double total_weight_;
  
  // Parse a range string component (e.g., "AK", "QQ+", "89s")
  void parse_range_component(const std::string& component);
};

TrainingRange BuildTrainingRange(const HandRange& range);

} // namespace poker
