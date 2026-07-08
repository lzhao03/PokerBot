#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "src/combo.h"

namespace poker {

struct WeightedHandRange {
  std::vector<ComboId> combos;
  std::vector<double> weights;
  std::vector<CardMask> masks;

  size_t size() const { return combos.size(); }
  bool empty() const { return combos.empty(); }

  void reserve(size_t count) {
    combos.reserve(count);
    weights.reserve(count);
    masks.reserve(count);
  }

  void clear() {
    combos.clear();
    weights.clear();
    masks.clear();
  }

  void add(ComboId combo_id, double weight) {
    combos.push_back(combo_id);
    weights.push_back(weight);
    masks.push_back(ComboMask(combo_id));
  }
};

struct WeightedHandRangeView {
  std::optional<std::reference_wrapper<const WeightedHandRange>> source;
  bool all_source_hands = false;
  std::vector<size_t> indices;
  std::vector<double> weights;

  WeightedHandRangeView() = default;

  explicit WeightedHandRangeView(const WeightedHandRange& source_range) {
    reset_to_all(source_range);
  }

  void reset_to_all(const WeightedHandRange& source_range) {
    source = std::cref(source_range);
    all_source_hands = true;
    indices.clear();
    weights.clear();
  }

  void reset_to_filtered(const WeightedHandRange& source_range) {
    source = std::cref(source_range);
    all_source_hands = false;
    indices.clear();
    weights.clear();
  }

  void clear() {
    source = std::nullopt;
    all_source_hands = false;
    indices.clear();
    weights.clear();
  }

  void reserve(size_t count) {
    indices.reserve(count);
    weights.reserve(count);
  }

  void add(size_t source_index, double weight) {
    all_source_hands = false;
    indices.push_back(source_index);
    weights.push_back(weight);
  }

  bool has_source() const { return source.has_value(); }
  bool empty() const { return size() == 0; }

  size_t size() const {
    if (!source.has_value()) {
      return 0;
    }
    return all_source_hands ? source->get().size() : indices.size();
  }

  const WeightedHandRange& source_range() const {
    return source->get();
  }

  size_t source_index(size_t index) const {
    return all_source_hands ? index : indices[index];
  }

  ComboId combo(size_t index) const {
    return source_range().combos[source_index(index)];
  }

  double weight(size_t index) const {
    return all_source_hands ? source_range().weights[index] : weights[index];
  }

  CardMask mask(size_t index) const {
    return source_range().masks[source_index(index)];
  }
};

class HandRange {
public:
  HandRange();
  
  // Add an exact two-card combo to the range with a weight.
  void add_combo(ComboId combo_id, double weight);
  
  // Add a hand by index with a weight
  void add_hand_by_index(int index, double weight);
  
  // Get the probability of a specific hand
  double get_probability(ComboId combo_id) const;

  // Get every exact two-card combo represented by the range.
  const WeightedHandRange& get_all_weighted_combos() const;
  
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
  
  // Exact combo expansion cache.
  mutable WeightedHandRange cached_weighted_combos_;
  mutable bool weighted_combos_cache_valid_;
  
  // Total weight of all hands
  double total_weight_;
  
  // Parse a range string component (e.g., "AK", "QQ+", "89s")
  void parse_range_component(const std::string& component);
  
  // Invalidate caches when the range changes
  void invalidate_caches();
  
};

} // namespace poker
