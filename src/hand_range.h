#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "src/poker.pb.h"

namespace poker {

struct WeightedHandRange {
  std::vector<Hand> hands;
  std::vector<double> weights;

  size_t size() const { return hands.size(); }
  bool empty() const { return hands.empty(); }

  void reserve(size_t count) {
    hands.reserve(count);
    weights.reserve(count);
  }

  void clear() {
    hands.clear();
    weights.clear();
  }

  void add(const Hand& hand, double weight) {
    hands.push_back(hand);
    weights.push_back(weight);
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

  const Hand& hand(size_t index) const {
    return source_range().hands[source_index(index)];
  }

  double weight(size_t index) const {
    return all_source_hands ? source_range().weights[index] : weights[index];
  }
};

class HandRange {
public:
  HandRange();
  
  // Add a hand to the range with a weight
  void add_hand(const Hand& hand, double weight);
  
  // Add a hand by index with a weight
  void add_hand_by_index(int index, double weight);
  
  // Get the probability of a specific hand
  double get_probability(const Hand& hand) const;

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
  
  // Convert a hand to its canonical index (0-168)
  static int hand_to_index(const Hand& hand);
  
  // Convert an index to a hand
  static Hand index_to_hand(int index);
  
  // Convert a hand to a string representation (for display only)
  static std::string hand_to_string(const Hand& hand);
  
  // Convert a string to a hand index (for parsing only)
  static int string_to_index(const std::string& hand_str);

private:
  // Direct mapping from hand index to weight
  // Using a sparse representation for efficiency
  std::vector<std::pair<int, double>> hand_weights_;

  // Exact two-card combos added directly with add_hand().
  std::vector<std::pair<Hand, double>> exact_hand_weights_;
  
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
