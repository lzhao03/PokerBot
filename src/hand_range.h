#pragma once

#include <cstddef>
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
