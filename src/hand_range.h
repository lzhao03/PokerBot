#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <random>
#include <memory>
#include <bitset>
#include <array>
#include "src/poker.pb.h"

namespace poker {

class HandEvaluator;

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
  ~HandRange();
  
  // Add a hand to the range with a weight
  void add_hand(const Hand& hand, double weight);
  
  // Add a hand by index with a weight
  void add_hand_by_index(int index, double weight);
  
  // Sample hands from the range
  std::vector<Hand> sample(int count) const;
  
  // Get the probability of a specific hand
  double get_probability(const Hand& hand) const;
  
  // Get all hands in the range
  std::vector<Hand> get_all_hands() const;

  // Get every exact two-card combo represented by the range.
  WeightedHandRange get_all_weighted_combos() const;
  
  // Get all hand indices with their weights
  const std::vector<std::pair<int, double>>& get_all_weights() const;
  
  // Clear the range
  void clear();
  
  // Set a uniform range (all hands with equal weight)
  void set_uniform_range();
  
  // Set a range from a string representation (e.g., "AK,QQ+,89s")
  void set_from_string(const std::string& range_str);
  
  // Get a string representation of the range
  std::string to_string() const;
  
  // Check if a hand is in the range
  bool contains(const Hand& hand) const;
  
  // Check if an index is in the range
  bool contains_index(int index) const;
  
  // Get the total weight of all hands in the range
  double get_total_weight() const;
  
  // Normalize the weights to sum to 1
  void normalize();
  
  // Pre-compute equity against common ranges
  void precompute_equity(const HandEvaluator& evaluator, const BoardState& board_state);
  
  // Get pre-computed equity for a hand
  double get_precomputed_equity(const Hand& hand) const;
  
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
  
  // Cache of pre-computed equities by index
  std::vector<double> equity_cache_;
  
  // Bitset representation of the range for fast lookups
  // Each bit represents a specific hand configuration
  // 169 possible hand types (13 pairs, 78 suited, 78 offsuit)
  std::bitset<169> range_bitset_;
  
  // Matrix representation for easier manipulation
  // 13x13 matrix where each cell represents a hand type
  // [0][0] = AA, [0][1] = AKs, [1][0] = AKo, etc.
  std::array<std::array<double, 13>, 13> hand_matrix_;
  
  // Cached hands for faster access
  mutable std::vector<Hand> cached_hands_;
  mutable bool hands_cache_valid_;
  
  // Total weight of all hands
  double total_weight_;
  
  // Random number generator for sampling
  mutable std::mt19937 rng_;
  
  // Helper methods
  void update_bitset(int index, bool value);
  void update_matrix_from_index(int index, double weight);
  int matrix_to_index(int row, int col) const;
  std::pair<int, int> index_to_matrix(int index) const;
  
  // Parse a range string component (e.g., "AK", "QQ+", "89s")
  void parse_range_component(const std::string& component);
  
  // Generate all possible hands
  std::vector<Hand> generate_all_hands() const;
  
  // Invalidate caches when the range changes
  void invalidate_caches();
  
};

} // namespace poker
