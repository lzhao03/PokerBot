#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include "src/poker.pb.h"

namespace poker {

enum class HandRank {
  HIGH_CARD = 0,
  PAIR = 1,
  TWO_PAIR = 2,
  THREE_OF_A_KIND = 3,
  STRAIGHT = 4,
  FLUSH = 5,
  FULL_HOUSE = 6,
  FOUR_OF_A_KIND = 7,
  STRAIGHT_FLUSH = 8,
  ROYAL_FLUSH = 9
};

struct HandEvaluation {
  HandRank rank;
  std::vector<int> kickers;  
  
  // Comparison operators
  bool operator<(const HandEvaluation& other) const {
    if (rank != other.rank) {
      return static_cast<int>(rank) < static_cast<int>(other.rank);
    }
    
    for (size_t i = 0; i < std::min(kickers.size(), other.kickers.size()); ++i) {
      if (kickers[i] != other.kickers[i]) {
        return kickers[i] < other.kickers[i];
      }
    }
    
    return kickers.size() < other.kickers.size();
  }
  
  bool operator>(const HandEvaluation& other) const {
    return other < *this;
  }
  
  bool operator==(const HandEvaluation& other) const {
    return rank == other.rank && kickers == other.kickers;
  }

  // Helper methods for evaluating hand types
  static bool is_flush(const std::vector<Card>& cards);
  static bool is_straight(const std::vector<int>& ranks);
  static bool is_straight_flush(const std::vector<Card>& cards);
  static bool is_royal_flush(const std::vector<Card>& cards);
  static std::vector<int> get_rank_counts(const std::vector<int>& ranks);
};

class HandEvaluator {
public:
  HandEvaluator() = default;
  
  // Evaluate a 5-card hand
  HandEvaluation evaluate(const Hand& hand) const;
  
  // Evaluate the best possible hand given hole cards and a board state
  HandEvaluation evaluate_hand(const Hand& hole_cards, const BoardState& board_state) const;
  
  // Compare two hands and return positive if hand1 wins, negative if hand2 wins, 0 if tie
  int compare_hands(const Hand& hand1, const Hand& hand2, const BoardState& board_state) const;
  
  // Find the winner between two hole cards given a board state
  // Returns 1 if hand1 wins, -1 if hand2 wins, 0 if tie
  int find_winner(const Hand& hand1, const Hand& hand2, const BoardState& board_state) const;
  
  // Calculate equity (winning probability) for a hand against a range of hands
  double calculate_equity(const Hand& hand, const std::vector<Hand>& opponent_range, const BoardState& board_state) const;

private:
  // Utility methods for working with protobuf objects
  std::vector<Card> hand_to_vector(const Hand& hand) const;
  std::vector<Card> board_to_vector(const BoardState& board_state) const;
  std::vector<Card> combine_cards(const Hand& hand, const BoardState& board_state) const;
  
  // Find the best 5-card hand from a set of cards
  HandEvaluation find_best_hand(const std::vector<Card>& cards) const;
};

} // namespace poker
