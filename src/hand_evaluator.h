#pragma once

#include <array>
#include <vector>
#include <algorithm>

#include "src/combo.h"
#include "src/poker_types.h"

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
  static bool is_flush(const std::vector<CardId>& cards);
  static bool is_straight(const std::vector<int>& ranks);
  static bool is_straight_flush(const std::vector<CardId>& cards);
  static bool is_royal_flush(const std::vector<CardId>& cards);
  static std::vector<int> get_rank_counts(const std::vector<int>& ranks);
};

class HandEvaluator {
public:
  HandEvaluator() = default;
  
  // Evaluate a 5-card hand.
  HandEvaluation evaluate(const std::array<CardId, 5>& cards) const;
  
  // Evaluate the best possible hand given hole cards and a board state
  HandEvaluation evaluate_hand(ComboId hole_cards,
                               const GameState& board_state) const;
  
  // Compare two hands and return positive if hand1 wins, negative if hand2 wins, 0 if tie
  int compare_hands(ComboId hand1, ComboId hand2,
                    const GameState& board_state) const;
  
  // Find the winner between two hole cards given a board state
  // Returns 1 if hand1 wins, -1 if hand2 wins, 0 if tie
  int find_winner(ComboId hand1, ComboId hand2,
                  const GameState& board_state) const;
  
  // Calculate equity (winning probability) for a hand against a range of hands
  double calculate_equity(ComboId hand, const std::vector<ComboId>& opponent_range,
                          const GameState& board_state) const;

private:
  // Find the best 5-card hand from a set of cards
  HandEvaluation find_best_hand(const std::vector<CardId>& cards) const;
};

} // namespace poker
