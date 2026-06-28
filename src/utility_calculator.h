#pragma once

#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include "src/poker.pb.h"
#include "src/hand_evaluator.h"
#include "src/hand_range.h"

namespace poker {

// Enum for different types of utility calculations
enum class UtilityType {
  TERMINAL,      // Terminal state utility
  EXPECTED_VALUE, // EV against a range
  EQUITY,        // Equity calculation
  SHOWDOWN,      // Showdown value
  FOLD_EQUITY    // Fold equity
};

class UtilityCalculator {
public:
  UtilityCalculator();
  ~UtilityCalculator();
  
  // Core unified calculation method
  double calculate(
      const BoardState& state,
      const Hand& player_hand,
      const std::variant<Hand, HandRange>& opponent,
      int player_id = 0,
      UtilityType type = UtilityType::TERMINAL,
      const Action* action = nullptr) const;
  
  // Legacy wrapper methods for backward compatibility
  double calculate_utility(const BoardState& state, const Hand& player_hand, 
                         const Hand& opponent_hand, int player_id) const {
    return calculate(state, player_hand, opponent_hand, player_id, UtilityType::TERMINAL);
  }
  
  double calculate_expected_value(const BoardState& state, const Hand& player_hand,
                                const HandRange& opponent_range, int player_id) const {
    return calculate(state, player_hand, opponent_range, player_id, UtilityType::EXPECTED_VALUE);
  }
  
  double calculate_equity(const Hand& player_hand, const HandRange& opponent_range,
                        const BoardState& board_state) const {
    return calculate(board_state, player_hand, opponent_range, 0, UtilityType::EQUITY);
  }
  
  double calculate_showdown_value(const Hand& player_hand, const Hand& opponent_hand,
                                const BoardState& board_state) const {
    return calculate(board_state, player_hand, opponent_hand, 0, UtilityType::SHOWDOWN);
  }
  
  double calculate_fold_equity(const BoardState& state, const Hand& player_hand,
                             const HandRange& opponent_range, const Action& action,
                             int player_id) const {
    return calculate(state, player_hand, opponent_range, player_id, UtilityType::FOLD_EQUITY, &action);
  }

private:
  HandEvaluator* hand_evaluator_;
  mutable std::unordered_map<std::string, double> calculation_cache_;
  
  // Helper methods and specialized implementations
  std::string create_cache_key(const BoardState& state, const Hand& player_hand,
                             const std::variant<Hand, HandRange>& opponent,
                             int player_id, UtilityType type, const Action* action = nullptr) const;
  
  double calculate_pot_size(const BoardState& state) const { return state.pot(); }
  double pot_odds(const BoardState& state, double bet_size) const {
    return bet_size / (calculate_pot_size(state) + bet_size);
  }
  
  // Implementation methods
  double calculate_terminal_impl(const BoardState& state, const Hand& player_hand,
                              const Hand& opponent_hand, int player_id) const;
  double calculate_ev_impl(const BoardState& state, const Hand& player_hand,
                         const HandRange& opponent_range, int player_id) const;
  double calculate_equity_impl(const Hand& player_hand, const HandRange& opponent_range,
                            const BoardState& board_state) const;
  double calculate_showdown_impl(const Hand& player_hand, const Hand& opponent_hand,
                              const BoardState& board_state) const;
  double calculate_fold_equity_impl(const BoardState& state, const Hand& player_hand,
                                 const HandRange& opponent_range, const Action& action,
                                 int player_id) const;
};

} // namespace poker
