#include "src/utility_calculator.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace poker {

namespace {

bool HandsOverlap(const Hand& left, const Hand& right) {
  for (const Card& left_card : left.cards()) {
    for (const Card& right_card : right.cards()) {
      if (left_card.rank() == right_card.rank() &&
          left_card.suit() == right_card.suit()) {
        return true;
      }
    }
  }
  return false;
}

bool HandOverlapsBoard(const Hand& hand, const BoardState& state) {
  for (const Card& hand_card : hand.cards()) {
    for (const Card& board_card : state.cards()) {
      if (hand_card.rank() == board_card.rank() &&
          hand_card.suit() == board_card.suit()) {
        return true;
      }
    }
  }
  return false;
}

WeightedHandRange CompatibleWeightedHands(
    const HandRange& range,
    const Hand& player_hand,
    const BoardState& state) {
  WeightedHandRange compatible_hands;
  WeightedHandRange weighted_combos = range.get_all_weighted_combos();
  compatible_hands.reserve(weighted_combos.size());
  for (size_t i = 0; i < weighted_combos.size(); ++i) {
    if (!HandsOverlap(weighted_combos.hands[i], player_hand) &&
        !HandOverlapsBoard(weighted_combos.hands[i], state)) {
      compatible_hands.add(weighted_combos.hands[i], weighted_combos.weights[i]);
    }
  }
  return compatible_hands;
}

}  // namespace

UtilityCalculator::UtilityCalculator() 
  : hand_evaluator_(new HandEvaluator()), calculation_cache_() {
}

UtilityCalculator::~UtilityCalculator() {
  delete hand_evaluator_;
}

// Unified calculation method
double UtilityCalculator::calculate(
    const BoardState& state,
    const Hand& player_hand,
    const std::variant<Hand, HandRange>& opponent,
    int player_id,
    UtilityType type,
    const Action* action) const {
    
  // Check cache first
  std::string cache_key = create_cache_key(state, player_hand, opponent, player_id, type, action);
  auto cache_it = calculation_cache_.find(cache_key);
  if (cache_it != calculation_cache_.end()) {
    return cache_it->second;
  }
  
  // Dispatch to specialized implementation
  double result = 0.0;
  switch (type) {
    case UtilityType::TERMINAL:
      if (std::holds_alternative<Hand>(opponent)) {
        result = calculate_terminal_impl(state, player_hand, std::get<Hand>(opponent), player_id);
      } else {
        throw std::invalid_argument("Terminal utility requires opponent hand, not range");
      }
      break;
      
    case UtilityType::EXPECTED_VALUE:
      if (std::holds_alternative<HandRange>(opponent)) {
        result = calculate_ev_impl(state, player_hand, std::get<HandRange>(opponent), player_id);
      } else {
        // Convert single hand to range with 100% weight
        HandRange single_hand_range;
        single_hand_range.add_hand(std::get<Hand>(opponent), 1.0);
        result = calculate_ev_impl(state, player_hand, single_hand_range, player_id);
      }
      break;
      
    case UtilityType::EQUITY:
      if (std::holds_alternative<HandRange>(opponent)) {
        result = calculate_equity_impl(player_hand, std::get<HandRange>(opponent), state);
      } else {
        // For single hand equity, result is either 1, 0.5, or 0
        int comparison = hand_evaluator_->compare_hands(player_hand, std::get<Hand>(opponent), state);
        result = (comparison > 0) ? 1.0 : (comparison == 0) ? 0.5 : 0.0;
      }
      break;
      
    case UtilityType::SHOWDOWN:
      if (std::holds_alternative<Hand>(opponent)) {
        result = calculate_showdown_impl(player_hand, std::get<Hand>(opponent), state);
      } else {
        throw std::invalid_argument("Showdown value requires opponent hand, not range");
      }
      break;
      
    case UtilityType::FOLD_EQUITY:
      if (std::holds_alternative<HandRange>(opponent) && action != nullptr) {
        result = calculate_fold_equity_impl(state, player_hand, std::get<HandRange>(opponent), *action, player_id);
      } else {
        throw std::invalid_argument("Fold equity requires opponent range and action");
      }
      break;
  }
  
  // Cache the result
  calculation_cache_[cache_key] = result;
  return result;
}

double UtilityCalculator::calculate_terminal_impl(const BoardState& state, 
                                                 const Hand& player_hand, 
                                                 const Hand& opponent_hand, 
                                                 int player_id) const {
  // Handle fold state
  if (state.folded_player() >= 0) {
    return (state.folded_player() == player_id) 
        ? -state.player_contribution(player_id)  // Player folded
        : calculate_pot_size(state);            // Opponent folded
  }
  
  // Handle showdown state
  if (state.street() == Street::RIVER || state.all_in()) {
    return calculate_showdown_impl(player_hand, opponent_hand, state);
  }
  
  return 0.0;  // Should not happen in terminal state
}

double UtilityCalculator::calculate_ev_impl(const BoardState& state,
                                           const Hand& player_hand,
                                           const HandRange& opponent_range,
                                           int player_id) const {
  WeightedHandRange opponent_hands =
      CompatibleWeightedHands(opponent_range, player_hand, state);
  if (opponent_hands.empty()) return 0.0;
  
  double total_ev = 0.0;
  double total_weight = 0.0;
  
  for (size_t i = 0; i < opponent_hands.size(); ++i) {
    double utility =
        calculate_terminal_impl(state, player_hand, opponent_hands.hands[i],
                                player_id);
    
    total_ev += utility * opponent_hands.weights[i];
    total_weight += opponent_hands.weights[i];
  }
  
  return (total_weight > 0.0) ? total_ev / total_weight : 0.0;
}

double UtilityCalculator::calculate_equity_impl(const Hand& player_hand,
                                               const HandRange& opponent_range,
                                               const BoardState& board_state) const {
  WeightedHandRange opponent_hands =
      CompatibleWeightedHands(opponent_range, player_hand, board_state);
  if (opponent_hands.empty()) return 0.5;  // Default 50% equity if range is empty
  
  double total_equity = 0.0;
  double total_weight = 0.0;
  
  for (size_t i = 0; i < opponent_hands.size(); ++i) {
    int comparison =
        hand_evaluator_->compare_hands(player_hand, opponent_hands.hands[i],
                                       board_state);
    
    // Calculate equity: 1 for win, 0.5 for tie, 0 for loss
    double equity = (comparison > 0) ? 1.0 : (comparison == 0) ? 0.5 : 0.0;
    total_equity += equity * opponent_hands.weights[i];
    total_weight += opponent_hands.weights[i];
  }
  
  return (total_weight > 0.0) ? total_equity / total_weight : 0.5;
}

double UtilityCalculator::calculate_showdown_impl(const Hand& player_hand,
                                                 const Hand& opponent_hand,
                                                 const BoardState& board_state) const {
  int comparison = hand_evaluator_->compare_hands(player_hand, opponent_hand, board_state);
  
  if (comparison > 0) return calculate_pot_size(board_state);  // Win
  if (comparison < 0) return -board_state.player_contribution(0);  // Loss
  return 0.0;  // Tie
}

double UtilityCalculator::calculate_fold_equity_impl(const BoardState& state,
                                                    const Hand& player_hand,
                                                    const HandRange& opponent_range,
                                                    const Action& action,
                                                    int player_id) const {
  double fold_value = calculate_pot_size(state);
  double fold_probability = 0.0;
  
  // Calculate fold probability based on action and opponent range
  if (action.action() == ActionType::BET || action.action() == ActionType::RAISE) {
    double odds = pot_odds(state, action.amount());
    
    // Calculate opponent's average equity
    double avg_equity = 0.0;
    double total_weight = 0.0;
    
    WeightedHandRange opponent_hands =
        CompatibleWeightedHands(opponent_range, player_hand, state);
    for (size_t i = 0; i < opponent_hands.size(); ++i) {
      int comparison =
          hand_evaluator_->compare_hands(opponent_hands.hands[i], player_hand,
                                         state);
      double equity = (comparison > 0) ? 1.0 : (comparison == 0) ? 0.5 : 0.0;
      
      avg_equity += equity * opponent_hands.weights[i];
      total_weight += opponent_hands.weights[i];
    }
    
    if (total_weight > 0.0) avg_equity /= total_weight;
    
    // Fold probability based on pot odds vs equity
    fold_probability = std::max(0.0, 1.0 - (avg_equity / odds));
  }
  
  // EV is weighted average of fold and call outcomes
  double call_value = calculate_ev_impl(state, player_hand, opponent_range, player_id);
  return fold_probability * fold_value + (1.0 - fold_probability) * call_value;
}

std::string UtilityCalculator::create_cache_key(
    const BoardState& state,
    const Hand& player_hand,
    const std::variant<Hand, HandRange>& opponent,
    int player_id,
    UtilityType type,
    const Action* action) const {
  
  std::ostringstream oss;
  
  // State info
  oss << "s:" << static_cast<int>(state.street()) << ":" << state.pot() << ":";
  for (int i = 0; i < state.cards_size(); ++i) {
    const Card& card = state.cards(i);
    oss << card.rank() << static_cast<int>(card.suit()) << ",";
  }
  
  // Player hand
  oss << ":p:";
  for (int i = 0; i < player_hand.cards_size(); ++i) {
    const Card& card = player_hand.cards(i);
    oss << card.rank() << static_cast<int>(card.suit()) << ",";
  }
  
  // Opponent
  oss << ":o:";
  if (std::holds_alternative<Hand>(opponent)) {
    const Hand& opponent_hand = std::get<Hand>(opponent);
    for (int i = 0; i < opponent_hand.cards_size(); ++i) {
      const Card& card = opponent_hand.cards(i);
      oss << card.rank() << static_cast<int>(card.suit()) << ",";
    }
  } else {
    // Use hash of range
    oss << "r:" << std::get<HandRange>(opponent).to_string();
  }
  
  // Other info
  oss << ":id:" << player_id << ":t:" << static_cast<int>(type);
  
  // Action if provided
  if (action != nullptr) {
    oss << ":a:" << static_cast<int>(action->action()) << ":" << action->amount();
  }
  
  return oss.str();
}

} // namespace poker
