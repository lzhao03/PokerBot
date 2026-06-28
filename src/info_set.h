#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "src/poker.pb.h"

namespace poker {

class InfoSetAbstraction {
public:
  InfoSetAbstraction() = default;
  
  // Convert a state to an information set key (what the player knows)
  std::string state_to_info_set(const BoardState& state, int player_id, const Hand& player_hand) const;
  
  // Get the possible hands in an information set
  std::vector<Hand> get_possible_hands(const std::string& info_set_key) const;
  
  // Parse an information set key back to its components
  struct InfoSetComponents {
    std::vector<Card> board_cards;
    std::vector<Card> player_cards;
    std::vector<Action> betting_history;
    int player_id;
  };
  InfoSetComponents parse_info_set(const std::string& info_set_key) const;
  
  // Check if two states belong to the same information set for a player
  bool same_info_set(const BoardState& state1, const BoardState& state2, int player_id) const;
  
  // Get a human-readable representation of an information set
  std::string info_set_to_string(const std::string& info_set_key) const;

private:
  // Helper methods
  std::string cards_to_string(const std::vector<Card>& cards) const;
  std::string betting_history_to_string(const std::vector<Action>& actions) const;
  std::vector<Card> string_to_cards(const std::string& cards_str) const;
  std::vector<Action> string_to_betting_history(const std::string& history_str) const;
};

} // namespace poker
