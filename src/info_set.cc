#include "src/info_set.h"
#include <sstream>
#include <algorithm>

namespace poker {

std::string InfoSetAbstraction::state_to_info_set(const BoardState& state, int player_id, const Hand& player_hand) const {
  std::ostringstream oss;
  
  // Player ID
  oss << "P" << player_id << ":";
  
  // Player's cards
  oss << "H[";
  for (int i = 0; i < player_hand.cards_size(); ++i) {
    const Card& card = player_hand.cards(i);
    if (i > 0) oss << ",";
    oss << card.rank() << ":" << static_cast<int>(card.suit());
  }
  oss << "]:";
  
  // Board cards
  oss << "B[";
  for (int i = 0; i < state.cards_size(); ++i) {
    const Card& card = state.cards(i);
    if (i > 0) oss << ",";
    oss << card.rank() << ":" << static_cast<int>(card.suit());
  }
  oss << "]:";
  
  // Betting history (placeholder - would need to be extracted from state)
  oss << "A[]";
  
  return oss.str();
}

std::vector<Hand> InfoSetAbstraction::get_possible_hands(const std::string& info_set_key) const {
  // Parse the information set to get the board cards and player's hand
  InfoSetComponents components = parse_info_set(info_set_key);
  
  // In a real implementation, this would generate all possible opponent hands
  // given the known cards (board + player's hand)
  std::vector<Hand> possible_hands;
  
  // This is just a placeholder that returns a single hand
  Hand hand;
  Card* card1 = hand.add_cards();
  card1->set_rank(14); // Ace
  card1->set_suit(Suit::SPADES);
  
  Card* card2 = hand.add_cards();
  card2->set_rank(13); // King
  card2->set_suit(Suit::SPADES);
  
  possible_hands.push_back(hand);
  
  return possible_hands;
}

InfoSetAbstraction::InfoSetComponents InfoSetAbstraction::parse_info_set(const std::string& info_set_key) const {
  InfoSetComponents components;
  
  // This is a simplified parser for the info set key format:
  // P{player_id}:H[{hand_cards}]:B[{board_cards}]:A[{actions}]
  
  size_t player_start = info_set_key.find('P') + 1;
  size_t player_end = info_set_key.find(':', player_start);
  components.player_id = std::stoi(info_set_key.substr(player_start, player_end - player_start));
  
  size_t hand_start = info_set_key.find("H[") + 2;
  size_t hand_end = info_set_key.find(']', hand_start);
  std::string hand_str = info_set_key.substr(hand_start, hand_end - hand_start);
  components.player_cards = string_to_cards(hand_str);
  
  size_t board_start = info_set_key.find("B[") + 2;
  size_t board_end = info_set_key.find(']', board_start);
  std::string board_str = info_set_key.substr(board_start, board_end - board_start);
  components.board_cards = string_to_cards(board_str);
  
  size_t actions_start = info_set_key.find("A[") + 2;
  size_t actions_end = info_set_key.find(']', actions_start);
  std::string actions_str = info_set_key.substr(actions_start, actions_end - actions_start);
  components.betting_history = string_to_betting_history(actions_str);
  
  return components;
}

bool InfoSetAbstraction::same_info_set(const BoardState& state1, const BoardState& state2, int player_id) const {
  // In a real implementation, this would check if two states are indistinguishable
  // from the perspective of the given player
  
  // For now, we'll just check if they have the same number of board cards
  return state1.cards_size() == state2.cards_size();
}

std::string InfoSetAbstraction::info_set_to_string(const std::string& info_set_key) const {
  InfoSetComponents components = parse_info_set(info_set_key);
  
  std::ostringstream oss;
  oss << "Player " << components.player_id << " with ";
  oss << cards_to_string(components.player_cards);
  oss << " on board " << cards_to_string(components.board_cards);
  
  if (!components.betting_history.empty()) {
    oss << " after betting: " << betting_history_to_string(components.betting_history);
  }
  
  return oss.str();
}

std::string InfoSetAbstraction::cards_to_string(const std::vector<Card>& cards) const {
  std::ostringstream oss;
  
  for (size_t i = 0; i < cards.size(); ++i) {
    if (i > 0) oss << ", ";
    
    // Convert rank to string representation
    int rank = cards[i].rank();
    if (rank == 14) oss << 'A';
    else if (rank == 13) oss << 'K';
    else if (rank == 12) oss << 'Q';
    else if (rank == 11) oss << 'J';
    else if (rank == 10) oss << 'T';
    else oss << rank;
    
    // Convert suit to string representation
    switch (cards[i].suit()) {
      case Suit::HEARTS: oss << 'h'; break;
      case Suit::DIAMONDS: oss << 'd'; break;
      case Suit::CLUBS: oss << 'c'; break;
      case Suit::SPADES: oss << 's'; break;
      default: oss << '?'; break;
    }
  }
  
  return oss.str();
}

std::string InfoSetAbstraction::betting_history_to_string(const std::vector<Action>& actions) const {
  std::ostringstream oss;
  
  for (size_t i = 0; i < actions.size(); ++i) {
    if (i > 0) oss << ", ";
    
    switch (actions[i].action()) {
      case ActionType::FOLD: oss << "fold"; break;
      case ActionType::CHECK: oss << "check"; break;
      case ActionType::CALL: oss << "call"; break;
      case ActionType::BET: oss << "bet " << actions[i].amount(); break;
      case ActionType::RAISE: oss << "raise " << actions[i].amount(); break;
      default: oss << "unknown"; break;
    }
  }
  
  return oss.str();
}

std::vector<Card> InfoSetAbstraction::string_to_cards(const std::string& cards_str) const {
  std::vector<Card> cards;
  
  // Parse a string like "14:1,13:4" into cards
  std::istringstream iss(cards_str);
  std::string card_str;
  
  while (std::getline(iss, card_str, ',')) {
    size_t colon_pos = card_str.find(':');
    if (colon_pos != std::string::npos) {
      int rank = std::stoi(card_str.substr(0, colon_pos));
      int suit = std::stoi(card_str.substr(colon_pos + 1));
      
      Card card;
      card.set_rank(rank);
      card.set_suit(static_cast<Suit>(suit));
      cards.push_back(card);
    }
  }
  
  return cards;
}

std::vector<Action> InfoSetAbstraction::string_to_betting_history(const std::string& history_str) const {
  // This is a placeholder - in a real implementation, this would parse a betting history string
  return std::vector<Action>();
}

} // namespace poker
