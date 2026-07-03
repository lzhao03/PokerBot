#include "src/info_set.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace poker {

namespace {

void AppendInt(std::string* out, int value) {
  out->append(std::to_string(value));
}

void AppendNumber(std::string* out, double value) {
  const double rounded = std::round(value);
  if (std::fabs(value - rounded) < 1e-6) {
    out->append(std::to_string(static_cast<int>(rounded)));
    return;
  }

  std::ostringstream oss;
  oss << value;
  out->append(oss.str());
}

void AppendCard(std::string* out, const Card& card) {
  AppendInt(out, card.rank());
  out->push_back(':');
  AppendInt(out, static_cast<int>(card.suit()));
}

std::string ActionHistoryKey(const ActionHistory& history) {
  std::string key;
  key.reserve(history.actions_size() * 8);
  for (int i = 0; i < history.actions_size(); ++i) {
    const Action& action = history.actions(i);
    if (i > 0) {
      key.push_back(',');
    }
    AppendInt(&key, action.player());
    key.push_back(':');
    AppendInt(&key, static_cast<int>(action.action()));
    key.push_back(':');
    AppendNumber(&key, action.amount());
  }
  return key;
}

std::string PublicStateKey(const BoardState& state) {
  std::string key;
  key.reserve(64);
  key.push_back('S');
  AppendInt(&key, static_cast<int>(state.street()));
  key.append(":POT");
  AppendInt(&key, state.pot());
  key.append(":ST");
  AppendInt(&key, state.stack_a());
  key.push_back(',');
  AppendInt(&key, state.stack_b());
  key.append(":AI");
  AppendInt(&key, state.all_in() ? 1 : 0);
  key.append(":F");
  AppendInt(&key, state.folded_player());
  key.append(":T");
  AppendInt(&key, state.player_to_act());
  key.append(":C[");
  for (int i = 0; i < state.player_contribution_size(); ++i) {
    if (i > 0) {
      key.push_back(',');
    }
    AppendNumber(&key, state.player_contribution(i));
  }
  key.push_back(']');
  return key;
}

bool SameCard(const Card& left, const Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

bool ContainsCard(const std::vector<Card>& cards, const Card& candidate) {
  return std::any_of(cards.begin(), cards.end(), [&](const Card& card) {
    return SameCard(card, candidate);
  });
}

Hand MakeHand(const Card& first, const Card& second) {
  Hand hand;
  *hand.add_cards() = first;
  *hand.add_cards() = second;
  return hand;
}

}  // namespace

std::string InfoSetAbstraction::state_to_info_set(const BoardState& state, int player_id, const Hand& player_hand) const {
  std::string key;
  key.reserve(128 + state.history().actions_size() * 8);
  
  // Player ID
  key.push_back('P');
  AppendInt(&key, player_id);
  key.push_back(':');
  
  // Player's cards
  key.append("H[");
  for (int i = 0; i < player_hand.cards_size(); ++i) {
    const Card& card = player_hand.cards(i);
    if (i > 0) key.push_back(',');
    AppendCard(&key, card);
  }
  key.append("]:");
  
  // Board cards
  key.append("B[");
  for (int i = 0; i < state.cards_size(); ++i) {
    const Card& card = state.cards(i);
    if (i > 0) key.push_back(',');
    AppendCard(&key, card);
  }
  key.append("]:");
  
  key.append(PublicStateKey(state));
  key.append(":A[");
  key.append(ActionHistoryKey(state.history()));
  key.push_back(']');
  
  return key;
}

std::vector<Hand> InfoSetAbstraction::get_possible_hands(const std::string& info_set_key) const {
  // Parse the information set to get the board cards and player's hand
  InfoSetComponents components = parse_info_set(info_set_key);

  std::vector<Card> known_cards = components.board_cards;
  known_cards.insert(known_cards.end(), components.player_cards.begin(),
                     components.player_cards.end());

  std::vector<Card> deck;
  deck.reserve(52);
  for (int suit = Suit::HEARTS; suit <= Suit::SPADES; ++suit) {
    for (int rank = 2; rank <= 14; ++rank) {
      Card card;
      card.set_rank(rank);
      card.set_suit(static_cast<Suit>(suit));
      if (!ContainsCard(known_cards, card)) {
        deck.push_back(card);
      }
    }
  }

  std::vector<Hand> possible_hands;
  possible_hands.reserve(deck.size() * (deck.size() - 1) / 2);
  for (size_t i = 0; i < deck.size(); ++i) {
    for (size_t j = i + 1; j < deck.size(); ++j) {
      possible_hands.push_back(MakeHand(deck[i], deck[j]));
    }
  }

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
  (void)player_id;
  if (state1.cards_size() != state2.cards_size()) {
    return false;
  }
  for (int i = 0; i < state1.cards_size(); ++i) {
    if (state1.cards(i).rank() != state2.cards(i).rank() ||
        state1.cards(i).suit() != state2.cards(i).suit()) {
      return false;
    }
  }
  return PublicStateKey(state1) == PublicStateKey(state2) &&
         ActionHistoryKey(state1.history()) == ActionHistoryKey(state2.history());
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
  std::vector<Action> actions;
  std::istringstream iss(history_str);
  std::string action_str;

  while (std::getline(iss, action_str, ',')) {
    size_t first_colon = action_str.find(':');
    size_t second_colon = action_str.find(':', first_colon + 1);
    if (first_colon == std::string::npos || second_colon == std::string::npos) {
      continue;
    }

    Action action;
    action.set_player(std::stoi(action_str.substr(0, first_colon)));
    action.set_action(static_cast<ActionType>(
        std::stoi(action_str.substr(first_colon + 1, second_colon - first_colon - 1))));
    action.set_amount(std::stof(action_str.substr(second_colon + 1)));
    actions.push_back(action);
  }

  return actions;
}

} // namespace poker
