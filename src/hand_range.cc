#include "src/hand_range.h"
#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>

namespace poker {

namespace {

Card MakeRangeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

Hand MakeCombo(int first_rank, Suit first_suit, int second_rank,
               Suit second_suit) {
  Hand hand;
  *hand.add_cards() = MakeRangeCard(first_rank, first_suit);
  *hand.add_cards() = MakeRangeCard(second_rank, second_suit);
  return hand;
}

bool SameCard(const Card& left, const Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

bool SameHand(const Hand& left, const Hand& right) {
  if (left.cards_size() != 2 || right.cards_size() != 2) {
    return false;
  }

  return (SameCard(left.cards(0), right.cards(0)) &&
          SameCard(left.cards(1), right.cards(1))) ||
         (SameCard(left.cards(0), right.cards(1)) &&
          SameCard(left.cards(1), right.cards(0)));
}

void AddWeightedCombo(WeightedHandRange& weighted_combos, const Hand& hand,
                      double weight) {
  for (size_t i = 0; i < weighted_combos.size(); ++i) {
    if (SameHand(weighted_combos.hands[i], hand)) {
      weighted_combos.weights[i] += weight;
      return;
    }
  }

  weighted_combos.add(hand, weight);
}

std::string ExactHandKey(const Hand& hand) {
  std::ostringstream oss;
  for (int i = 0; i < hand.cards_size(); ++i) {
    if (i > 0) {
      oss << "-";
    }
    oss << hand.cards(i).rank() << ":"
        << static_cast<int>(hand.cards(i).suit());
  }
  return oss.str();
}

std::vector<Hand> ExpandIndexToCombos(int index) {
  std::vector<Hand> combos;
  const Suit suits[] = {Suit::HEARTS, Suit::DIAMONDS, Suit::CLUBS,
                        Suit::SPADES};

  if (index < 0 || index >= 169) {
    return combos;
  }

  if (index < 13) {
    int rank = index + 2;
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        combos.push_back(MakeCombo(rank, suits[i], rank, suits[j]));
      }
    }
    return combos;
  }

  bool is_suited = index < 91;
  int offset = is_suited ? index - 13 : index - 91;
  int r1 = static_cast<int>(std::sqrt(2 * offset + 0.25) + 0.5);
  int r2 = offset - (r1 * (r1 - 1) / 2);
  int rank1 = r1 + 2;
  int rank2 = r2 + 2;

  for (Suit first_suit : suits) {
    for (Suit second_suit : suits) {
      if ((first_suit == second_suit) == is_suited) {
        combos.push_back(MakeCombo(rank1, first_suit, rank2, second_suit));
      }
    }
  }
  return combos;
}

}  // namespace

HandRange::HandRange() 
  : weighted_combos_cache_valid_(false),
    total_weight_(0.0) {
}

void HandRange::add_hand(const Hand& hand, double weight) {
  if (hand.cards_size() != 2 || weight <= 0.0) {
    return;
  }

  for (auto& pair : exact_hand_weights_) {
    if (SameHand(pair.first, hand)) {
      total_weight_ += weight - pair.second;
      pair.second = weight;
      invalidate_caches();
      return;
    }
  }

  exact_hand_weights_.emplace_back(hand, weight);
  total_weight_ += weight;
  invalidate_caches();
}

void HandRange::add_hand_by_index(int index, double weight) {
  if (index < 0 || index >= 169 || weight <= 0.0) {
    return;
  }
  
  bool found = false;
  for (auto& pair : hand_weights_) {
    if (pair.first == index) {
      total_weight_ += weight - pair.second;
      pair.second = weight;
      found = true;
      break;
    }
  }
  
  if (!found) {
    hand_weights_.emplace_back(index, weight);
    total_weight_ += weight;
  }
  
  invalidate_caches();
}

double HandRange::get_probability(const Hand& hand) const {
  if (hand.cards_size() != 2 || total_weight_ <= 0.0) {
    return 0.0;
  }

  double hand_weight = 0.0;
  for (const auto& pair : exact_hand_weights_) {
    if (SameHand(pair.first, hand)) {
      hand_weight += pair.second;
    }
  }

  int index = hand_to_index(hand);
  if (index >= 0) {
    for (const auto& pair : hand_weights_) {
      if (pair.first == index) {
        std::vector<Hand> combos = ExpandIndexToCombos(index);
        if (!combos.empty()) {
          hand_weight += pair.second / combos.size();
        }
        break;
      }
    }
  }
  
  return hand_weight / total_weight_;
}

const WeightedHandRange& HandRange::get_all_weighted_combos() const {
  if (weighted_combos_cache_valid_) {
    return cached_weighted_combos_;
  }

  cached_weighted_combos_.clear();
  for (const auto& hand_weight : exact_hand_weights_) {
    AddWeightedCombo(cached_weighted_combos_, hand_weight.first,
                     hand_weight.second);
  }

  for (const auto& hand_weight : hand_weights_) {
    std::vector<Hand> combos = ExpandIndexToCombos(hand_weight.first);
    if (combos.empty()) {
      continue;
    }

    double combo_weight = hand_weight.second / combos.size();
    for (const Hand& combo : combos) {
      AddWeightedCombo(cached_weighted_combos_, combo, combo_weight);
    }
  }

  weighted_combos_cache_valid_ = true;
  return cached_weighted_combos_;
}

void HandRange::clear() {
  hand_weights_.clear();
  exact_hand_weights_.clear();
  total_weight_ = 0.0;
  invalidate_caches();
}

void HandRange::set_uniform_range() {
  // Clear current range
  clear();
  
  // Assign uniform weight to all 169 hand types
  double weight = 1.0 / 169.0;
  
  for (int i = 0; i < 169; ++i) {
    add_hand_by_index(i, weight);
  }
}

void HandRange::set_from_string(const std::string& range_str) {
  clear();
  
  // Split the range string by commas
  std::istringstream iss(range_str);
  std::string component;
  
  while (std::getline(iss, component, ',')) {
    // Trim whitespace
    component.erase(0, component.find_first_not_of(" \t"));
    component.erase(component.find_last_not_of(" \t") + 1);
    
    if (!component.empty()) {
      parse_range_component(component);
    }
  }
  
  // Normalize the weights
  normalize();
}

std::string HandRange::to_string() const {
  std::ostringstream oss;
  
  // Sort indices for consistent output
  std::vector<int> indices;
  indices.reserve(hand_weights_.size());
  
  for (const auto& pair : hand_weights_) {
    indices.push_back(pair.first);
  }
  
  std::sort(indices.begin(), indices.end());

  std::vector<std::string> exact_hands;
  exact_hands.reserve(exact_hand_weights_.size());
  for (const auto& pair : exact_hand_weights_) {
    exact_hands.push_back("[" + ExactHandKey(pair.first) + "]");
  }
  std::sort(exact_hands.begin(), exact_hands.end());
  
  bool needs_separator = false;
  for (int index : indices) {
    if (needs_separator) {
      oss << ",";
    }
    
    Hand hand = index_to_hand(index);
    oss << hand_to_string(hand);
    needs_separator = true;
  }

  for (const std::string& hand : exact_hands) {
    if (needs_separator) {
      oss << ",";
    }

    oss << hand;
    needs_separator = true;
  }
  
  return oss.str();
}

double HandRange::get_total_weight() const {
  return total_weight_;
}

void HandRange::normalize() {
  if (total_weight_ <= 0.0) {
    return;
  }
  
  // Normalize weights in the vector
  for (auto& pair : hand_weights_) {
    pair.second /= total_weight_;
  }

  for (auto& pair : exact_hand_weights_) {
    pair.second /= total_weight_;
  }
  
  total_weight_ = 1.0;
  invalidate_caches();
}

int HandRange::hand_to_index(const Hand& hand) {
  if (hand.cards_size() < 2) {
    return -1;
  }
  
  // Extract the ranks and suits
  int rank1 = hand.cards(0).rank();
  int rank2 = hand.cards(1).rank();
  Suit suit1 = hand.cards(0).suit();
  Suit suit2 = hand.cards(1).suit();
  
  // Normalize ranks (2-14) to 0-12
  int r1 = rank1 - 2;
  int r2 = rank2 - 2;
  
  // Ensure r1 >= r2
  if (r1 < r2) {
    std::swap(r1, r2);
    std::swap(suit1, suit2);
  }
  
  // Calculate index:
  // - Pairs: 0-12 (13 total)
  // - Suited: 13-90 (78 total)
  // - Offsuit: 91-168 (78 total)
  if (r1 == r2) {
    // Pair
    return r1;
  } else if (suit1 == suit2) {
    // Suited
    return 13 + (r1 * (r1 - 1) / 2) + r2;
  } else {
    // Offsuit
    return 91 + (r1 * (r1 - 1) / 2) + r2;
  }
}

Hand HandRange::index_to_hand(int index) {
  Hand hand;
  
  if (index < 0 || index >= 169) {
    return hand;
  }
  
  int r1, r2;
  bool is_suited;
  
  if (index < 13) {
    // Pair
    r1 = r2 = index;
    is_suited = false;
  } else if (index < 91) {
    // Suited
    index -= 13;
    // Solve for r1: (r1 * (r1 - 1) / 2) + r2 = index
    r1 = static_cast<int>(std::sqrt(2 * index + 0.25) + 0.5);
    r2 = index - (r1 * (r1 - 1) / 2);
    is_suited = true;
  } else {
    // Offsuit
    index -= 91;
    // Solve for r1: (r1 * (r1 - 1) / 2) + r2 = index
    r1 = static_cast<int>(std::sqrt(2 * index + 0.25) + 0.5);
    r2 = index - (r1 * (r1 - 1) / 2);
    is_suited = false;
  }
  
  // Convert back to ranks (2-14)
  int rank1 = r1 + 2;
  int rank2 = r2 + 2;
  
  *hand.add_cards() = MakeRangeCard(rank1, Suit::SPADES);
  *hand.add_cards() =
      MakeRangeCard(rank2, is_suited ? Suit::SPADES : Suit::HEARTS);
  
  return hand;
}

std::string HandRange::hand_to_string(const Hand& hand) {
  if (hand.cards_size() < 2) {
    return "";
  }
  
  // Extract the ranks and suits
  int rank1 = hand.cards(0).rank();
  int rank2 = hand.cards(1).rank();
  Suit suit1 = hand.cards(0).suit();
  Suit suit2 = hand.cards(1).suit();
  
  // Ensure rank1 >= rank2
  if (rank1 < rank2) {
    std::swap(rank1, rank2);
    std::swap(suit1, suit2);
  }
  
  // Convert ranks to characters
  char rank1_char, rank2_char;
  
  if (rank1 == 14) rank1_char = 'A';
  else if (rank1 == 13) rank1_char = 'K';
  else if (rank1 == 12) rank1_char = 'Q';
  else if (rank1 == 11) rank1_char = 'J';
  else if (rank1 == 10) rank1_char = 'T';
  else rank1_char = '0' + rank1;
  
  if (rank2 == 14) rank2_char = 'A';
  else if (rank2 == 13) rank2_char = 'K';
  else if (rank2 == 12) rank2_char = 'Q';
  else if (rank2 == 11) rank2_char = 'J';
  else if (rank2 == 10) rank2_char = 'T';
  else rank2_char = '0' + rank2;
  
  // Create the key
  std::string key;
  key += rank1_char;
  key += rank2_char;
  
  // Add suited/offsuit suffix
  if (suit1 == suit2 && rank1 != rank2) {
    key += 's'; // suited
  } else if (rank1 != rank2) {
    key += 'o'; // offsuit
  }
  
  return key;
}

int HandRange::string_to_index(const std::string& hand_str) {
  if (hand_str.length() < 2) {
    return -1;
  }
  
  // Parse the key
  char rank1_char = hand_str[0];
  char rank2_char = hand_str[1];
  bool is_suited = (hand_str.length() > 2 && hand_str[2] == 's');
  
  // Convert characters to ranks
  int rank1, rank2;
  
  if (rank1_char == 'A') rank1 = 14;
  else if (rank1_char == 'K') rank1 = 13;
  else if (rank1_char == 'Q') rank1 = 12;
  else if (rank1_char == 'J') rank1 = 11;
  else if (rank1_char == 'T') rank1 = 10;
  else rank1 = rank1_char - '0';
  
  if (rank2_char == 'A') rank2 = 14;
  else if (rank2_char == 'K') rank2 = 13;
  else if (rank2_char == 'Q') rank2 = 12;
  else if (rank2_char == 'J') rank2 = 11;
  else if (rank2_char == 'T') rank2 = 10;
  else rank2 = rank2_char - '0';
  
  // Ensure rank1 >= rank2
  if (rank1 < rank2) {
    std::swap(rank1, rank2);
  }
  
  // Normalize ranks (2-14) to 0-12
  int r1 = rank1 - 2;
  int r2 = rank2 - 2;
  
  // Calculate index:
  if (r1 == r2) {
    // Pair
    return r1;
  } else if (is_suited) {
    // Suited
    return 13 + (r1 * (r1 - 1) / 2) + r2;
  } else {
    // Offsuit or unspecified (treat as offsuit)
    return 91 + (r1 * (r1 - 1) / 2) + r2;
  }
}

void HandRange::parse_range_component(const std::string& component) {
  // Check for pocket pairs with a plus (e.g., "QQ+")
  std::regex pair_plus_regex("([AKQJT98765432])(\\1)\\+");
  std::smatch pair_plus_match;
  
  if (std::regex_match(component, pair_plus_match, pair_plus_regex)) {
    char rank_char = pair_plus_match[1].str()[0];
    int start_rank;
    
    if (rank_char == 'A') start_rank = 14;
    else if (rank_char == 'K') start_rank = 13;
    else if (rank_char == 'Q') start_rank = 12;
    else if (rank_char == 'J') start_rank = 11;
    else if (rank_char == 'T') start_rank = 10;
    else start_rank = rank_char - '0';
    
    // Add all pairs from the specified rank up to AA
    for (int rank = start_rank; rank <= 14; ++rank) {
      int r = rank - 2; // Normalize to 0-12
      add_hand_by_index(r, 1.0);
    }
    
    return;
  }
  
  // Check for suited connectors with a plus (e.g., "89s+")
  std::regex suited_plus_regex("([AKQJT98765432])([AKQJT98765432])s\\+");
  std::smatch suited_plus_match;
  
  if (std::regex_match(component, suited_plus_match, suited_plus_regex)) {
    // Not implemented in this simplified version
    return;
  }
  
  // Check for simple hand (e.g., "AK", "QJs", "T9o")
  std::regex simple_hand_regex("([AKQJT98765432])([AKQJT98765432])(s|o)?");
  std::smatch simple_hand_match;
  
  if (std::regex_match(component, simple_hand_match, simple_hand_regex)) {
    std::string hand_str = component;
    int index = string_to_index(hand_str);
    
    if (index >= 0) {
      add_hand_by_index(index, 1.0);
    }
  }
}

void HandRange::invalidate_caches() {
  weighted_combos_cache_valid_ = false;
  cached_weighted_combos_.clear();
}

} // namespace poker
