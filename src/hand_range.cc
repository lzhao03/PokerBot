#include "src/hand_range.h"
#include "src/hand_evaluator.h"
#include <sstream>
#include <algorithm>
#include <ctime>
#include <regex>
#include <unordered_set>
#include <cmath>
#include <iomanip>

namespace poker {

namespace {

Hand MakeCombo(int first_rank, Suit first_suit, int second_rank,
               Suit second_suit) {
  Hand hand;
  Card* first = hand.add_cards();
  first->set_rank(first_rank);
  first->set_suit(first_suit);
  Card* second = hand.add_cards();
  second->set_rank(second_rank);
  second->set_suit(second_suit);
  return hand;
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
  : hands_cache_valid_(false),
    weights_cache_valid_(false),
    total_weight_(0.0) {
  // Initialize random number generator with current time
  rng_.seed(static_cast<unsigned int>(std::time(nullptr)));
  
  // Initialize the matrix with zeros
  for (auto& row : hand_matrix_) {
    row.fill(0.0);
  }
  
  // Initialize equity cache with default values
  equity_cache_.resize(169, 0.5);
}

HandRange::~HandRange() {
  // Nothing to clean up
}

void HandRange::add_hand(const Hand& hand, double weight) {
  int index = hand_to_index(hand);
  add_hand_by_index(index, weight);
}

void HandRange::add_hand_by_index(int index, double weight) {
  if (index < 0 || index >= 169 || weight <= 0.0) {
    return;
  }
  
  // Update the bitset
  range_bitset_.set(index);
  
  // Update the matrix
  auto [row, col] = index_to_matrix(index);
  hand_matrix_[row][col] = weight;
  
  // Update the hand weights vector
  bool found = false;
  for (auto& pair : hand_weights_) {
    if (pair.first == index) {
      pair.second = weight;
      found = true;
      break;
    }
  }
  
  if (!found) {
    hand_weights_.emplace_back(index, weight);
  }
  
  // Update total weight
  total_weight_ += weight;
  
  // Invalidate caches
  invalidate_caches();
}

std::vector<Hand> HandRange::sample(int count) const {
  std::vector<Hand> sampled_hands;
  
  if (hand_weights_.empty() || total_weight_ <= 0.0) {
    return sampled_hands;
  }
  
  // Make sure the weight vector is up to date
  if (!weights_cache_valid_) {
    rebuild_weight_vector();
  }
  
  // Create indices vector
  std::vector<int> indices;
  indices.reserve(hand_weights_.size());
  for (const auto& pair : hand_weights_) {
    indices.push_back(pair.first);
  }
  
  // Create a distribution based on hand weights
  std::discrete_distribution<size_t> dist(weight_vector_.begin(), weight_vector_.end());
  
  // Sample hands
  sampled_hands.reserve(count);
  for (int i = 0; i < count; ++i) {
    size_t idx = dist(rng_);
    sampled_hands.push_back(index_to_hand(indices[idx]));
  }
  
  return sampled_hands;
}

double HandRange::get_probability(const Hand& hand) const {
  int index = hand_to_index(hand);
  
  // Check if the hand is in the range
  if (!contains_index(index) || total_weight_ <= 0.0) {
    return 0.0;
  }
  
  // Find the weight
  for (const auto& pair : hand_weights_) {
    if (pair.first == index) {
      return pair.second / total_weight_;
    }
  }
  
  return 0.0;
}

std::vector<Hand> HandRange::get_all_hands() const {
  // Use cached hands if available
  if (hands_cache_valid_ && !cached_hands_.empty()) {
    return cached_hands_;
  }
  
  std::vector<Hand> hands;
  hands.reserve(hand_weights_.size());
  
  for (const auto& pair : hand_weights_) {
    hands.push_back(index_to_hand(pair.first));
  }
  
  // Cache the result
  cached_hands_ = hands;
  hands_cache_valid_ = true;
  
  return hands;
}

std::vector<std::pair<Hand, double>> HandRange::get_all_weighted_combos() const {
  std::vector<std::pair<Hand, double>> weighted_combos;

  for (const auto& hand_weight : hand_weights_) {
    std::vector<Hand> combos = ExpandIndexToCombos(hand_weight.first);
    if (combos.empty()) {
      continue;
    }

    double combo_weight = hand_weight.second / combos.size();
    for (const Hand& combo : combos) {
      weighted_combos.emplace_back(combo, combo_weight);
    }
  }

  return weighted_combos;
}

const std::vector<std::pair<int, double>>& HandRange::get_all_weights() const {
  return hand_weights_;
}

void HandRange::clear() {
  hand_weights_.clear();
  equity_cache_.assign(169, 0.5);
  range_bitset_.reset();
  
  // Clear the matrix
  for (auto& row : hand_matrix_) {
    row.fill(0.0);
  }
  
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
  
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    
    Hand hand = index_to_hand(indices[i]);
    oss << hand_to_string(hand);
  }
  
  return oss.str();
}

bool HandRange::contains(const Hand& hand) const {
  int index = hand_to_index(hand);
  return contains_index(index);
}

bool HandRange::contains_index(int index) const {
  if (index < 0 || index >= 169) {
    return false;
  }
  
  return range_bitset_[index];
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
  
  // Normalize weights in the matrix
  for (auto& row : hand_matrix_) {
    for (auto& weight : row) {
      if (weight > 0.0) {
        weight /= total_weight_;
      }
    }
  }
  
  total_weight_ = 1.0;
  invalidate_caches();
}

void HandRange::precompute_equity(const HandEvaluator& evaluator, const BoardState& board_state) {
  // Clear existing cache
  equity_cache_.assign(169, 0.5);
  
  // Get all hands in the range
  std::vector<Hand> range_hands = get_all_hands();
  std::vector<int> range_indices;
  range_indices.reserve(range_hands.size());
  
  for (const Hand& hand : range_hands) {
    range_indices.push_back(hand_to_index(hand));
  }
  
  // For each hand in our range, compute equity against all other hands
  for (size_t i = 0; i < range_hands.size(); ++i) {
    const Hand& hand = range_hands[i];
    int hand_index = range_indices[i];
    
    double total_equity = 0.0;
    double total_weight = 0.0;
    
    for (size_t j = 0; j < range_hands.size(); ++j) {
      // Skip comparing a hand against itself
      if (i == j) {
        continue;
      }
      
      const Hand& opponent_hand = range_hands[j];
      
      // Get the weight of the opponent hand
      double weight = 0.0;
      for (const auto& pair : hand_weights_) {
        if (pair.first == range_indices[j]) {
          weight = pair.second;
          break;
        }
      }
      
      // Compare the hands
      int comparison = evaluator.compare_hands(hand, opponent_hand, board_state);
      
      // Calculate equity: 1 for win, 0.5 for tie, 0 for loss
      double equity = 0.0;
      if (comparison > 0) {
        equity = 1.0; // Win
      } else if (comparison == 0) {
        equity = 0.5; // Tie
      }
      
      total_equity += equity * weight;
      total_weight += weight;
    }
    
    // Store the equity in the cache
    if (total_weight > 0.0) {
      equity_cache_[hand_index] = total_equity / total_weight;
    }
  }
}

double HandRange::get_precomputed_equity(const Hand& hand) const {
  int index = hand_to_index(hand);
  
  if (index >= 0 && index < 169) {
    return equity_cache_[index];
  }
  
  return 0.5; // Default to 50% if not precomputed
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
  
  // Create the cards
  Card* card1 = hand.add_cards();
  card1->set_rank(rank1);
  card1->set_suit(Suit::SPADES);
  
  Card* card2 = hand.add_cards();
  card2->set_rank(rank2);
  card2->set_suit(is_suited ? Suit::SPADES : Suit::HEARTS);
  
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

void HandRange::update_bitset(int index, bool value) {
  if (index >= 0 && index < 169) {
    if (value) {
      range_bitset_.set(index);
    } else {
      range_bitset_.reset(index);
    }
  }
}

void HandRange::update_matrix_from_index(int index, double weight) {
  auto [row, col] = index_to_matrix(index);
  hand_matrix_[row][col] = weight;
}

int HandRange::matrix_to_index(int row, int col) const {
  if (row < 0 || row >= 13 || col < 0 || col >= 13) {
    return -1;
  }
  
  if (row == col) {
    // Pair
    return row;
  } else if (row > col) {
    // Suited (row > col)
    return 13 + (row * (row - 1) / 2) + col;
  } else {
    // Offsuit (row < col)
    return 91 + (col * (col - 1) / 2) + row;
  }
}

std::pair<int, int> HandRange::index_to_matrix(int index) const {
  if (index < 0 || index >= 169) {
    return {-1, -1};
  }
  
  if (index < 13) {
    // Pair
    return {index, index};
  } else if (index < 91) {
    // Suited
    index -= 13;
    int r1 = static_cast<int>(std::sqrt(2 * index + 0.25) + 0.5);
    int r2 = index - (r1 * (r1 - 1) / 2);
    return {r1, r2}; // row > col for suited
  } else {
    // Offsuit
    index -= 91;
    int r1 = static_cast<int>(std::sqrt(2 * index + 0.25) + 0.5);
    int r2 = index - (r1 * (r1 - 1) / 2);
    return {r2, r1}; // row < col for offsuit
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

std::vector<Hand> HandRange::generate_all_hands() const {
  std::vector<Hand> all_hands;
  all_hands.reserve(169); // 13 pairs + 78 suited + 78 offsuit
  
  // Generate all 169 hand types
  for (int i = 0; i < 169; ++i) {
    all_hands.push_back(index_to_hand(i));
  }
  
  return all_hands;
}

void HandRange::invalidate_caches() {
  hands_cache_valid_ = false;
  weights_cache_valid_ = false;
  cached_hands_.clear();
  weight_vector_.clear();
}

void HandRange::rebuild_weight_vector() const {
  weight_vector_.clear();
  weight_vector_.reserve(hand_weights_.size());
  
  for (const auto& pair : hand_weights_) {
    weight_vector_.push_back(pair.second);
  }
  
  weights_cache_valid_ = true;
}

} // namespace poker
