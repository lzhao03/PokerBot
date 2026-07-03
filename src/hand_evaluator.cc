#include "src/hand_evaluator.h"
#include <set>

namespace poker {
namespace {

HandEvaluation EvaluateFiveCards(const std::array<Card, 5>& cards) {
  std::vector<int> ranks;
  ranks.reserve(cards.size());
  for (const Card& card : cards) {
    ranks.push_back(card.rank());
  }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  bool flush = true;
  for (const Card& card : cards) {
    if (card.suit() != cards[0].suit()) {
      flush = false;
      break;
    }
  }

  const bool wheel_straight = ranks[0] == 14 && ranks[1] == 5 &&
                              ranks[2] == 4 && ranks[3] == 3 &&
                              ranks[4] == 2;
  bool straight = wheel_straight;
  if (!straight) {
    straight = true;
    for (size_t i = 1; i < ranks.size(); ++i) {
      if (ranks[i - 1] != ranks[i] + 1) {
        straight = false;
        break;
      }
    }
  }

  if (flush && ranks[0] == 14 && ranks[1] == 13 && ranks[2] == 12 &&
      ranks[3] == 11 && ranks[4] == 10) {
    return {HandRank::ROYAL_FLUSH, {}};
  }

  if (flush && straight) {
    return {HandRank::STRAIGHT_FLUSH, {ranks[0]}};
  }

  std::array<int, 15> rank_counts = {};
  for (int rank : ranks) {
    ++rank_counts[rank];
  }

  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 4) {
      std::vector<int> kickers = {static_cast<int>(i)};
      for (int rank : ranks) {
        if (rank != static_cast<int>(i)) {
          kickers.push_back(rank);
          break;
        }
      }
      return {HandRank::FOUR_OF_A_KIND, kickers};
    }
  }

  int three_of_a_kind_rank = -1;
  int pair_rank = -1;
  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 3) {
      three_of_a_kind_rank = static_cast<int>(i);
    } else if (rank_counts[i] == 2) {
      pair_rank = static_cast<int>(i);
    }
  }

  if (three_of_a_kind_rank != -1 && pair_rank != -1) {
    return {HandRank::FULL_HOUSE, {three_of_a_kind_rank, pair_rank}};
  }

  if (flush) {
    return {HandRank::FLUSH, ranks};
  }

  if (straight) {
    return {HandRank::STRAIGHT, {ranks[0]}};
  }

  if (three_of_a_kind_rank != -1) {
    std::vector<int> kickers = {three_of_a_kind_rank};
    for (int rank : ranks) {
      if (rank != three_of_a_kind_rank) {
        kickers.push_back(rank);
      }
    }
    return {HandRank::THREE_OF_A_KIND, kickers};
  }

  std::vector<int> pairs;
  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 2) {
      pairs.push_back(static_cast<int>(i));
    }
  }

  if (pairs.size() >= 2) {
    std::sort(pairs.begin(), pairs.end(), std::greater<int>());
    std::vector<int> kickers = {pairs[0], pairs[1]};
    for (int rank : ranks) {
      if (rank != pairs[0] && rank != pairs[1]) {
        kickers.push_back(rank);
        break;
      }
    }
    return {HandRank::TWO_PAIR, kickers};
  }

  if (!pairs.empty()) {
    std::vector<int> kickers = {pairs[0]};
    for (int rank : ranks) {
      if (rank != pairs[0]) {
        kickers.push_back(rank);
      }
    }
    return {HandRank::PAIR, kickers};
  }

  return {HandRank::HIGH_CARD, ranks};
}

}  // namespace

// Move implementations from HandEvaluator to HandEvaluation as static methods
bool HandEvaluation::is_flush(const std::vector<Card>& cards) {
  if (cards.empty()) return false;
  
  Suit firstSuit = cards[0].suit();
  for (const auto& card : cards) {
    if (card.suit() != firstSuit) {
      return false;
    }
  }
  
  return true;
}

bool HandEvaluation::is_straight(const std::vector<int>& ranks) {
  if (ranks.size() < 5) return false;
  
  // Special case: A-5-4-3-2 straight
  if (ranks[0] == 14 && ranks[1] == 5 && ranks[2] == 4 && 
      ranks[3] == 3 && ranks[4] == 2) {
    return true;
  }
  
  // Regular straight check
  for (size_t i = 1; i < ranks.size(); ++i) {
    if (ranks[i-1] != ranks[i] + 1) {
      return false;
    }
  }
  
  return true;
}

bool HandEvaluation::is_straight_flush(const std::vector<Card>& cards) {
  if (!is_flush(cards)) return false;
  
  std::vector<int> ranks;
  for (const auto& card : cards) {
    ranks.push_back(card.rank());
  }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());
  
  return is_straight(ranks);
}

bool HandEvaluation::is_royal_flush(const std::vector<Card>& cards) {
  if (!is_flush(cards)) return false;
  
  std::set<int> royalRanks = {14, 13, 12, 11, 10}; // A, K, Q, J, 10
  std::set<int> handRanks;
  
  for (const auto& card : cards) {
    handRanks.insert(card.rank());
  }
  
  return handRanks == royalRanks;
}

std::vector<int> HandEvaluation::get_rank_counts(const std::vector<int>& ranks) {
  std::vector<int> counts(14, 0); // Ranks 1-14 (A=14)
  
  for (int rank : ranks) {
    counts[rank-1]++;
  }
  
  return counts;
}

// HandEvaluator implementation methods
std::vector<Card> HandEvaluator::hand_to_vector(const Hand& hand) const {
  std::vector<Card> cards;
  for (int i = 0; i < hand.cards_size(); ++i) {
    cards.push_back(hand.cards(i));
  }
  return cards;
}

std::vector<Card> HandEvaluator::board_to_vector(const BoardState& board_state) const {
  std::vector<Card> cards;
  for (int i = 0; i < board_state.cards_size(); ++i) {
    cards.push_back(board_state.cards(i));
  }
  return cards;
}

std::vector<Card> HandEvaluator::combine_cards(const Hand& hand, const BoardState& board_state) const {
  std::vector<Card> combined = hand_to_vector(hand);
  std::vector<Card> board_cards = board_to_vector(board_state);
  combined.insert(combined.end(), board_cards.begin(), board_cards.end());
  return combined;
}

HandEvaluation HandEvaluator::evaluate(const Hand& hand) const {
  std::vector<Card> cards = hand_to_vector(hand);
  
  if (cards.size() != 5) {
    throw std::invalid_argument("Hand evaluation requires exactly 5 cards");
  }

  std::array<Card, 5> five_cards;
  std::copy(cards.begin(), cards.end(), five_cards.begin());
  return EvaluateFiveCards(five_cards);
}

HandEvaluation HandEvaluator::evaluate_hand(const Hand& hole_cards, const BoardState& board_state) const {
  std::vector<Card> all_cards = combine_cards(hole_cards, board_state);
  return find_best_hand(all_cards);
}

int HandEvaluator::compare_hands(const Hand& hand1, const Hand& hand2, const BoardState& board_state) const {
  HandEvaluation eval1 = evaluate_hand(hand1, board_state);
  HandEvaluation eval2 = evaluate_hand(hand2, board_state);
  
  if (eval1 > eval2) {
    return 1;
  } else if (eval1 < eval2) {
    return -1;
  } else {
    return 0;
  }
}

int HandEvaluator::find_winner(const Hand& hand1, const Hand& hand2, const BoardState& board_state) const {
  return compare_hands(hand1, hand2, board_state);
}

double HandEvaluator::calculate_equity(const Hand& hand, const std::vector<Hand>& opponent_range, const BoardState& board_state) const {
  int wins = 0;
  int total = opponent_range.size();
  
  for (const auto& opponent_hand : opponent_range) {
    int result = find_winner(hand, opponent_hand, board_state);
    if (result > 0) {
      wins++;
    } else if (result == 0) {
      // Tie, add half a win
      wins += 0.5;
    }
  }
  
  return static_cast<double>(wins) / total;
}

HandEvaluation HandEvaluator::find_best_hand(const std::vector<Card>& cards) const {
  if (cards.size() < 5) {
    throw std::invalid_argument("Need at least 5 cards to find best hand");
  }

  HandEvaluation bestEval = {HandRank::HIGH_CARD, {}};
  std::array<Card, 5> combo;
  for (size_t a = 0; a + 4 < cards.size(); ++a) {
    combo[0] = cards[a];
    for (size_t b = a + 1; b + 3 < cards.size(); ++b) {
      combo[1] = cards[b];
      for (size_t c = b + 1; c + 2 < cards.size(); ++c) {
        combo[2] = cards[c];
        for (size_t d = c + 1; d + 1 < cards.size(); ++d) {
          combo[3] = cards[d];
          for (size_t e = d + 1; e < cards.size(); ++e) {
            combo[4] = cards[e];
            HandEvaluation currentEval = EvaluateFiveCards(combo);
            if (currentEval > bestEval) {
              bestEval = currentEval;
            }
          }
        }
      }
    }
  }
  
  return bestEval;
}

} // namespace poker
