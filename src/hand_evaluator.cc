#include "src/hand_evaluator.h"
#include <initializer_list>
#include <set>

namespace poker {
namespace {

struct EvaluationScore {
  HandRank rank = HandRank::HIGH_CARD;
  std::array<int, 5> kickers = {};
  size_t kicker_count = 0;

  bool operator<(const EvaluationScore& other) const {
    if (rank != other.rank) {
      return static_cast<int>(rank) < static_cast<int>(other.rank);
    }

    const size_t limit = std::min(kicker_count, other.kicker_count);
    for (size_t i = 0; i < limit; ++i) {
      if (kickers[i] != other.kickers[i]) {
        return kickers[i] < other.kickers[i];
      }
    }
    return kicker_count < other.kicker_count;
  }
};

EvaluationScore MakeScore(HandRank rank, std::initializer_list<int> kickers) {
  EvaluationScore score;
  score.rank = rank;
  score.kicker_count = kickers.size();
  size_t index = 0;
  for (int kicker : kickers) {
    score.kickers[index] = kicker;
    ++index;
  }
  return score;
}

HandEvaluation ToHandEvaluation(const EvaluationScore& score) {
  std::vector<int> kickers;
  kickers.reserve(score.kicker_count);
  for (size_t i = 0; i < score.kicker_count; ++i) {
    kickers.push_back(score.kickers[i]);
  }
  return {score.rank, kickers};
}

EvaluationScore EvaluateFiveCardScore(const std::array<CardId, 5>& cards) {
  std::array<int, 5> ranks;
  size_t rank_count = 0;
  for (CardId card : cards) {
    ranks[rank_count] = RankFromCardId(card);
    ++rank_count;
  }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  bool flush = true;
  for (CardId card : cards) {
    if (SuitFromCardId(card) != SuitFromCardId(cards[0])) {
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
    return MakeScore(HandRank::ROYAL_FLUSH, {});
  }

  if (flush && straight) {
    return MakeScore(HandRank::STRAIGHT_FLUSH, {ranks[0]});
  }

  std::array<int, 15> rank_counts = {};
  for (int rank : ranks) {
    ++rank_counts[rank];
  }

  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 4) {
      int kicker = -1;
      for (int rank : ranks) {
        if (rank != static_cast<int>(i)) {
          kicker = rank;
          break;
        }
      }
      return MakeScore(HandRank::FOUR_OF_A_KIND,
                       {static_cast<int>(i), kicker});
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
    return MakeScore(HandRank::FULL_HOUSE,
                     {three_of_a_kind_rank, pair_rank});
  }

  if (flush) {
    return MakeScore(HandRank::FLUSH,
                     {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
  }

  if (straight) {
    return MakeScore(HandRank::STRAIGHT, {ranks[0]});
  }

  if (three_of_a_kind_rank != -1) {
    std::array<int, 3> kickers = {three_of_a_kind_rank, 0, 0};
    size_t kicker_count = 1;
    for (int rank : ranks) {
      if (rank != three_of_a_kind_rank) {
        kickers[kicker_count] = rank;
        ++kicker_count;
      }
    }
    return MakeScore(HandRank::THREE_OF_A_KIND,
                     {kickers[0], kickers[1], kickers[2]});
  }

  std::array<int, 2> pairs = {};
  size_t pair_count = 0;
  for (size_t i = 1; i < rank_counts.size(); ++i) {
    if (rank_counts[i] == 2) {
      pairs[pair_count] = static_cast<int>(i);
      ++pair_count;
    }
  }

  if (pair_count >= 2) {
    const int high_pair = pairs[1];
    const int low_pair = pairs[0];
    int kicker = -1;
    for (int rank : ranks) {
      if (rank != high_pair && rank != low_pair) {
        kicker = rank;
        break;
      }
    }
    return MakeScore(HandRank::TWO_PAIR, {high_pair, low_pair, kicker});
  }

  if (pair_count > 0) {
    std::array<int, 4> kickers = {pairs[0], 0, 0, 0};
    size_t kicker_count = 1;
    for (int rank : ranks) {
      if (rank != pairs[0]) {
        kickers[kicker_count] = rank;
        ++kicker_count;
      }
    }
    return MakeScore(HandRank::PAIR,
                     {kickers[0], kickers[1], kickers[2], kickers[3]});
  }

  return MakeScore(HandRank::HIGH_CARD,
                   {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
}

}  // namespace

// Move implementations from HandEvaluator to HandEvaluation as static methods
bool HandEvaluation::is_flush(const std::vector<CardId>& cards) {
  if (cards.empty()) return false;
  
  SuitKind first_suit = SuitFromCardId(cards[0]);
  for (CardId card : cards) {
    if (SuitFromCardId(card) != first_suit) {
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

bool HandEvaluation::is_straight_flush(const std::vector<CardId>& cards) {
  if (!is_flush(cards)) return false;
  
  std::vector<int> ranks;
  for (CardId card : cards) {
    ranks.push_back(RankFromCardId(card));
  }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());
  
  return is_straight(ranks);
}

bool HandEvaluation::is_royal_flush(const std::vector<CardId>& cards) {
  if (!is_flush(cards)) return false;
  
  std::set<int> royalRanks = {14, 13, 12, 11, 10}; // A, K, Q, J, 10
  std::set<int> handRanks;
  
  for (CardId card : cards) {
    handRanks.insert(RankFromCardId(card));
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

HandEvaluation HandEvaluator::evaluate(
    const std::array<CardId, 5>& cards) const {
  return ToHandEvaluation(EvaluateFiveCardScore(cards));
}

HandEvaluation HandEvaluator::evaluate_hand(
    ComboId hole_cards,
    const GameState& board_state) const {
  const ComboInfo& combo = GetComboInfo(hole_cards);
  std::vector<CardId> all_cards;
  all_cards.reserve(2 + board_state.board_cards.size());
  all_cards.push_back(combo.card0);
  all_cards.push_back(combo.card1);
  all_cards.insert(all_cards.end(), board_state.board_cards.begin(),
                   board_state.board_cards.end());
  return find_best_hand(all_cards);
}

HandEvaluation HandEvaluator::evaluate_hand(
    ComboId hole_cards,
    const CompactPublicState& board_state) const {
  const ComboInfo& combo = GetComboInfo(hole_cards);
  std::vector<CardId> all_cards;
  all_cards.reserve(2 + board_state.board_count);
  all_cards.push_back(combo.card0);
  all_cards.push_back(combo.card1);
  for (uint8_t i = 0; i < board_state.board_count; ++i) {
    all_cards.push_back(board_state.board_cards[static_cast<size_t>(i)]);
  }
  return find_best_hand(all_cards);
}

int HandEvaluator::compare_hands(
    ComboId hand1,
    ComboId hand2,
    const GameState& board_state) const {
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

int HandEvaluator::compare_hands(
    ComboId hand1,
    ComboId hand2,
    const CompactPublicState& board_state) const {
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

int HandEvaluator::find_winner(
    ComboId hand1,
    ComboId hand2,
    const GameState& board_state) const {
  return compare_hands(hand1, hand2, board_state);
}

int HandEvaluator::find_winner(
    ComboId hand1,
    ComboId hand2,
    const CompactPublicState& board_state) const {
  return compare_hands(hand1, hand2, board_state);
}

double HandEvaluator::calculate_equity(
    ComboId hand,
    const std::vector<ComboId>& opponent_range,
    const GameState& board_state) const {
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

HandEvaluation HandEvaluator::find_best_hand(
    const std::vector<CardId>& cards) const {
  if (cards.size() < 5) {
    throw std::invalid_argument("Need at least 5 cards to find best hand");
  }

  EvaluationScore bestEval;
  std::array<CardId, 5> combo;
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
            EvaluationScore currentEval = EvaluateFiveCardScore(combo);
            if (bestEval < currentEval) {
              bestEval = currentEval;
            }
          }
        }
      }
    }
  }
  
  return ToHandEvaluation(bestEval);
}

} // namespace poker
