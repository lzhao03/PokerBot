#ifndef POKERBOT_SRC_SIMULATOR_SIMULATOR_H_
#define POKERBOT_SRC_SIMULATOR_SIMULATOR_H_

#include <random>
#include <vector>
#include <array>
#include <algorithm>

#include "src/poker.pb.h"

namespace poker {

// Minimal No-Limit Hold'em simulator supporting heads-up play.
// Responsibilities:
// - Create a shuffled deck and deal hole/board cards
// - Manage BoardState fields (stacks, pot, contributions, street, to-act)
// - Apply basic betting actions (check, bet, call, raise, fold, all-in)
// - Advance streets and determine terminal states and showdown result
class NoLimitHoldemSimulator {
 public:
  struct Config {
    int starting_stack = 100;   // chips per player
    int small_blind = 1;
    int big_blind = 2;
  };

  struct Result {
    bool terminal = false;
    int winner = -1;         // -1 if not decided or split
    bool split = false;      // true if pot split
  };

  explicit NoLimitHoldemSimulator(uint32_t seed = std::random_device{}())
      : rng_(seed) {}

  // Starts a new hand: shuffles, deals hole cards, posts blinds, and initializes state.
  // Returns true on success.
  bool StartNewHand(const Config& cfg, BoardState& state, Hand& p0_hole,
                    Hand& p1_hole);

  // Apply an action for the current player_to_act. Returns false if illegal.
  bool ApplyAction(const Action& action, BoardState& state);

  // If the betting round is complete and hand not terminal, advance to next street
  // and deal community cards. Returns true if advanced.
  bool AdvanceIfReady(BoardState& state);

  // Returns whether the hand is finished (fold or completed showdown logic pending).
  bool IsTerminal(const BoardState& state) const;

  // If terminal and no fold occurred, evaluate showdown and set Result.
  Result Showdown(const BoardState& state, const Hand& p0_hole,
                  const Hand& p1_hole) const;

 private:
  // Simple hand ranking for showdown (0=high card, 1=pair, etc.)
  enum HandRank {
    HIGH_CARD = 0, PAIR = 1, TWO_PAIR = 2, THREE_KIND = 3,
    STRAIGHT = 4, FLUSH = 5, FULL_HOUSE = 6, FOUR_KIND = 7,
    STRAIGHT_FLUSH = 8, ROYAL_FLUSH = 9
  };

  struct HandValue {
    HandRank rank;
    std::array<int, 5> kickers;  // tie-breakers in descending order
    
    bool operator>(const HandValue& other) const {
      if (rank != other.rank) return rank > other.rank;
      for (int i = 0; i < 5; ++i) {
        if (kickers[i] != other.kickers[i]) return kickers[i] > other.kickers[i];
      }
      return false;
    }
  };

  // Deck helpers
  void BuildFreshDeck();
  void ShuffleDeck();
  void DealCommunity(BoardState& state, int n);
  void DealHoles(Hand& p0_hole, Hand& p1_hole);

  // Betting helpers
  bool BettingRoundOver(const BoardState& state) const;
  int ContributionSize(const BoardState& state, int player) const;
  int OutstandingToCall(const BoardState& state, int player) const;

  // Hand evaluation helpers
  HandValue EvaluateBestHand(const Hand& hole, const BoardState& board) const;
  HandValue EvaluateHand(const std::vector<Card>& cards) const;
  std::vector<Card> CombineCards(const Hand& hole, const BoardState& board) const;

  std::vector<Card> deck_;
  size_t deck_index_ = 0;
  std::mt19937 rng_;
};

}  // namespace poker

#endif  // POKERBOT_SRC_SIMULATOR_SIMULATOR_H_
