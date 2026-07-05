#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "absl/container/inlined_vector.h"

namespace poker {

using CardId = uint8_t;
using CardMask = uint64_t;

constexpr int kDeckCardCount = 52;
constexpr int kMaxBoardCards = 5;
constexpr int kPlayerCount = 2;

enum class SuitKind : uint8_t {
  kHearts = 0,
  kDiamonds = 1,
  kClubs = 2,
  kSpades = 3,
};

enum class StreetKind : uint8_t {
  kPreflop = 0,
  kFlop = 1,
  kTurn = 2,
  kRiver = 3,
};

enum class ActionKind : uint8_t {
  kNoAction = 0,
  kBet = 1,
  kFold = 2,
  kCall = 3,
  kRaise = 4,
  kCheck = 5,
  kAllIn = 6,
};

struct GameAction {
  ActionKind kind = ActionKind::kNoAction;
  int amount = 0;
  int player = -1;
};

struct SolverConfig {
  std::vector<double> bet_sizes;
  int starting_stack_size = 100;
  int max_depth = 0;
  bool enable_logging = false;
  int small_blind = 1;
  int big_blind = 2;
  int chance_samples = 1;
  std::vector<double> preflop_bet_sizes;
  std::vector<double> flop_bet_sizes;
  std::vector<double> turn_bet_sizes;
  std::vector<double> river_bet_sizes;
  bool regret_only_training = false;
  // Maximum number of info sets to allocate. 0 means unlimited.
  // Once this limit is reached, new (combo, public_state) pairs fall back to
  // uniform strategy rather than allocating new entries. Controls peak memory.
  int max_info_sets = 0;
};

struct GameState {
  int stack[2] = {0, 0};
  int pot = 0;
  absl::InlinedVector<CardId, kMaxBoardCards> board_cards;
  CardMask board_mask = 0;
  absl::InlinedVector<GameAction, 32> history;
  StreetKind street = StreetKind::kPreflop;
  bool all_in = false;
  int folded_player = -1;
  int player_to_act = 0;
  std::array<int, 2> player_contribution = {0, 0};
  int player_contribution_count = 0;

  void set_stack_a(int value) { stack[0] = value; }
  void set_stack_b(int value) { stack[1] = value; }
  void set_pot(int value) { pot = value; }
  template <typename StreetValue>
  void set_street(StreetValue value) {
    street = static_cast<StreetKind>(static_cast<int>(value));
  }
  void set_all_in(bool value) { all_in = value; }
  void set_folded_player(int value) { folded_player = value; }
  void set_player_to_act(int value) { player_to_act = value; }
  void add_player_contribution(int value) {
    if (player_contribution_count < 2) {
      player_contribution[player_contribution_count] = value;
      ++player_contribution_count;
    }
  }
};

inline int SuitIndex(SuitKind suit) {
  return static_cast<int>(suit);
}

inline int RankFromCardId(CardId card_id) {
  return 2 + static_cast<int>(card_id % 13);
}

inline SuitKind SuitFromCardId(CardId card_id) {
  return static_cast<SuitKind>(card_id / 13);
}

inline CardId MakeCardId(int rank, SuitKind suit) {
  if (rank == 1) {
    rank = 14;
  }
  const int rank_index = rank - 2;
  const int suit_index = SuitIndex(suit);
  if (rank_index < 0 || rank_index >= 13 || suit_index < 0 ||
      suit_index >= 4) {
    throw std::invalid_argument("Invalid card");
  }
  return static_cast<CardId>(suit_index * 13 + rank_index);
}

inline CardMask CardBit(CardId card_id) {
  return CardMask{1} << card_id;
}

inline void AddBoardCard(GameState& state, CardId card_id) {
  if (state.board_cards.size() >= kMaxBoardCards) {
    throw std::invalid_argument("Board already has five cards");
  }
  if ((state.board_mask & CardBit(card_id)) != 0) {
    throw std::invalid_argument("Duplicate board card");
  }
  state.board_cards.push_back(card_id);
  state.board_mask |= CardBit(card_id);
}

inline int Stack(const GameState& state, int player) {
  return state.stack[player];
}

inline void SetStack(GameState& state, int player, int stack) {
  state.stack[player] = stack;
}

inline int Contribution(const GameState& state, int player) {
  return state.player_contribution[player];
}

inline int EncodedCard(CardId card_id) {
  return RankFromCardId(card_id) * 8 + 1 + SuitIndex(SuitFromCardId(card_id));
}

inline bool IsPlayer(int player) {
  return player == 0 || player == 1;
}

inline int Opponent(int player) {
  return 1 - player;
}

}  // namespace poker
