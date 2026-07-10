#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "absl/types/span.h"

namespace poker {

using CardId = uint8_t;
using CardMask = uint64_t;
using Chips = int32_t;

constexpr int kDeckCardCount = 52;
constexpr int kMaxBoardCards = 5;
constexpr int kPlayerCount = 2;

enum class Player : uint8_t {
  kA = 0,
  kB = 1,
};

inline int PlayerIndex(Player player) {
  return static_cast<int>(player);
}

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
  Chips amount = 0;
  int player = -1;
};

// Maximum number of legal actions at any decision node.
// With 3 bet sizes: fold/call + 3 raises + all-in = 6 (facing bet),
// or check + 3 bets + all-in = 5 (no bet). 8 gives headroom.
inline constexpr int kMaxActionsPerNode = 8;

struct BettingState {
  std::array<Chips, kPlayerCount> stack = {0, 0};
  std::array<Chips, kPlayerCount> committed = {0, 0};
  StreetKind street = StreetKind::kPreflop;
  int8_t player_to_act = 0;
  int8_t folded_player = -1;
  uint8_t actions_this_street = 0;
  GameAction last_action;
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
  // Maximum number of compact public-state rows to allocate. 0 means
  // unlimited. Once this limit is reached, new public states are skipped
  // instead of expanding the sampled branch. Controls peak traversal memory.
  int max_public_states = 0;
  // Number of threads to use for parallel training. 0 or 1 = single-threaded.
  int num_training_threads = 0;
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

struct Board {
  std::array<CardId, kMaxBoardCards> cards = {};
  CardMask mask = 0;
  uint8_t count = 0;

  absl::Span<const CardId> span() const {
    return absl::Span<const CardId>(cards.data(), count);
  }

  bool contains(CardId card) const {
    return (mask & CardBit(card)) != 0;
  }

  void add(CardId card) {
    if (count >= cards.size()) {
      throw std::logic_error("board is full");
    }
    if (contains(card)) {
      throw std::invalid_argument("duplicate board card");
    }
    cards[static_cast<size_t>(count)] = card;
    ++count;
    mask |= CardBit(card);
  }
};

struct ExactGameState {
  BettingState betting;
  Board board;
};

inline int EncodedCard(CardId card_id) {
  return RankFromCardId(card_id) * 8 + 1 + SuitIndex(SuitFromCardId(card_id));
}

inline bool IsPlayer(int player) {
  return player == 0 || player == 1;
}

inline int Opponent(int player) {
  return 1 - player;
}

inline Chips Pot(const BettingState& state) noexcept {
  return state.committed[0] + state.committed[1];
}

inline Chips ToCall(const BettingState& state, int player) noexcept {
  return std::max(Chips{0},
                  state.committed[Opponent(player)] -
                      state.committed[player]);
}

inline bool AnyPlayerAllIn(const BettingState& state) noexcept {
  return state.stack[0] == 0 || state.stack[1] == 0;
}

inline void ValidateBettingState(const BettingState& state) {
  assert(state.stack[0] >= 0);
  assert(state.stack[1] >= 0);
  assert(state.committed[0] >= 0);
  assert(state.committed[1] >= 0);
  assert(state.folded_player >= -1 &&
         state.folded_player < kPlayerCount);
  assert(state.player_to_act >= -1 &&
         state.player_to_act < kPlayerCount);
}

}  // namespace poker
