#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "absl/types/span.h"

namespace poker {

using CardId = uint8_t;
using CardMask = uint64_t;

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
  int amount = 0;
  int player = -1;
};

// Maximum number of legal actions at any decision node.
// With 3 bet sizes: fold/call + 3 raises + all-in = 6 (facing bet),
// or check + 3 bets + all-in = 5 (no bet). 8 gives headroom.
inline constexpr int kMaxActionsPerNode = 8;

struct BettingState {
  std::array<int, kPlayerCount> stack = {0, 0};
  std::array<int, kPlayerCount> contribution = {0, 0};
  int pot = 0;
  StreetKind street = StreetKind::kPreflop;
  int8_t player_to_act = 0;
  int8_t folded_player = -1;
  uint8_t actions_this_street = 0;
  ActionKind last_action = ActionKind::kNoAction;
  bool all_in = false;
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

struct CompactAction {
  int amount = 0;
  int8_t player = -1;
  ActionKind kind = ActionKind::kNoAction;
};

struct CompactPublicState {
  static constexpr int kMaxHistoryActions = 32;

  std::array<int, kPlayerCount> stack = {0, 0};
  int pot = 0;
  std::array<CardId, kMaxBoardCards> board_cards = {};
  uint8_t board_count = 0;
  CardMask board_mask = 0;
  std::array<int, kMaxHistoryActions> history_amounts = {};
  std::array<int8_t, kMaxHistoryActions> history_players = {};
  std::array<ActionKind, kMaxHistoryActions> history_kinds = {};
  uint16_t history_size = 0;
  CompactAction last_action;
  StreetKind street = StreetKind::kPreflop;
  bool all_in = false;
  int folded_player = -1;
  int player_to_act = 0;
  std::array<int, kPlayerCount> player_contribution = {0, 0};
  int player_contribution_count = 0;
};

inline CompactAction MakeCompactAction(const GameAction& action) {
  return {action.amount, static_cast<int8_t>(action.player), action.kind};
}

inline GameAction MakeGameAction(const CompactAction& action) {
  return {action.kind, action.amount, action.player};
}

inline CompactAction CompactHistoryAction(const CompactPublicState& state,
                                          uint16_t action_index) {
  if (action_index >= state.history_size ||
      action_index >= CompactPublicState::kMaxHistoryActions) {
    throw std::logic_error("Compact history action index out of range");
  }
  const size_t index = static_cast<size_t>(action_index);
  return {state.history_amounts[index],
          state.history_players[index],
          state.history_kinds[index]};
}

inline void AppendHistoryAction(CompactPublicState& state,
                                const GameAction& action) {
  if (state.history_size >= CompactPublicState::kMaxHistoryActions) {
    throw std::logic_error("Compact public state history is full");
  }
  const size_t index = static_cast<size_t>(state.history_size);
  state.history_amounts[index] = action.amount;
  state.history_players[index] = static_cast<int8_t>(action.player);
  state.history_kinds[index] = action.kind;
  state.last_action = MakeCompactAction(action);
  ++state.history_size;
}

inline void ResetHistory(CompactPublicState& state) {
  state.history_size = 0;
  state.last_action = CompactAction{};
}

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

inline void AddBoardCard(CompactPublicState& state, CardId card_id) {
  Board board{state.board_cards, state.board_mask, state.board_count};
  board.add(card_id);
  state.board_cards = board.cards;
  state.board_count = board.count;
  state.board_mask = board.mask;
}

inline void CopyBoardFrom(CompactPublicState& state,
                          const CompactPublicState& source) {
  state.board_cards = source.board_cards;
  state.board_count = source.board_count;
  state.board_mask = source.board_mask;
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

inline Board BoardFromCompact(const CompactPublicState& state) {
  return Board{state.board_cards, state.board_mask, state.board_count};
}

inline void SetBoard(CompactPublicState& state, const Board& board) {
  state.board_cards = board.cards;
  state.board_count = board.count;
  state.board_mask = board.mask;
}

inline BettingState BettingStateFromCompact(const CompactPublicState& state) {
  return BettingState{
      state.stack,
      state.player_contribution,
      state.pot,
      state.street,
      static_cast<int8_t>(state.player_to_act),
      static_cast<int8_t>(state.folded_player),
      static_cast<uint8_t>(
          std::min<uint16_t>(state.history_size, UINT8_MAX)),
      state.last_action.kind,
      state.all_in,
  };
}

inline ExactGameState ExactGameStateFromCompact(
    const CompactPublicState& state) {
  return ExactGameState{BettingStateFromCompact(state), BoardFromCompact(state)};
}

inline CompactPublicState ToCompact(const BettingState& betting,
                                    const Board& board) {
  CompactPublicState state;
  state.stack = betting.stack;
  state.pot = betting.pot;
  SetBoard(state, board);
  state.street = betting.street;
  state.all_in = betting.all_in;
  state.folded_player = betting.folded_player;
  state.player_to_act = betting.player_to_act;
  state.player_contribution = betting.contribution;
  state.last_action.kind = betting.last_action;
  state.history_size = betting.actions_this_street;
  return state;
}

inline CompactPublicState ToCompact(const ExactGameState& state) {
  return ToCompact(state.betting, state.board);
}

}  // namespace poker
