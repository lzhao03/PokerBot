#include "src/game_tree.h"

#include <algorithm>
#include <stdexcept>

#include "src/hand_evaluator.h"

namespace poker {
namespace {

int FirstPlayerForStreet(StreetKind street) {
  return street == StreetKind::kPreflop ? 0 : 1;
}

bool BoardComplete(const BettingState& state, const Board& board) {
  return state.street == StreetKind::kRiver && board.count >= kMaxBoardCards;
}

Chips CommitChips(BettingState& state, int player, Chips requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  const Chips committed = std::min(requested, state.stack[player]);
  state.committed[player] += committed;
  state.stack[player] -= committed;
  return committed;
}

void AdvanceStreet(ExactGameState& state, absl::Span<const CardId> cards) {
  switch (state.betting.street) {
    case StreetKind::kPreflop:
      state.betting.street = StreetKind::kFlop;
      break;
    case StreetKind::kFlop:
      state.betting.street = StreetKind::kTurn;
      break;
    case StreetKind::kTurn:
      state.betting.street = StreetKind::kRiver;
      break;
    case StreetKind::kRiver:
      break;
  }

  for (CardId card : cards) {
    state.board.add(card);
  }
  state.betting.pending_action_mask = kAllPlayersMask;
  state.betting.player_to_act = FirstPlayerForStreet(state.betting.street);
  ValidateBettingState(state.betting);
}

bool HandOver(const BettingState& state, const Board& board) {
  return BoardComplete(state, board) && IsBettingRoundOver(state);
}

int PlayerToAct(const BettingState& state) {
  if (state.folded_player >= 0 || IsBettingRoundOver(state)) {
    return -1;
  }
  if (IsPlayer(state.player_to_act)) {
    return state.player_to_act;
  }
  return FirstPlayerForStreet(state.street);
}

}  // namespace

bool IsBettingRoundOver(const BettingState& state) noexcept {
  if (state.folded_player >= 0) {
    return true;
  }
  if (AnyPlayerAllIn(state)) {
    const int player = state.player_to_act;
    return !IsPlayer(player) || state.stack[player] == 0 ||
           ToCall(state, player) == 0;
  }
  return state.pending_action_mask == 0 && ToCall(state, 0) == 0 &&
         ToCall(state, 1) == 0;
}

bool IsTerminal(const BettingState& state, const Board& board) {
  return state.folded_player >= 0 || HandOver(state, board);
}

int GetPlayerToAct(const BettingState& state, const Board& board) {
  if (IsTerminal(state, board) || IsBettingRoundOver(state)) {
    return -1;
  }
  if (IsPlayer(state.player_to_act)) {
    return state.player_to_act;
  }
  return FirstPlayerForStreet(state.street);
}

bool IsLegalAction(const BettingState& state, const GameAction& action) {
  int player = state.player_to_act;
  if (!IsPlayer(player)) {
    player = PlayerToAct(state);
  }
  if (!IsPlayer(player) || state.folded_player >= 0 ||
      state.stack[player] <= 0) {
    return false;
  }

  const Chips to_call = ToCall(state, player);
  switch (action.kind) {
    case ActionKind::kFold:
      return true;
    case ActionKind::kCheck:
      return to_call == 0;
    case ActionKind::kCall:
      return to_call > 0;
    case ActionKind::kBet:
      return to_call == 0 && action.amount > 0 &&
             action.amount < state.stack[player];
    case ActionKind::kRaise:
      return to_call > 0 && action.amount > to_call &&
             state.stack[player] > to_call &&
             action.amount < state.stack[player];
    case ActionKind::kAllIn:
      return true;
    case ActionKind::kNoAction:
      return false;
  }
}

BettingState ApplyLegalActionUnchecked(const BettingState& state,
                                       const GameAction& action) {
  BettingState child = state;
  int player = child.player_to_act;
  if (!IsPlayer(player)) {
    player = PlayerToAct(child);
  }

  const int opponent = Opponent(player);
  const Chips to_call_before = ToCall(child, player);
  Chips committed = 0;

  switch (action.kind) {
    case ActionKind::kFold:
      child.folded_player = player;
      child.player_to_act = -1;
      break;
    case ActionKind::kCheck:
      child.player_to_act = opponent;
      break;
    case ActionKind::kCall:
      committed = CommitChips(child, player, to_call_before);
      child.player_to_act = opponent;
      break;
    case ActionKind::kBet:
    case ActionKind::kRaise:
      committed = CommitChips(child, player, action.amount);
      child.player_to_act = opponent;
      break;
    case ActionKind::kAllIn:
      committed = CommitChips(child, player, child.stack[player]);
      child.player_to_act = opponent;
      break;
    case ActionKind::kNoAction:
      break;
  }

  const bool aggressive =
      action.kind == ActionKind::kBet ||
      action.kind == ActionKind::kRaise ||
      (action.kind == ActionKind::kAllIn && committed > to_call_before);
  if (action.kind != ActionKind::kFold) {
    if (aggressive) {
      child.pending_action_mask = PlayerBit(opponent);
    } else {
      child.pending_action_mask &=
          static_cast<uint8_t>(~PlayerBit(player));
    }
  }

  ValidateBettingState(child);
  return child;
}

BettingState ApplyAction(const BettingState& state,
                         const GameAction& action) {
  if (!IsLegalAction(state, action)) {
    throw std::invalid_argument("illegal poker action");
  }
  return ApplyLegalActionUnchecked(state, action);
}

ExactGameState ApplyChance(const ExactGameState& state,
                           absl::Span<const CardId> cards) {
  if (IsTerminal(state.betting, state.board) ||
      GetPlayerToAct(state.betting, state.board) != -1) {
    throw std::invalid_argument("State is not a chance node");
  }

  ExactGameState child = state;
  AdvanceStreet(child, cards);
  return child;
}

double GetUtility(const ExactGameState& state,
                  ComboId player_a_hand,
                  ComboId player_b_hand) {
  static const HandEvaluator evaluator;
  const double a_committed = state.betting.committed[0];

  if (state.betting.folded_player >= 0) {
    if (state.betting.folded_player == 0) {
      return -a_committed;
    }
    return Pot(state.betting) - a_committed;
  }

  if (state.board.count + 2 < 5) {
    return 0.0;
  }

  const int comparison =
      evaluator.compare_hands(player_a_hand, player_b_hand, state.board);
  if (comparison > 0) {
    return Pot(state.betting) - a_committed;
  }
  if (comparison < 0) {
    return -a_committed;
  }
  return (Pot(state.betting) / 2.0) - a_committed;
}

}  // namespace poker
