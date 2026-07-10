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

void AppendAction(BettingState& state, const GameAction& action) {
  state.last_action = action;
  if (state.actions_this_street == UINT8_MAX) {
    throw std::logic_error("Betting state action count overflow");
  }
  ++state.actions_this_street;
}

void ResetActions(BettingState& state) {
  state.actions_this_street = 0;
  state.last_action = GameAction{};
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
  ResetActions(state.betting);
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

bool IsBettingRoundOver(const BettingState& state) {
  if (state.folded_player >= 0) {
    return true;
  }
  const bool calls_matched = ToCall(state, 0) == 0 && ToCall(state, 1) == 0;
  if (AnyPlayerAllIn(state)) {
    const int player = state.player_to_act;
    return calls_matched || !IsPlayer(player) || state.stack[player] == 0 ||
           ToCall(state, player) == 0;
  }
  if (state.actions_this_street == 0 || !calls_matched) {
    return false;
  }

  if (state.last_action.kind == ActionKind::kCall) {
    return state.actions_this_street > 1;
  }
  return state.last_action.kind == ActionKind::kCheck &&
         state.player_to_act == FirstPlayerForStreet(state.street);
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
  const Chips to_call = ToCall(child, player);
  GameAction applied = action;
  applied.player = player;

  switch (action.kind) {
    case ActionKind::kFold:
      applied.amount = 0;
      child.folded_player = player;
      child.player_to_act = -1;
      break;
    case ActionKind::kCheck:
      applied.amount = 0;
      child.player_to_act = opponent;
      break;
    case ActionKind::kCall:
      applied.amount = CommitChips(child, player, to_call);
      child.player_to_act = opponent;
      break;
    case ActionKind::kBet:
    case ActionKind::kRaise:
      applied.amount = CommitChips(child, player, action.amount);
      child.player_to_act = opponent;
      break;
    case ActionKind::kAllIn:
      applied.amount = CommitChips(child, player, child.stack[player]);
      child.player_to_act = opponent;
      break;
    case ActionKind::kNoAction:
      break;
  }

  AppendAction(child, applied);
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
