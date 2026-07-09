#include "src/game_tree.h"

#include <algorithm>
#include <stdexcept>

#include "src/hand_evaluator.h"

namespace poker {

namespace {

template <typename State>
int StackForState(const State& state, int player) {
  return state.stack[player];
}

template <typename State>
void SetStackForState(State& state, int player, int stack) {
  state.stack[player] = stack;
}

template <typename State>
int ContributionForState(const State& state, int player) {
  return state.player_contribution[player];
}

int BoardCardCount(const CompactPublicState& state) {
  return state.board_count;
}

int HistorySize(const CompactPublicState& state) {
  return state.history_size;
}

GameAction LastAction(const CompactPublicState& state) {
  return MakeGameAction(state.last_action);
}

template <typename State>
int OutstandingToCall(const State& state, int player) {
  return std::max(
      0, ContributionForState(state, Opponent(player)) -
             ContributionForState(state, player));
}

template <typename State>
int FirstPlayerForStreet(const State& state) {
  return state.street == StreetKind::kPreflop ? 0 : 1;
}

template <typename State>
bool BoardComplete(const State& state) {
  return state.street == StreetKind::kRiver &&
         BoardCardCount(state) >= kMaxBoardCards;
}

template <typename State>
bool IsBettingRoundOverForState(const State& state) {
  if (state.folded_player >= 0) {
    return true;
  }
  const bool calls_matched =
      OutstandingToCall(state, 0) == 0 && OutstandingToCall(state, 1) == 0;
  if (state.all_in) {
    const int player = state.player_to_act;
    return calls_matched || !IsPlayer(player) ||
           StackForState(state, player) == 0 ||
           OutstandingToCall(state, player) == 0;
  }
  if (HistorySize(state) == 0 || !calls_matched) {
    return false;
  }

  const GameAction last = LastAction(state);
  if (last.kind == ActionKind::kCall) {
    return HistorySize(state) > 1;
  }
  return last.kind == ActionKind::kCheck &&
         state.player_to_act == FirstPlayerForStreet(state);
}

template <typename State>
bool IsHandOver(const State& state) {
  return BoardComplete(state) && IsBettingRoundOverForState(state);
}

template <typename State>
bool IsTerminalForState(const State& state) {
  return state.folded_player >= 0 || IsHandOver(state);
}

template <typename State>
int PlayerToActForState(const State& state) {
  if (IsTerminalForState(state)) {
    return -1;
  }
  if (IsBettingRoundOverForState(state)) {
    return -1;
  }
  if (IsPlayer(state.player_to_act)) {
    return state.player_to_act;
  }
  return FirstPlayerForStreet(state);
}

void AppendStateHistory(CompactPublicState& state, const GameAction& action) {
  AppendHistoryAction(state, action);
}

void ResetStateHistory(CompactPublicState& state) {
  ResetHistory(state);
}

template <typename State>
int CommitChips(State& state, int player, int requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  const int committed = std::min(requested, StackForState(state, player));
  state.player_contribution[player] += committed;
  SetStackForState(state, player, StackForState(state, player) - committed);
  state.pot += committed;
  if (StackForState(state, player) == 0) {
    state.all_in = true;
  }
  return committed;
}

template <typename State>
void AdvanceStreet(State& state, absl::Span<const CardId> cards) {
  switch (state.street) {
    case StreetKind::kPreflop:
      state.street = StreetKind::kFlop;
      break;
    case StreetKind::kFlop:
      state.street = StreetKind::kTurn;
      break;
    case StreetKind::kTurn:
      state.street = StreetKind::kRiver;
      break;
    case StreetKind::kRiver:
      break;
  }

  for (CardId card : cards) {
    AddBoardCard(state, card);
  }
  ResetStateHistory(state);
  state.player_to_act = FirstPlayerForStreet(state);
}

template <typename State>
State ApplyActionForState(const State& state, const GameAction& action) {
  State new_state = state;

  int player = new_state.player_to_act;
  if (!IsPlayer(player)) {
    player = PlayerToActForState(new_state);
  }
  if (!IsPlayer(player)) {
    throw std::invalid_argument("No player can act in this state");
  }
  if (new_state.folded_player >= 0) {
    throw std::invalid_argument("Cannot act after a player has folded");
  }
  if (StackForState(new_state, player) <= 0) {
    throw std::invalid_argument("Player has no chips to act");
  }

  const int opponent = Opponent(player);
  const int to_call = OutstandingToCall(new_state, player);
  GameAction applied = action;
  applied.player = player;

  switch (action.kind) {
    case ActionKind::kFold:
      applied.amount = 0;
      new_state.folded_player = player;
      new_state.player_to_act = -1;
      break;
    case ActionKind::kCheck:
      if (to_call != 0) {
        throw std::invalid_argument("Cannot check facing a bet");
      }
      applied.amount = 0;
      new_state.player_to_act = opponent;
      break;
    case ActionKind::kCall: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot call without a bet");
      }
      const int committed = CommitChips(new_state, player, to_call);
      applied.amount = committed;
      new_state.player_to_act = opponent;
      break;
    }
    case ActionKind::kBet: {
      if (to_call != 0) {
        throw std::invalid_argument("Cannot bet facing a bet");
      }
      if (action.amount >= StackForState(new_state, player)) {
        throw std::invalid_argument("Use all-in for full-stack bets");
      }
      const int committed = CommitChips(new_state, player, action.amount);
      applied.amount = committed;
      new_state.player_to_act = opponent;
      break;
    }
    case ActionKind::kRaise: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot raise without a bet");
      }
      if (action.amount <= to_call ||
          StackForState(new_state, player) <= to_call) {
        throw std::invalid_argument("Raise must exceed the call amount");
      }
      if (action.amount >= StackForState(new_state, player)) {
        throw std::invalid_argument("Use all-in for full-stack raises");
      }
      const int committed = CommitChips(new_state, player, action.amount);
      applied.amount = committed;
      new_state.player_to_act = opponent;
      break;
    }
    case ActionKind::kAllIn: {
      const int committed =
          CommitChips(new_state, player, StackForState(new_state, player));
      applied.amount = committed;
      new_state.player_to_act = opponent;
      break;
    }
    case ActionKind::kNoAction:
      throw std::invalid_argument("Unknown action type");
  }

  AppendStateHistory(new_state, applied);
  return new_state;
}

template <typename State>
double UtilityForState(const State& state,
                       ComboId player_a_hand,
                       ComboId player_b_hand,
                       const HandEvaluator& hand_evaluator) {
  const double player_a_contribution = ContributionForState(state, 0);

  if (state.folded_player >= 0) {
    if (state.folded_player == 0) {
      return -player_a_contribution;
    }
    return state.pot - player_a_contribution;
  }

  if (BoardCardCount(state) + 2 < 5) {
    return 0.0;
  }

  const int comparison =
      hand_evaluator.compare_hands(player_a_hand, player_b_hand, state);
  if (comparison > 0) {
    return state.pot - player_a_contribution;
  }
  if (comparison < 0) {
    return -player_a_contribution;
  }
  return (state.pot / 2.0) - player_a_contribution;
}

}  // namespace

CompactPublicState ApplyAction(const CompactPublicState& state,
                               const GameAction& action) {
  return ApplyActionForState(state, action);
}

CompactPublicState ApplyChance(const CompactPublicState& state,
                               absl::Span<const CardId> cards) {
  if (IsTerminalForState(state) || PlayerToActForState(state) != -1) {
    throw std::invalid_argument("State is not a chance node");
  }

  CompactPublicState child = state;
  AdvanceStreet(child, cards);
  return child;
}

double GetUtility(const CompactPublicState& state,
                  ComboId player_a_hand,
                  ComboId player_b_hand) {
  static const HandEvaluator evaluator;
  return UtilityForState(state, player_a_hand, player_b_hand, evaluator);
}

bool IsTerminal(const CompactPublicState& state) {
  return IsTerminalForState(state);
}

int GetPlayerToAct(const CompactPublicState& state) {
  return PlayerToActForState(state);
}

bool IsBettingRoundOver(const CompactPublicState& state) {
  return IsBettingRoundOverForState(state);
}

}  // namespace poker
