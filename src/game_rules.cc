#include "src/game_rules.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "absl/container/inlined_vector.h"
#include "src/card_utils.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

int FirstPlayerForStreet(StreetKind street) {
  return street == StreetKind::kPreflop ? 0 : 1;
}

bool BoardComplete(const BettingState& state, const BoardRunout& board) {
  return state.street == StreetKind::kRiver &&
         board.count() == kMaxBoardCards;
}

Chips CommitChips(BettingState& state, int player, Chips requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  const Chips committed = std::min(requested, state.stack[player]);
  state.stack[player] -= committed;
  state.total_committed[player] += committed;
  state.street_committed[player] += committed;
  return committed;
}

void AdvanceStreet(ExactPublicState& state,
                   absl::Span<const CardId> cards,
                   const BettingRules& rules) {
  switch (state.betting.street) {
    case StreetKind::kPreflop:
      state.board.deal_flop(cards);
      state.betting.street = StreetKind::kFlop;
      break;
    case StreetKind::kFlop:
      state.board.deal_turn(cards[0]);
      state.betting.street = StreetKind::kTurn;
      break;
    case StreetKind::kTurn:
      state.board.deal_river(cards[0]);
      state.betting.street = StreetKind::kRiver;
      break;
    case StreetKind::kRiver:
      break;
  }
  state.betting.street_committed = {0, 0};
  state.betting.last_full_raise = rules.minimum_bet;
  state.betting.pending_action_mask = kAllPlayersMask;
  state.betting.player_to_act = FirstPlayerForStreet(state.betting.street);
  if (IsBettingRoundOver(state.betting)) {
    state.betting.player_to_act = -1;
  }
  assert(IsValidBettingState(state.betting));
}

void RefundUnmatchedCommitment(BettingState& state) {
  if (state.folded_player >= 0 ||
      state.street_committed[0] == state.street_committed[1]) {
    return;
  }
  const int player = state.street_committed[0] > state.street_committed[1]
                         ? 0
                         : 1;
  const Chips excess =
      state.street_committed[player] -
      state.street_committed[Opponent(player)];
  state.street_committed[player] -= excess;
  state.total_committed[player] -= excess;
  state.stack[player] += excess;
}

bool HandOver(const BettingState& state, const BoardRunout& board) {
  return BoardComplete(state, board) && IsBettingRoundOver(state);
}

Chips ConcreteBetAmount(const BettingState& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(
      Chips{1},
      static_cast<Chips>(std::max(Chips{1}, Pot(state)) * size));
}

bool IsLegalAction(const BettingState& state, const GameAction& action) {
  const int player = state.player_to_act;
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

void AddAction(ActionMenu& menu,
               const BettingState& state,
               ActionKind kind,
               Chips amount) {
  if (menu.count >= kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  const GameAction action{kind, amount};
  if (!IsLegalAction(state, action)) {
    throw std::logic_error("Generated an illegal poker action");
  }
  menu.actions[static_cast<size_t>(menu.count)] = action;
  ++menu.count;
}

BettingState ApplyActionUnchecked(const BettingState& state,
                                  const GameAction& action) {
  BettingState child = state;
  assert(IsPlayer(child.player_to_act));
  const int player = child.player_to_act;
  const int opponent = Opponent(player);
  const Chips to_call_before = ToCall(child, player);
  const Chips highest_before = HighestStreetCommitment(child);
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
  if (aggressive) {
    const Chips raise_size =
        child.street_committed[player] - highest_before;
    if (raise_size >= child.last_full_raise) {
      child.last_full_raise = raise_size;
    }
  }
  if (action.kind != ActionKind::kFold) {
    if (aggressive) {
      child.pending_action_mask = PlayerBit(opponent);
    } else {
      child.pending_action_mask &=
          static_cast<uint8_t>(~PlayerBit(player));
    }
  }
  if (IsBettingRoundOver(child)) {
    RefundUnmatchedCommitment(child);
    child.player_to_act = -1;
  }

  assert(IsValidBettingState(child));
  return child;
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

bool IsTerminal(const BettingState& state, const BoardRunout& board) {
  return state.folded_player >= 0 || HandOver(state, board);
}

ActionMenu LegalActions(const BettingState& state,
                        absl::Span<const double> bet_sizes) {
  const int player = state.player_to_act;
  if (!IsPlayer(player) || state.folded_player >= 0 ||
      state.stack[player] <= 0) {
    throw std::logic_error("LegalActions requires a decision state");
  }

  ActionMenu menu;
  const Chips stack = state.stack[player];
  const Chips outstanding_call = ToCall(state, player);
  if (outstanding_call > 0) {
    AddAction(menu, state, ActionKind::kFold, 0);
    AddAction(menu, state, ActionKind::kCall,
              std::min(outstanding_call, stack));
  } else {
    AddAction(menu, state, ActionKind::kCheck, 0);
  }

  ActionKind sized_kind = ActionKind::kBet;
  if (outstanding_call > 0) {
    sized_kind = ActionKind::kRaise;
  }
  absl::InlinedVector<GameAction, kMaxActionsPerNode> sized_actions;
  for (double bet_size : bet_sizes) {
    const Chips bet = ConcreteBetAmount(state, bet_size);
    const Chips amount = outstanding_call + bet;
    if (amount < stack) {
      sized_actions.push_back({sized_kind, amount});
    }
  }
  std::sort(sized_actions.begin(), sized_actions.end(),
            [](const GameAction& left, const GameAction& right) {
              return left.amount < right.amount;
            });
  const auto unique_end =
      std::unique(sized_actions.begin(), sized_actions.end());
  sized_actions.erase(unique_end, sized_actions.end());

  for (const GameAction& action : sized_actions) {
    AddAction(menu, state, action.kind, action.amount);
  }

  if (outstanding_call == 0 || stack > outstanding_call) {
    AddAction(menu, state, ActionKind::kAllIn, stack);
  }
  return menu;
}

BettingState ApplyAction(const BettingState& state,
                         const GameAction& action) {
  if (!IsLegalAction(state, action)) {
    throw std::invalid_argument("illegal poker action");
  }
  return ApplyActionUnchecked(state, action);
}

ExactPublicState ApplyChance(const ExactPublicState& state,
                             absl::Span<const CardId> cards,
                             const BettingRules& rules) {
  if (rules.minimum_bet <= 0) {
    throw std::invalid_argument("minimum bet must be positive");
  }
  if (IsTerminal(state.betting, state.board) ||
      !IsBettingRoundOver(state.betting) ||
      state.betting.player_to_act != -1) {
    throw std::invalid_argument("State is not a chance node");
  }
  if (cards.size() !=
      static_cast<size_t>(CardsForNextStreet(state.betting.street))) {
    throw std::invalid_argument("Incorrect number of chance cards");
  }

  ExactPublicState child = state;
  AdvanceStreet(child, cards, rules);
  return child;
}

double GetUtility(const ExactPublicState& state,
                  ComboId player_a_hand,
                  ComboId player_b_hand) {
  static const HandEvaluator evaluator;
  const double a_committed = state.betting.total_committed[0];

  if (state.betting.folded_player >= 0) {
    if (state.betting.folded_player == 0) {
      return -a_committed;
    }
    return Pot(state.betting) - a_committed;
  }

  if (state.board.count() + 2 < 5) {
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
