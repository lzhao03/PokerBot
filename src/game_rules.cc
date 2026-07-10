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

Chips CommitChips(BettingState& state, int player, Chips requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action commitment delta must be positive");
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

Chips ConcreteBetAmount(const BettingState& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(
      Chips{1},
      static_cast<Chips>(std::max(Chips{1}, Pot(state)) * size));
}

struct ActionLimits {
  Chips current = 0;
  Chips highest = 0;
  Chips call_target = 0;
  Chips maximum_target = 0;
  Chips minimum_aggressive_target = 0;
  bool wager_open = false;
};

ActionLimits LimitsFor(const BettingState& state, int player) {
  ActionLimits limits;
  limits.current = state.street_committed[player];
  limits.highest = HighestStreetCommitment(state);
  limits.call_target =
      std::min(limits.highest, limits.current + state.stack[player]);
  limits.maximum_target =
      limits.current + MaxContestableAdditional(state, player);
  limits.wager_open = limits.highest > 0;
  limits.minimum_aggressive_target =
      limits.wager_open ? limits.highest + state.last_full_raise
                        : limits.current + state.last_full_raise;
  return limits;
}

bool IsLegalAction(const BettingState& state, const GameAction& action) {
  const int player = state.player_to_act;
  if (!IsPlayer(player) || state.folded_player >= 0 ||
      state.stack[player] <= 0) {
    return false;
  }

  const ActionLimits limits = LimitsFor(state, player);
  const Chips to_call = limits.highest - limits.current;
  const Chips target = action.target_street_commitment;
  switch (action.kind) {
    case ActionKind::kFold:
      return to_call > 0 && target == 0;
    case ActionKind::kCheck:
      return to_call == 0 && target == 0;
    case ActionKind::kCall:
      return to_call > 0 && target == limits.call_target;
    case ActionKind::kBet:
      return !limits.wager_open &&
             target >= limits.minimum_aggressive_target &&
             target < limits.maximum_target;
    case ActionKind::kRaise:
      return limits.wager_open &&
             target >= limits.minimum_aggressive_target &&
             target < limits.maximum_target;
    case ActionKind::kAllIn:
      return limits.maximum_target > limits.call_target &&
             target == limits.maximum_target;
    case ActionKind::kNoAction:
      return false;
  }
}

void AddAction(ActionMenu& menu,
               const BettingState& state,
               ActionKind kind,
               Chips target_street_commitment) {
  if (menu.count >= kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  const GameAction action{kind, target_street_commitment};
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
  const Chips highest_before = HighestStreetCommitment(child);
  const Chips current = child.street_committed[player];
  const Chips delta = action.target_street_commitment - current;

  switch (action.kind) {
    case ActionKind::kFold:
      child.folded_player = player;
      child.player_to_act = -1;
      break;
    case ActionKind::kCheck:
      child.player_to_act = opponent;
      break;
    case ActionKind::kCall:
    case ActionKind::kBet:
    case ActionKind::kRaise:
    case ActionKind::kAllIn:
      CommitChips(child, player, delta);
      child.player_to_act = opponent;
      break;
    case ActionKind::kNoAction:
      break;
  }

  const bool aggressive =
      action.target_street_commitment > highest_before;
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

ExactPublicState MakeInitialState(
    const BettingRules& rules,
    std::array<Chips, kPlayerCount> stacks,
    std::array<Chips, kPlayerCount> blinds) {
  if (rules.minimum_bet <= 0) {
    throw std::invalid_argument("minimum bet must be positive");
  }
  for (size_t player = 0; player < kPlayerCount; ++player) {
    if (blinds[player] < 0 || stacks[player] < blinds[player]) {
      throw std::invalid_argument("stacks must cover posted blinds");
    }
  }

  BettingState betting;
  for (size_t player = 0; player < kPlayerCount; ++player) {
    betting.stack[player] = stacks[player] - blinds[player];
  }
  betting.total_committed = blinds;
  betting.street_committed = blinds;
  betting.last_full_raise = rules.minimum_bet;
  return ExactPublicState{betting, BoardRunout::Preflop()};
}

bool IsBettingRoundOver(const BettingState& state) noexcept {
  if (state.folded_player >= 0) {
    return true;
  }
  const bool commitments_match =
      state.street_committed[0] == state.street_committed[1];
  if (state.pending_action_mask == 0 && commitments_match) {
    return true;
  }
  if (!AnyPlayerAllIn(state)) {
    return false;
  }
  if (state.stack[0] == 0 && state.stack[1] == 0) {
    return true;
  }
  const int live_player = state.stack[0] > 0 ? 0 : 1;
  return ToCall(state, live_player) == 0;
}

bool IsTerminal(const ExactPublicState& state) {
  if (state.betting.folded_player >= 0) {
    return true;
  }
  return state.betting.street == StreetKind::kRiver &&
         state.board.count() == kMaxBoardCards &&
         IsBettingRoundOver(state.betting);
}

ActionMenu LegalActions(const BettingState& state,
                        absl::Span<const double> bet_sizes) {
  const int player = state.player_to_act;
  if (!IsPlayer(player) || state.folded_player >= 0 ||
      state.stack[player] <= 0) {
    throw std::logic_error("LegalActions requires a decision state");
  }

  ActionMenu menu;
  const ActionLimits limits = LimitsFor(state, player);
  const Chips outstanding_call = limits.highest - limits.current;
  if (outstanding_call > 0) {
    AddAction(menu, state, ActionKind::kFold, 0);
    AddAction(menu, state, ActionKind::kCall, limits.call_target);
  } else {
    AddAction(menu, state, ActionKind::kCheck, 0);
  }

  const ActionKind sized_kind =
      limits.wager_open ? ActionKind::kRaise : ActionKind::kBet;
  absl::InlinedVector<GameAction, kMaxActionsPerNode> sized_actions;
  for (double bet_size : bet_sizes) {
    const Chips bet = ConcreteBetAmount(state, bet_size);
    const Chips target = limits.highest + bet;
    if (target >= limits.minimum_aggressive_target &&
        target < limits.maximum_target) {
      sized_actions.push_back({sized_kind, target});
    }
  }
  std::sort(sized_actions.begin(), sized_actions.end(),
            [](const GameAction& left, const GameAction& right) {
              return left.target_street_commitment <
                     right.target_street_commitment;
            });
  const auto unique_end =
      std::unique(sized_actions.begin(), sized_actions.end());
  sized_actions.erase(unique_end, sized_actions.end());

  for (const GameAction& action : sized_actions) {
    AddAction(menu, state, action.kind,
              action.target_street_commitment);
  }

  if (limits.maximum_target > limits.call_target) {
    AddAction(menu, state, ActionKind::kAllIn, limits.maximum_target);
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
  if (IsTerminal(state) ||
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

double TerminalUtility(const ExactPublicState& state,
                       ComboId player0_hand,
                       ComboId player1_hand) {
  if (!IsTerminal(state)) {
    throw std::invalid_argument("TerminalUtility requires a terminal state");
  }

  static const HandEvaluator evaluator;
  const double player0_committed = state.betting.total_committed[0];

  if (state.betting.folded_player >= 0) {
    if (state.betting.folded_player == 0) {
      return -player0_committed;
    }
    return Pot(state.betting) - player0_committed;
  }

  const int comparison =
      evaluator.compare_hands(player0_hand, player1_hand, state.board);
  if (comparison > 0) {
    return Pot(state.betting) - player0_committed;
  }
  if (comparison < 0) {
    return -player0_committed;
  }
  return (Pot(state.betting) / 2.0) - player0_committed;
}

}  // namespace poker
