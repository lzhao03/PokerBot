#include "src/game_tree.h"

#include <algorithm>
#include <stdexcept>

namespace poker {

namespace {

constexpr int kActionKeyMultiplier = 1000000;

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

int BoardCardCount(const GameState& state) {
  return static_cast<int>(state.board_cards.size());
}

int BoardCardCount(const CompactPublicState& state) {
  return state.board_count;
}

int HistorySize(const GameState& state) {
  return static_cast<int>(state.history.size());
}

int HistorySize(const CompactPublicState& state) {
  return state.history_size;
}

GameAction LastAction(const GameState& state) {
  return state.history.back();
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
int ConcreteBetAmount(const State& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(1, static_cast<int>(std::max(1, state.pot) * size));
}

int StreetBetSizesSize(const SolverConfig& config, StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return config.preflop_bet_sizes.size();
    case StreetKind::kFlop:
      return config.flop_bet_sizes.size();
    case StreetKind::kTurn:
      return config.turn_bet_sizes.size();
    case StreetKind::kRiver:
      return config.river_bet_sizes.size();
  }
}

double BetSizeForStreet(const SolverConfig& config,
                        StreetKind street,
                        int index) {
  if (StreetBetSizesSize(config, street) == 0) {
    return config.bet_sizes[index];
  }

  switch (street) {
    case StreetKind::kPreflop:
      return config.preflop_bet_sizes[index];
    case StreetKind::kFlop:
      return config.flop_bet_sizes[index];
    case StreetKind::kTurn:
      return config.turn_bet_sizes[index];
    case StreetKind::kRiver:
      return config.river_bet_sizes[index];
  }
}

int BetSizesSize(const SolverConfig& config, StreetKind street) {
  const int street_sizes = StreetBetSizesSize(config, street);
  return street_sizes > 0 ? street_sizes
                          : static_cast<int>(config.bet_sizes.size());
}

void AddAction(std::array<GameAction, GameTree::kMaxActionsPerNode>& actions,
               uint8_t& action_count,
               ActionKind kind,
               int amount) {
  if (action_count >= GameTree::kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  actions[static_cast<size_t>(action_count)] = {kind, amount, -1};
  ++action_count;
}

void AddActionIfMissing(
    std::array<GameAction, GameTree::kMaxActionsPerNode>& actions,
    uint8_t& action_count,
    ActionKind kind,
    int amount) {
  for (uint8_t i = 0; i < action_count; ++i) {
    const GameAction& action = actions[static_cast<size_t>(i)];
    if (action.kind == kind && action.amount == amount) {
      return;
    }
  }
  AddAction(actions, action_count, kind, amount);
}

template <typename State>
bool IsBettingRoundOver(const State& state) {
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
  return BoardComplete(state) && IsBettingRoundOver(state);
}

template <typename State>
bool IsTerminal(const State& state) {
  return state.folded_player >= 0 || IsHandOver(state);
}

template <typename State>
int PlayerToAct(const State& state) {
  if (IsTerminal(state)) {
    return -1;
  }
  if (IsBettingRoundOver(state)) {
    return -1;
  }
  if (IsPlayer(state.player_to_act)) {
    return state.player_to_act;
  }
  return FirstPlayerForStreet(state);
}

template <typename State>
uint8_t LegalActionsForState(
    const SolverConfig& config,
    const State& state,
    std::array<GameAction, GameTree::kMaxActionsPerNode>& actions) {
  uint8_t action_count = 0;
  if (IsTerminal(state)) {
    return action_count;
  }

  const int player = PlayerToAct(state);
  if (!IsPlayer(player)) {
    return action_count;
  }

  const int stack = StackForState(state, player);
  if (stack <= 0) {
    return action_count;
  }

  const int to_call = OutstandingToCall(state, player);
  if (to_call > 0) {
    AddAction(actions, action_count, ActionKind::kFold, 0);
    AddAction(actions, action_count, ActionKind::kCall,
              std::min(to_call, stack));

    for (int i = 0; i < BetSizesSize(config, state.street); ++i) {
      const int raise_amount =
          to_call + ConcreteBetAmount(
                        state, BetSizeForStreet(config, state.street, i));
      if (raise_amount < stack) {
        AddActionIfMissing(actions, action_count, ActionKind::kRaise,
                           raise_amount);
      }
    }
    if (stack > to_call) {
      AddAction(actions, action_count, ActionKind::kAllIn, stack);
    }
  } else {
    AddAction(actions, action_count, ActionKind::kCheck, 0);

    for (int i = 0; i < BetSizesSize(config, state.street); ++i) {
      const int bet_amount =
          ConcreteBetAmount(state, BetSizeForStreet(config, state.street, i));
      if (bet_amount < stack) {
        AddActionIfMissing(actions, action_count, ActionKind::kBet,
                           bet_amount);
      }
    }
    AddAction(actions, action_count, ActionKind::kAllIn, stack);
  }

  return action_count;
}

template <typename State>
std::vector<GameAction> LegalActionsForState(const SolverConfig& config,
                                             const State& state) {
  std::array<GameAction, GameTree::kMaxActionsPerNode> action_table = {};
  const uint8_t action_count =
      LegalActionsForState(config, state, action_table);
  std::vector<GameAction> actions;
  actions.reserve(action_count);
  for (uint8_t i = 0; i < action_count; ++i) {
    actions.push_back(action_table[static_cast<size_t>(i)]);
  }
  return actions;
}

void AppendStateHistory(GameState& state, const GameAction& action) {
  state.history.push_back(action);
}

void AppendStateHistory(CompactPublicState& state, const GameAction& action) {
  AppendHistoryAction(state, action);
}

void ResetStateHistory(GameState& state) {
  state.history.clear();
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
    player = PlayerToAct(new_state);
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

GameTree::GameTree(const SolverConfig& config) : config_(config) {}

int GameTree::action_key(const GameAction& action) {
  if (action.amount < 0 || action.amount >= kActionKeyMultiplier) {
    throw std::invalid_argument("Action amount is outside action-key range");
  }
  return static_cast<int>(action.kind) * kActionKeyMultiplier + action.amount;
}

std::vector<GameAction> GameTree::get_legal_actions(
    const GameState& state) const {
  return LegalActionsForState(config_, state);
}

std::vector<GameAction> GameTree::get_legal_actions(
    const CompactPublicState& state) const {
  return LegalActionsForState(config_, state);
}

uint8_t GameTree::get_legal_actions(
    const CompactPublicState& state,
    std::array<GameAction, kMaxActionsPerNode>& actions) const {
  return LegalActionsForState(config_, state, actions);
}

GameState GameTree::apply_action(const GameState& state,
                                 const GameAction& action) const {
  return ApplyActionForState(state, action);
}

CompactPublicState GameTree::apply_action(
    const CompactPublicState& state,
    const GameAction& action) const {
  return ApplyActionForState(state, action);
}

GameState GameTree::apply_chance(const GameState& state,
                                 absl::Span<const CardId> cards) const {
  if (is_terminal(state) || get_player_to_act(state) != -1) {
    throw std::invalid_argument("State is not a chance node");
  }

  GameState new_state = state;
  AdvanceStreet(new_state, cards);
  return new_state;
}

CompactPublicState GameTree::apply_chance(
    const CompactPublicState& state,
    absl::Span<const CardId> cards) const {
  if (is_terminal(state) || get_player_to_act(state) != -1) {
    throw std::invalid_argument("State is not a chance node");
  }

  CompactPublicState child = state;
  AdvanceStreet(child, cards);
  return child;
}

double GameTree::get_utility(const GameState& state,
                             ComboId player_a_hand,
                             ComboId player_b_hand) const {
  return UtilityForState(state, player_a_hand, player_b_hand,
                         hand_evaluator_);
}

double GameTree::get_utility(const CompactPublicState& state,
                             ComboId player_a_hand,
                             ComboId player_b_hand) const {
  return UtilityForState(state, player_a_hand, player_b_hand,
                         hand_evaluator_);
}

bool GameTree::is_terminal(const GameState& state) const {
  return IsTerminal(state);
}

bool GameTree::is_terminal(const CompactPublicState& state) const {
  return IsTerminal(state);
}

int GameTree::get_player_to_act(const GameState& state) const {
  return PlayerToAct(state);
}

int GameTree::get_player_to_act(const CompactPublicState& state) const {
  return PlayerToAct(state);
}

bool GameTree::is_betting_round_over(const GameState& state) const {
  return IsBettingRoundOver(state);
}

bool GameTree::is_betting_round_over(
    const CompactPublicState& state) const {
  return IsBettingRoundOver(state);
}

bool GameTree::is_hand_over(const GameState& state) const {
  return IsHandOver(state);
}

bool GameTree::is_hand_over(const CompactPublicState& state) const {
  return IsHandOver(state);
}

}  // namespace poker
