#include "src/game_rules.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <random>
#include <stdexcept>

#include "absl/container/inlined_vector.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

std::array<ComboInfo, kComboCount> BuildComboTable() {
  std::array<ComboInfo, kComboCount> combos;
  int combo = 0;
  for (int first = 0; first < kDeckCardCount; ++first) {
    for (int second = first + 1; second < kDeckCardCount; ++second) {
      combos[combo++] = {
          static_cast<CardId>(first),
          static_cast<CardId>(second),
          CardBit(static_cast<CardId>(first)) |
              CardBit(static_cast<CardId>(second)),
      };
    }
  }
  return combos;
}

const std::array<ComboInfo, kComboCount>& ComboTable() {
  static const std::array<ComboInfo, kComboCount> table = BuildComboTable();
  return table;
}

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

void DealNextStreet(BoardRunout& board,
                    StreetKind street,
                    absl::Span<const CardId> cards) {
  switch (street) {
    case StreetKind::kPreflop:
      board.deal_flop(cards);
      break;
    case StreetKind::kFlop:
      board.deal_turn(cards[0]);
      break;
    case StreetKind::kTurn:
      board.deal_river(cards[0]);
      break;
    case StreetKind::kRiver:
      break;
  }
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

void AddAction(SolverActions& actions,
               const BettingState& state,
               ActionKind kind,
               Chips target_street_commitment) {
  const GameAction action{kind, target_street_commitment};
  if (!IsLegalAction(state, action)) {
    throw std::logic_error("Generated an illegal poker action");
  }
  actions.push_back(action);
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

const ComboInfo& GetComboInfo(ComboId combo_id) {
  if (combo_id >= kComboCount) {
    throw std::invalid_argument("Invalid combo id");
  }
  return ComboTable()[combo_id];
}

CardMask ComboMask(ComboId combo_id) {
  return GetComboInfo(combo_id).mask;
}

std::optional<ComboId> MaybeCardsToComboId(CardId first, CardId second) {
  if (first >= kDeckCardCount || second >= kDeckCardCount || first == second) {
    return std::nullopt;
  }
  if (second < first) {
    std::swap(first, second);
  }

  ComboId combo = 0;
  for (int card = 0; card < first; ++card) {
    combo += static_cast<ComboId>(kDeckCardCount - card - 1);
  }
  combo += static_cast<ComboId>(second - first - 1);
  return combo;
}

ComboId CardsToComboId(CardId first, CardId second) {
  const std::optional<ComboId> combo = MaybeCardsToComboId(first, second);
  if (!combo.has_value()) {
    throw std::invalid_argument("Invalid exact two-card combo");
  }
  return *combo;
}

int CardsForNextStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 3;
    case StreetKind::kFlop:
    case StreetKind::kTurn:
      return 1;
    case StreetKind::kRiver:
      return 0;
  }
}

int BoardCardsForStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 0;
    case StreetKind::kFlop:
      return 3;
    case StreetKind::kTurn:
      return 4;
    case StreetKind::kRiver:
      return 5;
  }
}

absl::InlinedVector<CardId, 5> SampleStreetCards(
    StreetKind street,
    const BoardRunout& board,
    CardMask known_private_cards,
    std::mt19937& rng) {
  const int open_slots = std::max(0, kMaxBoardCards - board.count());
  const int count = std::min(CardsForNextStreet(street), open_slots);
  if (count <= 0) {
    return {};
  }

  const CardMask blocked = known_private_cards | board.mask();
  if (count == 1) {
    std::uniform_int_distribution<int> card_dist(0, kDeckCardCount - 1);
    for (int attempt = 0; attempt < kDeckCardCount; ++attempt) {
      const CardId candidate = static_cast<CardId>(card_dist(rng));
      if ((blocked & CardBit(candidate)) == 0) {
        return {candidate};
      }
    }
  }

  std::array<CardId, kDeckCardCount> candidates = {};
  int candidate_count = 0;
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    const CardId candidate = static_cast<CardId>(card_id);
    if ((blocked & CardBit(candidate)) == 0) {
      candidates[candidate_count++] = candidate;
    }
  }
  if (candidate_count < count) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  absl::InlinedVector<CardId, 5> sampled;
  sampled.reserve(count);
  for (int i = 0; i < count; ++i) {
    std::uniform_int_distribution<int> card_dist(i, candidate_count - 1);
    const int chosen = card_dist(rng);
    std::swap(candidates[i], candidates[chosen]);
    sampled.push_back(candidates[i]);
  }
  return sampled;
}

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

SolverActions GetSolverActions(const SolverConfig& config,
                               const BettingState& state) {
  const int player = state.player_to_act;
  if (!IsPlayer(player) || state.folded_player >= 0 ||
      state.stack[player] <= 0) {
    throw std::logic_error("GetSolverActions requires a decision state");
  }

  SolverActions actions;
  const ActionLimits limits = LimitsFor(state, player);
  const Chips outstanding_call = limits.highest - limits.current;
  if (outstanding_call > 0) {
    AddAction(actions, state, ActionKind::kFold, 0);
    AddAction(actions, state, ActionKind::kCall, limits.call_target);
  } else {
    AddAction(actions, state, ActionKind::kCheck, 0);
  }

  const ActionKind sized_kind =
      limits.wager_open ? ActionKind::kRaise : ActionKind::kBet;
  SolverActions sized_actions;
  const auto& bet_sizes =
      config.bet_sizes[static_cast<size_t>(state.street)];
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
    AddAction(actions, state, action.kind,
              action.target_street_commitment);
  }

  if (limits.maximum_target > limits.call_target) {
    AddAction(actions, state, ActionKind::kAllIn, limits.maximum_target);
  }
  return actions;
}

BettingState ApplyAction(const BettingState& state,
                         const GameAction& action) {
  if (!IsLegalAction(state, action)) {
    throw std::invalid_argument("illegal poker action");
  }
  return ApplyActionUnchecked(state, action);
}

BettingState AdvanceBettingStreet(const BettingState& state,
                                  const BettingRules& rules) {
  if (rules.minimum_bet <= 0) {
    throw std::invalid_argument("minimum bet must be positive");
  }
  if (!IsBettingRoundOver(state) || state.player_to_act != -1 ||
      state.street == StreetKind::kRiver) {
    throw std::invalid_argument("betting state is not a chance node");
  }

  BettingState child = state;
  child.street = static_cast<StreetKind>(static_cast<int>(state.street) + 1);
  child.street_committed = {0, 0};
  child.last_full_raise = rules.minimum_bet;
  child.pending_action_mask = kAllPlayersMask;
  child.player_to_act = FirstPlayerForStreet(child.street);
  if (IsBettingRoundOver(child)) {
    child.player_to_act = -1;
  }
  assert(IsValidBettingState(child));
  return child;
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

  ExactPublicState child{AdvanceBettingStreet(state.betting, rules),
                         state.board};
  DealNextStreet(child.board, state.betting.street, cards);
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
