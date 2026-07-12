#include "src/poker.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

std::array<ComboInfo, kComboCount> BuildComboTable() {
  std::array<ComboInfo, kComboCount> combos;
  size_t combo = 0;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      combos[combo++] = {
          kDeck[first],
          kDeck[second],
          CardBit(kDeck[first]) | CardBit(kDeck[second]),
      };
    }
  }
  return combos;
}

const std::array<ComboInfo, kComboCount>& ComboTable() {
  static const std::array<ComboInfo, kComboCount> table = BuildComboTable();
  return table;
}

Player FirstPlayerForStreet(StreetKind street) {
  return street == StreetKind::Preflop ? Player::A : Player::B;
}

Chips CommitChips(BettingData& state, Player player, Chips requested) {
  assert(requested > 0);
  const size_t index = Index(player);
  const Chips committed = std::min(requested, state.stack[index]);
  state.stack[index] -= committed;
  state.total_committed[index] += committed;
  state.street_committed[index] += committed;
  return committed;
}

void RefundUnmatchedCommitment(BettingData& state) {
  if (state.street_committed[0] == state.street_committed[1]) {
    return;
  }
  const Player player = state.street_committed[0] > state.street_committed[1]
                            ? Player::A
                            : Player::B;
  const size_t index = Index(player);
  const size_t opponent = Index(Opponent(player));
  const Chips excess =
      state.street_committed[index] - state.street_committed[opponent];
  state.street_committed[index] -= excess;
  state.total_committed[index] -= excess;
  state.stack[index] += excess;
}

bool IsBettingRoundOver(const BettingData& state) noexcept {
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
  const Player live_player =
      state.stack[0] > 0 ? Player::A : Player::B;
  return ToCall(state, live_player) == 0;
}

BettingState ApplyActionUnchecked(const DecisionState& state,
                                  const GameAction& action) {
  BettingData child = state.data;
  const Player player = state.actor;
  const Player opponent = Opponent(player);
  const size_t player_index = Index(player);
  const Chips highest_before = CurrentWager(child);
  const Chips current = child.street_committed[player_index];
  const Chips delta = action.target_street_commitment - current;

  switch (action.kind) {
    case ActionKind::Fold:
      return FoldTerminalState{child, player};
    case ActionKind::Check:
      break;
    case ActionKind::Call:
    case ActionKind::Bet:
    case ActionKind::Raise:
    case ActionKind::AllIn:
      CommitChips(child, player, delta);
      break;
  }

  const bool aggressive =
      action.target_street_commitment > highest_before;
  if (aggressive) {
    const Chips raise_size =
        child.street_committed[player_index] - highest_before;
    if (raise_size >= child.last_full_raise) {
      child.last_full_raise = raise_size;
    }
  }
  if (aggressive) {
    child.pending_action_mask = PlayerBit(opponent);
  } else {
    child.pending_action_mask &=
        static_cast<uint8_t>(~PlayerBit(player));
  }
  if (IsBettingRoundOver(child)) {
    RefundUnmatchedCommitment(child);
    assert(IsValidBettingData(child));
    return child.street == StreetKind::River
               ? BettingState(ShowdownState{child})
               : BettingState(ChanceState{child});
  }

  assert(IsValidBettingData(child));
  return DecisionState{child, opponent};
}

}  // namespace

FlopBoard DealFlop(const PreflopBoard&,
                   std::array<Card, 3> cards) noexcept {
  CardMask mask = 0;
  for (Card card : cards) {
    assert((mask & CardBit(card)) == 0);
    mask |= CardBit(card);
  }
  std::sort(cards.begin(), cards.end());
  return FlopBoard(cards, mask);
}

TurnBoard DealTurn(const FlopBoard& board, Card card) noexcept {
  assert((board.mask() & CardBit(card)) == 0);
  std::array<Card, 5> cards = {};
  std::copy(board.cards().begin(), board.cards().end(), cards.begin());
  cards[3] = card;
  return TurnBoard(cards, board.mask() | CardBit(card));
}

RiverBoard DealRiver(const TurnBoard& board, Card card) noexcept {
  assert((board.mask() & CardBit(card)) == 0);
  std::array<Card, 5> cards = {};
  std::copy(board.cards().begin(), board.cards().end(), cards.begin());
  cards[4] = card;
  return RiverBoard(cards, board.mask() | CardBit(card));
}

absl::StatusOr<FlopBoard> MakeFlop(std::array<Card, 3> cards) {
  CardMask mask = 0;
  for (Card card : cards) {
    if ((mask & CardBit(card)) != 0) {
      return absl::InvalidArgumentError("duplicate flop card");
    }
    mask |= CardBit(card);
  }
  std::sort(cards.begin(), cards.end());
  return FlopBoard(cards, mask);
}

absl::StatusOr<TurnBoard> MakeTurn(const FlopBoard& board, Card card) {
  if ((board.mask() & CardBit(card)) != 0) {
    return absl::InvalidArgumentError("duplicate turn card");
  }
  return DealTurn(board, card);
}

absl::StatusOr<RiverBoard> MakeRiver(const TurnBoard& board, Card card) {
  if ((board.mask() & CardBit(card)) != 0) {
    return absl::InvalidArgumentError("duplicate river card");
  }
  return DealRiver(board, card);
}

absl::Span<const Card> BoardCards(const Board& board) noexcept {
  return std::visit([](const auto& street) -> absl::Span<const Card> {
    using Street = std::decay_t<decltype(street)>;
    if constexpr (std::is_same_v<Street, PreflopBoard>) {
      return {};
    } else {
      return street.cards();
    }
  }, board);
}

CardMask BoardMask(const Board& board) noexcept {
  return std::visit([](const auto& street) -> CardMask {
    using Street = std::decay_t<decltype(street)>;
    if constexpr (std::is_same_v<Street, PreflopBoard>) {
      return 0;
    } else {
      return street.mask();
    }
  }, board);
}

uint8_t BoardCount(const Board& board) noexcept {
  return static_cast<uint8_t>(BoardCards(board).size());
}

bool BoardContains(const Board& board, Card card) noexcept {
  return (BoardMask(board) & CardBit(card)) != 0;
}

const ComboInfo& GetComboInfo(ComboId combo_id) {
  return ComboTable()[combo_id.index()];
}

CardMask ComboMask(ComboId combo_id) {
  return GetComboInfo(combo_id).mask;
}

std::optional<ComboId> MaybeCardsToComboId(Card first, Card second) {
  if (first == second) {
    return std::nullopt;
  }
  if (second < first) {
    std::swap(first, second);
  }

  uint16_t combo = 0;
  for (size_t card = 0; card < first.index(); ++card) {
    combo += static_cast<uint16_t>(kDeckCardCount - card - 1);
  }
  combo += static_cast<uint16_t>(second.index() - first.index() - 1);
  return ComboId(combo);
}

ComboId CardsToComboId(Card first, Card second) noexcept {
  const std::optional<ComboId> combo = MaybeCardsToComboId(first, second);
  assert(combo.has_value());
  return *combo;
}

absl::StatusOr<HoleCards> MakeHoleCards(Card first, Card second) {
  const std::optional<ComboId> combo = MaybeCardsToComboId(first, second);
  if (!combo.has_value()) {
    return absl::InvalidArgumentError("hole cards must be distinct");
  }
  return HoleCards(*combo);
}

int CardsForNextStreet(StreetKind street) {
  switch (street) {
    case StreetKind::Preflop:
      return 3;
    case StreetKind::Flop:
    case StreetKind::Turn:
      return 1;
    case StreetKind::River:
      return 0;
  }
}

int BoardCardsForStreet(StreetKind street) {
  switch (street) {
    case StreetKind::Preflop:
      return 0;
    case StreetKind::Flop:
      return 3;
    case StreetKind::Turn:
      return 4;
    case StreetKind::River:
      return 5;
  }
}

absl::StatusOr<absl::InlinedVector<Card, 5>> SampleStreetCards(
    StreetKind street,
    const Board& board,
    CardMask known_private_cards,
    std::mt19937& rng) {
  const int open_slots = std::max(0, kMaxBoardCards - BoardCount(board));
  const int count = std::min(CardsForNextStreet(street), open_slots);
  if (count <= 0) {
    return absl::InlinedVector<Card, 5>{};
  }

  const CardMask blocked = known_private_cards | BoardMask(board);
  if (count == 1) {
    std::uniform_int_distribution<int> card_dist(0, kDeckCardCount - 1);
    for (int attempt = 0; attempt < kDeckCardCount; ++attempt) {
      const Card candidate = kDeck[static_cast<size_t>(card_dist(rng))];
      if ((blocked & CardBit(candidate)) == 0) {
        return absl::InlinedVector<Card, 5>{candidate};
      }
    }
  }

  std::array<Card, kDeckCardCount> candidates = {};
  int candidate_count = 0;
  for (Card candidate : kDeck) {
    if ((blocked & CardBit(candidate)) == 0) {
      candidates[static_cast<size_t>(candidate_count++)] = candidate;
    }
  }
  if (candidate_count < count) {
    return absl::InvalidArgumentError("not enough unblocked cards");
  }

  absl::InlinedVector<Card, 5> sampled;
  sampled.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    std::uniform_int_distribution<int> card_dist(i, candidate_count - 1);
    const int chosen = card_dist(rng);
    std::swap(candidates[static_cast<size_t>(i)],
              candidates[static_cast<size_t>(chosen)]);
    sampled.push_back(candidates[static_cast<size_t>(i)]);
  }
  return sampled;
}

ExactPublicState MakeInitialState(
    const BettingRules& rules,
  std::array<Chips, kPlayerCount> stacks,
  std::array<Chips, kPlayerCount> blinds) {
  assert(rules.minimum_bet > 0);
  for (size_t player = 0; player < kPlayerCount; ++player) {
    assert(blinds[player] >= 0 && stacks[player] >= blinds[player]);
  }

  BettingData betting;
  for (size_t player = 0; player < kPlayerCount; ++player) {
    betting.stack[player] = stacks[player] - blinds[player];
  }
  betting.total_committed = blinds;
  betting.street_committed = blinds;
  betting.last_full_raise = rules.minimum_bet;
  return ExactPublicState{DecisionState{betting, Player::A},
                          PreflopBoard{}};
}

bool IsTerminal(const ExactPublicState& state) {
  if (std::holds_alternative<FoldTerminalState>(state.betting)) {
    return true;
  }
  return std::holds_alternative<ShowdownState>(state.betting) &&
         BoardCount(state.board) == kMaxBoardCards;
}

bool IsLegalAction(const DecisionState& state,
                   const GameAction& action) noexcept {
  const BettingData& data = state.data;
  const size_t player = Index(state.actor);
  if (data.stack[player] <= 0) {
    return false;
  }

  const Chips current_to = data.street_committed[player];
  const Chips highest_to = CurrentWager(data);
  const Chips to_call = highest_to - current_to;
  const Chips call_to =
      std::min(highest_to, current_to + data.stack[player]);
  const Chips all_in_to =
      current_to + MaxContestableAdditional(data, state.actor);
  const Chips min_full_raise_to =
      (highest_to > 0 ? highest_to : current_to) + data.last_full_raise;
  const Chips target = action.target_street_commitment;
  switch (action.kind) {
    case ActionKind::Fold:
      return to_call > 0 && target == 0;
    case ActionKind::Check:
      return to_call == 0 && target == 0;
    case ActionKind::Call:
      return to_call > 0 && target == call_to;
    case ActionKind::Bet:
      return highest_to == 0 && target >= min_full_raise_to &&
             target < all_in_to;
    case ActionKind::Raise:
      return highest_to > 0 && target >= min_full_raise_to &&
             target < all_in_to;
    case ActionKind::AllIn:
      return all_in_to > call_to && target == all_in_to;
  }
}

absl::StatusOr<BettingState> ApplyAction(const DecisionState& state,
                                         const GameAction& action) {
  if (!IsLegalAction(state, action)) {
    return absl::InvalidArgumentError("illegal poker action");
  }
  return ApplyActionUnchecked(state, action);
}

BettingState AdvanceBettingStreet(const ChanceState& state,
                                  const BettingRules& rules) {
  assert(rules.minimum_bet > 0);
  assert(state.data.street != StreetKind::River);

  BettingData child = state.data;
  child.street = static_cast<StreetKind>(
      static_cast<int>(state.data.street) + 1);
  child.street_committed = {0, 0};
  child.last_full_raise = rules.minimum_bet;
  child.pending_action_mask = kAllPlayersMask;
  if (IsBettingRoundOver(child)) {
    assert(IsValidBettingData(child));
    return child.street == StreetKind::River
               ? BettingState(ShowdownState{child})
               : BettingState(ChanceState{child});
  }
  assert(IsValidBettingData(child));
  return DecisionState{child, FirstPlayerForStreet(child.street)};
}

ExactPublicState AdvanceChance(const ChanceState& state,
                               const Board& board,
                               absl::Span<const Card> cards,
                               const BettingRules& rules) noexcept {
  assert(rules.minimum_bet > 0);
  assert(cards.size() ==
         static_cast<size_t>(CardsForNextStreet(state.data.street)));
  Board child_board;
  switch (state.data.street) {
    case StreetKind::Preflop: {
      std::array<Card, 3> flop;
      std::copy_n(cards.begin(), 3, flop.begin());
      const auto* preflop = std::get_if<PreflopBoard>(&board);
      assert(preflop != nullptr);
      child_board = DealFlop(*preflop, flop);
      break;
    }
    case StreetKind::Flop: {
      const auto* flop = std::get_if<FlopBoard>(&board);
      assert(flop != nullptr);
      child_board = DealTurn(*flop, cards[0]);
      break;
    }
    case StreetKind::Turn: {
      const auto* turn = std::get_if<TurnBoard>(&board);
      assert(turn != nullptr);
      child_board = DealRiver(*turn, cards[0]);
      break;
    }
    case StreetKind::River:
      child_board = board;
      break;
  }
  return {AdvanceBettingStreet(state, rules), child_board};
}

absl::StatusOr<ExactPublicState> TryApplyChance(
    const ExactPublicState& state,
    absl::Span<const Card> cards,
    const BettingRules& rules) {
  const auto* chance = std::get_if<ChanceState>(&state.betting);
  if (chance == nullptr) {
    return absl::InvalidArgumentError("state is not a chance node");
  }
  if (rules.minimum_bet <= 0 || cards.size() !=
      static_cast<size_t>(CardsForNextStreet(chance->data.street))) {
    return absl::InvalidArgumentError("invalid chance transition");
  }
  CardMask mask = BoardMask(state.board);
  for (Card card : cards) {
    if ((mask & CardBit(card)) != 0) {
      return absl::InvalidArgumentError("duplicate board card");
    }
    mask |= CardBit(card);
  }
  return AdvanceChance(*chance, state.board, cards, rules);
}

double TerminalUtility(const FoldTerminalState& state,
                       Player evaluated_player) noexcept {
  const BettingData& data = state.data;
  const double player0_committed = data.total_committed[0];
  const double player0_utility = state.folded == Player::A
                                     ? -player0_committed
                                     : Pot(data) - player0_committed;
  return evaluated_player == Player::A ? player0_utility : -player0_utility;
}

double TerminalUtility(const ShowdownState& state,
                       const RiverBoard& board,
                       HoleCards player_a,
                       HoleCards player_b) noexcept {
  return TerminalUtilityFromComparison(
      state, CompareHands(player_a.combo(), player_b.combo(), board),
      Player::A);
}

double TerminalUtilityFromComparison(const ShowdownState& state,
                                     int hand_comparison,
                                     Player evaluated_player) noexcept {
  const BettingData& data = state.data;
  const double player0_committed = data.total_committed[0];
  double player0_utility;
  if (hand_comparison > 0) {
    player0_utility = Pot(data) - player0_committed;
  } else if (hand_comparison < 0) {
    player0_utility = -player0_committed;
  } else {
    player0_utility = (Pot(data) / 2.0) - player0_committed;
  }
  return evaluated_player == Player::A ? player0_utility : -player0_utility;
}

}  // namespace poker
