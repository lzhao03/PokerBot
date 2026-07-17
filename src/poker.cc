#include "src/poker.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <random>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

inline constexpr auto kComboCards = [] {
  std::array<std::array<Card, 2>, kComboCount> combos;
  size_t combo = 0;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      combos[combo++] = {kDeck[first], kDeck[second]};
    }
  }
  return combos;
}();

void CommitChips(BettingData& state, Player player, Chips requested) {
  assert(requested > 0);
  const size_t index = Index(player);
  const Chips committed = std::min(requested, state.stack[index]);
  state.stack[index] -= committed;
  state.total_committed[index] += committed;
  state.street_committed[index] += committed;
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
  if (state.actions_remaining == 0 && commitments_match) {
    return true;
  }
  if (state.stack[0] != 0 && state.stack[1] != 0) {
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

  if (action.kind == ActionKind::Fold) {
    return FoldTerminalState{child, player};
  }
  if (delta > 0) CommitChips(child, player, delta);

  const bool aggressive =
      action.target_street_commitment > highest_before;
  if (aggressive) {
    const Chips raise_size =
        child.street_committed[player_index] - highest_before;
    if (raise_size >= child.last_full_raise) {
      child.last_full_raise = raise_size;
    }
    child.actions_remaining = 1;
  } else {
    assert(child.actions_remaining > 0);
    --child.actions_remaining;
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

Board DealCards(Board board, absl::Span<const Card> cards) noexcept {
  assert((board.count() == 0 && cards.size() == 3) ||
         (board.count() >= 3 && board.count() < kMaxBoardCards &&
          cards.size() == 1));
  for (Card card : cards) {
    assert(!board.contains(card));
    board.cards_[board.count_++] = card;
  }
  if (board.count_ == 3) {
    std::sort(board.cards_.begin(), board.cards_.begin() + 3);
  }
  return board;
}

absl::StatusOr<Board> MakeBoard(absl::Span<const Card> cards) {
  if (!(cards.empty() || (cards.size() >= 3 &&
                          cards.size() <= kMaxBoardCards))) {
    return absl::InvalidArgumentError("board must have 0, 3, 4, or 5 cards");
  }
  std::array<Card, kMaxBoardCards> stored = {};
  CardMask mask = 0;
  for (size_t index = 0; index < cards.size(); ++index) {
    const Card card = cards[index];
    if ((mask & CardBit(card)) != 0) {
      return absl::InvalidArgumentError("duplicate board card");
    }
    stored[index] = card;
    mask |= CardBit(card);
  }
  if (cards.size() >= 3) {
    std::sort(stored.begin(), stored.begin() + 3);
  }
  return Board(stored, static_cast<uint8_t>(cards.size()));
}

std::array<Card, 2> ComboId::cards() const noexcept {
  return kComboCards[index()];
}

CardMask ComboId::mask() const noexcept {
  const auto [first, second] = kComboCards[index()];
  return CardBit(first) | CardBit(second);
}

std::optional<ComboId> MaybeCardsToComboId(Card first, Card second) {
  if (first == second) {
    return std::nullopt;
  }
  if (second < first) {
    std::swap(first, second);
  }

  const size_t first_index = first.index();
  const size_t combo =
      first_index * (2 * kDeckCardCount - first_index - 1) / 2 +
      second.index() - first_index - 1;
  return ComboId(static_cast<uint16_t>(combo));
}

ComboId CardsToComboId(Card first, Card second) noexcept {
  const std::optional<ComboId> combo = MaybeCardsToComboId(first, second);
  assert(combo.has_value());
  return *combo;
}

size_t CardsForNextStreet(StreetKind street) {
  if (street == StreetKind::Preflop) return 3;
  return street == StreetKind::River ? 0 : 1;
}

absl::StatusOr<absl::InlinedVector<Card, 5>> SampleStreetCards(
    StreetKind street,
    const Board& board,
    CardMask known_private_cards,
    std::mt19937& rng) {
  const size_t open_slots = kMaxBoardCards - board.count();
  const size_t count = std::min(CardsForNextStreet(street), open_slots);
  if (count == 0) {
    return absl::InlinedVector<Card, 5>{};
  }

  CardMask blocked = known_private_cards | board.mask();
  absl::InlinedVector<Card, 5> sampled;
  sampled.reserve(count);
  std::uniform_int_distribution<uint32_t> card_dist(0, kDeckCardCount - 1);
  for (size_t attempt = 0;
       attempt < kDeckCardCount && sampled.size() < count; ++attempt) {
    const Card candidate = kDeck[card_dist(rng)];
    const CardMask bit = CardBit(candidate);
    if ((blocked & bit) == 0) {
      sampled.push_back(candidate);
      blocked |= bit;
    }
  }
  if (sampled.size() == count) return sampled;

  std::array<Card, kDeckCardCount> candidates = {};
  uint32_t candidate_count = 0;
  for (Card candidate : kDeck) {
    if ((blocked & CardBit(candidate)) == 0) {
      candidates[candidate_count++] = candidate;
    }
  }
  const size_t remaining = count - sampled.size();
  if (candidate_count < remaining) {
    return absl::InvalidArgumentError("not enough unblocked cards");
  }

  for (uint32_t i = 0; i < remaining; ++i) {
    std::uniform_int_distribution<uint32_t> card_dist(i, candidate_count - 1);
    const uint32_t chosen = card_dist(rng);
    std::swap(candidates[i], candidates[chosen]);
    sampled.push_back(candidates[i]);
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
  return ExactPublicState{DecisionState{betting, Player::A}, Board{}};
}

bool IsTerminal(const ExactPublicState& state) {
  return std::holds_alternative<FoldTerminalState>(state.betting) ||
         (std::holds_alternative<ShowdownState>(state.betting) &&
          state.board.count() == kMaxBoardCards);
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
      std::to_underlying(state.data.street) + 1);
  child.street_committed = {0, 0};
  child.last_full_raise = rules.minimum_bet;
  child.actions_remaining = 2;
  if (IsBettingRoundOver(child)) {
    assert(IsValidBettingData(child));
    return child.street == StreetKind::River
               ? BettingState(ShowdownState{child})
               : BettingState(ChanceState{child});
  }
  assert(IsValidBettingData(child));
  return DecisionState{child, Player::B};
}

absl::StatusOr<ExactPublicState> TryApplyChance(
    const ExactPublicState& state,
    absl::Span<const Card> cards,
    const BettingRules& rules) {
  const auto* chance = std::get_if<ChanceState>(&state.betting);
  if (chance == nullptr) {
    return absl::InvalidArgumentError("state is not a chance node");
  }
  if (rules.minimum_bet <= 0 || state.board.street() != chance->data.street ||
      cards.size() != CardsForNextStreet(chance->data.street)) {
    return absl::InvalidArgumentError("invalid chance transition");
  }
  CardMask mask = state.board.mask();
  for (Card card : cards) {
    if ((mask & CardBit(card)) != 0) {
      return absl::InvalidArgumentError("duplicate board card");
    }
    mask |= CardBit(card);
  }
  return ExactPublicState{
      AdvanceBettingStreet(*chance, rules), DealCards(state.board, cards)};
}

double TerminalUtility(const ShowdownState& state,
                       const Board& board,
                       ComboId player_a,
                       ComboId player_b) noexcept {
  return TerminalUtilityFromComparison(
      state, CompareHands(player_a, player_b, board), Player::A);
}

}  // namespace poker
