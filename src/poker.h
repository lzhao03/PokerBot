#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace poker {

using CardMask = uint64_t;
using Chips = int32_t;

constexpr size_t kDeckCardCount = 52;
constexpr size_t kMaxBoardCards = 5;
constexpr size_t kPlayerCount = 2;
constexpr size_t kComboCount = 1326;

enum class Player : uint8_t {
  A = 0,
  B = 1,
};

constexpr size_t Index(Player player) noexcept {
  return std::to_underlying(player);
}

constexpr Player Opponent(Player player) noexcept {
  return player == Player::A ? Player::B : Player::A;
}

enum class Suit : uint8_t {
  Hearts = 0,
  Diamonds = 1,
  Clubs = 2,
  Spades = 3,
};

enum class Rank : uint8_t {
  Two,
  Three,
  Four,
  Five,
  Six,
  Seven,
  Eight,
  Nine,
  Ten,
  Jack,
  Queen,
  King,
  Ace,
};

class Card {
 public:
  constexpr Card() = default;
  constexpr Card(Rank rank, Suit suit) noexcept
      : value_(static_cast<uint8_t>(
            std::to_underlying(suit) * 13 + std::to_underlying(rank))) {}

  constexpr size_t index() const noexcept { return value_; }
  constexpr Rank rank() const noexcept {
    return static_cast<Rank>(value_ % 13);
  }
  constexpr Suit suit() const noexcept {
    return static_cast<Suit>(value_ / 13);
  }

  friend constexpr auto operator<=>(const Card&, const Card&) = default;

 private:
  uint8_t value_ = 0;
};

class ComboId {
 public:
  constexpr ComboId() = default;
  constexpr size_t index() const noexcept { return value_; }
  std::array<Card, 2> cards() const noexcept;
  CardMask mask() const noexcept;

  friend constexpr auto operator<=>(const ComboId&,
                                    const ComboId&) = default;

 private:
  explicit constexpr ComboId(uint16_t value) noexcept : value_(value) {}

  friend std::optional<ComboId> MaybeCardsToComboId(Card first,
                                                     Card second);

  uint16_t value_ = 0;
};

enum class PublicObservationId : uint64_t {};
enum class PrivateObservationId : uint32_t {};

inline constexpr std::array<Card, kDeckCardCount> kDeck = [] {
  std::array<Card, kDeckCardCount> cards = {};
  size_t index = 0;
  for (uint8_t suit = 0; suit < 4; ++suit) {
    for (uint8_t rank = 0; rank < 13; ++rank) {
      cards[index++] = Card(static_cast<Rank>(rank),
                            static_cast<Suit>(suit));
    }
  }
  return cards;
}();

enum class StreetKind : uint8_t {
  Preflop = 0,
  Flop = 1,
  Turn = 2,
  River = 3,
};

enum class ActionKind : uint8_t {
  Bet,
  Fold,
  Call,
  Raise,
  Check,
  AllIn,
};

struct GameAction {
  ActionKind kind;
  Chips target_street_commitment = 0;

  friend bool operator==(const GameAction&, const GameAction&) = default;
};

inline CardMask CardBit(Card card) {
  return CardMask{1} << card.index();
}

struct BettingRules {
  Chips minimum_bet = 0;
};

struct BettingData {
  std::array<Chips, kPlayerCount> stack = {0, 0};
  std::array<Chips, kPlayerCount> total_committed = {0, 0};
  std::array<Chips, kPlayerCount> street_committed = {0, 0};
  Chips last_full_raise = 0;
  StreetKind street = StreetKind::Preflop;
  uint8_t actions_remaining = 2;

  friend bool operator==(const BettingData&, const BettingData&) = default;
};

struct DecisionState {
  BettingData data;
  Player actor = Player::A;

  friend bool operator==(const DecisionState&, const DecisionState&) = default;
};

struct ChanceState {
  BettingData data;

  friend bool operator==(const ChanceState&, const ChanceState&) = default;
};

struct FoldTerminalState {
  BettingData data;
  Player folded = Player::A;

  friend bool operator==(const FoldTerminalState&,
                         const FoldTerminalState&) = default;
};

struct ShowdownState {
  BettingData data;

  friend bool operator==(const ShowdownState&,
                         const ShowdownState&) = default;
};

using BettingState = std::variant<DecisionState, ChanceState,
                                  FoldTerminalState, ShowdownState>;

inline const BettingData& Data(const BettingState& state) noexcept {
  return std::visit([](const auto& phase) -> const BettingData& {
    return phase.data;
  }, state);
}

class Board {
 public:
  Board() = default;

  absl::Span<const Card> cards() const noexcept {
    return absl::Span<const Card>(cards_.data(), count_);
  }
  CardMask mask() const noexcept {
    CardMask result = 0;
    for (Card card : cards()) result |= CardBit(card);
    return result;
  }
  size_t count() const noexcept { return count_; }
  bool contains(Card card) const noexcept {
    return (mask() & CardBit(card)) != 0;
  }
  StreetKind street() const noexcept {
    return count_ == 0
               ? StreetKind::Preflop
               : static_cast<StreetKind>(count_ - 2);
  }

  friend bool operator==(const Board&, const Board&) = default;

 private:
  Board(std::array<Card, kMaxBoardCards> cards, uint8_t count)
      : cards_(cards), count_(count) {}

  friend Board DealCards(Board board, absl::Span<const Card> cards) noexcept;
  friend absl::StatusOr<Board> MakeBoard(absl::Span<const Card> cards);

  std::array<Card, kMaxBoardCards> cards_ = {};
  uint8_t count_ = 0;
};

Board DealCards(Board board, absl::Span<const Card> cards) noexcept;
absl::StatusOr<Board> MakeBoard(absl::Span<const Card> cards);

struct ExactPublicState {
  BettingState betting;
  Board board;
};

inline Chips Pot(const BettingData& state) noexcept {
  return state.total_committed[0] + state.total_committed[1];
}

inline Chips CurrentWager(const BettingData& state) noexcept {
  return std::max(state.street_committed[0], state.street_committed[1]);
}

inline Chips ToCall(const BettingData& state, Player player) noexcept {
  return CurrentWager(state) - state.street_committed[Index(player)];
}

inline Chips MaxContestableAdditional(const BettingData& state,
                                      Player player) noexcept {
  return std::min(state.stack[Index(player)],
                  ToCall(state, player) + state.stack[Index(Opponent(player))]);
}

inline bool IsValidBettingData(const BettingData& state) noexcept {
  return state.stack[0] >= 0 && state.stack[1] >= 0 &&
         state.total_committed[0] >= 0 &&
         state.total_committed[1] >= 0 &&
         state.street_committed[0] >= 0 &&
         state.street_committed[1] >= 0 &&
         state.street_committed[0] <= state.total_committed[0] &&
         state.street_committed[1] <= state.total_committed[1] &&
         state.last_full_raise > 0 &&
         state.actions_remaining <= 2;
}

std::optional<ComboId> MaybeCardsToComboId(Card first, Card second);
ComboId CardsToComboId(Card first, Card second) noexcept;

size_t CardsForNextStreet(StreetKind street);
absl::StatusOr<absl::InlinedVector<Card, 5>> SampleStreetCards(
    StreetKind street,
    const Board& board,
    CardMask known_private_cards,
    std::mt19937& rng);

ExactPublicState MakeInitialState(
    const BettingRules& rules,
    std::array<Chips, kPlayerCount> stacks,
    std::array<Chips, kPlayerCount> blinds);
bool IsLegalAction(const DecisionState& state,
                   const GameAction& action) noexcept;
absl::StatusOr<BettingState> ApplyAction(const DecisionState& state,
                                         const GameAction& action);
BettingState AdvanceBettingStreet(const ChanceState& state,
                                  const BettingRules& rules);
absl::StatusOr<ExactPublicState> TryApplyChance(
    const ExactPublicState& state,
    absl::Span<const Card> cards,
    const BettingRules& rules);
inline double TerminalUtility(const FoldTerminalState& state,
                              Player evaluated_player) noexcept {
  const double committed = state.data.total_committed[0];
  const double player_a_utility = state.folded == Player::A
                                      ? -committed
                                      : Pot(state.data) - committed;
  return evaluated_player == Player::A ? player_a_utility
                                        : -player_a_utility;
}
double TerminalUtility(const ShowdownState& state,
                       const Board& board,
                       ComboId player_a,
                       ComboId player_b) noexcept;
inline double TerminalUtilityFromComparison(
    const ShowdownState& state,
    int hand_comparison,
    Player evaluated_player) noexcept {
  const double committed = state.data.total_committed[0];
  const double player_a_utility =
      hand_comparison > 0
          ? Pot(state.data) - committed
          : hand_comparison < 0 ? -committed
                                : (Pot(state.data) / 2.0) - committed;
  return evaluated_player == Player::A ? player_a_utility
                                        : -player_a_utility;
}
}  // namespace poker
