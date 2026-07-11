#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace poker {

using CardMask = uint64_t;
using BoardBucketId = uint64_t;
using Chips = int32_t;

constexpr int kDeckCardCount = 52;
constexpr int kMaxBoardCards = 5;
constexpr int kPlayerCount = 2;
constexpr int kComboCount = 1326;

enum class Player : uint8_t {
  kA = 0,
  kB = 1,
};

constexpr size_t Index(Player player) noexcept {
  return static_cast<size_t>(player);
}

constexpr Player Opponent(Player player) noexcept {
  return player == Player::kA ? Player::kB : Player::kA;
}

enum class SuitKind : uint8_t {
  kHearts = 0,
  kDiamonds = 1,
  kClubs = 2,
  kSpades = 3,
};

enum class Rank : uint8_t {
  kTwo,
  kThree,
  kFour,
  kFive,
  kSix,
  kSeven,
  kEight,
  kNine,
  kTen,
  kJack,
  kQueen,
  kKing,
  kAce,
};

class Card {
 public:
  constexpr Card() = default;
  constexpr Card(Rank rank, SuitKind suit) noexcept
      : value_(static_cast<uint8_t>(
            static_cast<uint8_t>(suit) * 13 +
            static_cast<uint8_t>(rank))) {}

  constexpr size_t index() const noexcept { return value_; }
  constexpr Rank rank() const noexcept {
    return static_cast<Rank>(value_ % 13);
  }
  constexpr SuitKind suit() const noexcept {
    return static_cast<SuitKind>(value_ / 13);
  }

  friend constexpr auto operator<=>(const Card&, const Card&) = default;

 private:
  uint8_t value_ = 0;
};

class ComboId {
 public:
  constexpr ComboId() = default;
  constexpr size_t index() const noexcept { return value_; }

  friend constexpr auto operator<=>(const ComboId&,
                                    const ComboId&) = default;

 private:
  explicit constexpr ComboId(uint16_t value) noexcept : value_(value) {}

  friend std::optional<ComboId> MaybeCardsToComboId(Card first,
                                                     Card second);

  uint16_t value_ = 0;
};

class PublicObservationId {
 public:
  constexpr PublicObservationId() = default;
  explicit constexpr PublicObservationId(uint64_t value) noexcept
      : value_(value) {}

  constexpr uint64_t value() const noexcept { return value_; }
  friend constexpr auto operator<=>(const PublicObservationId&,
                                    const PublicObservationId&) = default;

 private:
  uint64_t value_ = 0;
};

class PrivateObservationId {
 public:
  constexpr PrivateObservationId() = default;
  explicit constexpr PrivateObservationId(uint64_t value) noexcept
      : value_(value) {}

  constexpr uint64_t value() const noexcept { return value_; }
  friend constexpr auto operator<=>(const PrivateObservationId&,
                                    const PrivateObservationId&) = default;

 private:
  uint64_t value_ = 0;
};

struct ComboInfo {
  Card card0;
  Card card1;
  CardMask mask = 0;
};

class HoleCards {
 public:
  constexpr HoleCards() = default;
  explicit constexpr HoleCards(ComboId combo) noexcept : combo_(combo) {}

  constexpr ComboId combo() const noexcept { return combo_; }

 private:
  ComboId combo_;
};

inline constexpr std::array<Card, kDeckCardCount> kDeck = [] {
  std::array<Card, kDeckCardCount> cards = {};
  size_t index = 0;
  for (uint8_t suit = 0; suit < 4; ++suit) {
    for (uint8_t rank = 0; rank < 13; ++rank) {
      cards[index++] = Card(static_cast<Rank>(rank),
                            static_cast<SuitKind>(suit));
    }
  }
  return cards;
}();

enum class StreetKind : uint8_t {
  kPreflop = 0,
  kFlop = 1,
  kTurn = 2,
  kRiver = 3,
};

enum class ActionKind : uint8_t {
  kNoAction = 0,
  kBet = 1,
  kFold = 2,
  kCall = 3,
  kRaise = 4,
  kCheck = 5,
  kAllIn = 6,
};

struct GameAction {
  ActionKind kind = ActionKind::kNoAction;
  Chips target_street_commitment = 0;

  friend bool operator==(const GameAction&, const GameAction&) = default;
};

struct SolverConfig {
  std::array<std::vector<double>, 4> bet_sizes = {{
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
      {0.25, 0.5, 1.0},
  }};
  Chips starting_stack = 100;
  Chips small_blind = 1;
  Chips big_blind = 2;
  int chance_samples = 1;
  bool accumulate_average_strategy = true;
  int max_info_sets = 500000;
};

inline int SuitIndex(SuitKind suit) {
  return static_cast<int>(suit);
}

inline int RankFromCardId(Card card) {
  return 2 + static_cast<int>(card.rank());
}

inline SuitKind SuitFromCardId(Card card) {
  return card.suit();
}

inline Card MakeCardId(int rank, SuitKind suit) {
  if (rank == 1) {
    rank = 14;
  }
  const int rank_index = rank - 2;
  const int suit_index = SuitIndex(suit);
  if (rank_index < 0 || rank_index >= 13 || suit_index < 0 ||
      suit_index >= 4) {
    throw std::invalid_argument("Invalid card");
  }
  return Card(static_cast<Rank>(rank_index), suit);
}

inline CardMask CardBit(Card card) {
  return CardMask{1} << card.index();
}

inline int EncodedCard(Card card) {
  return RankFromCardId(card) * 8 + 1 + SuitIndex(SuitFromCardId(card));
}

struct BettingRules {
  Chips minimum_bet = 0;
};

constexpr uint8_t PlayerBit(int player) {
  return static_cast<uint8_t>(1u << player);
}

constexpr uint8_t kAllPlayersMask =
    static_cast<uint8_t>((1u << kPlayerCount) - 1u);

struct BettingData {
  std::array<Chips, kPlayerCount> stack = {0, 0};
  std::array<Chips, kPlayerCount> total_committed = {0, 0};
  std::array<Chips, kPlayerCount> street_committed = {0, 0};
  Chips last_full_raise = 0;
  StreetKind street = StreetKind::kPreflop;
  uint8_t pending_action_mask = kAllPlayersMask;

  friend bool operator==(const BettingData&, const BettingData&) = default;
};

struct DecisionState {
  BettingData data;
  Player actor = Player::kA;

  friend bool operator==(const DecisionState&, const DecisionState&) = default;
};

struct ChanceState {
  BettingData data;

  friend bool operator==(const ChanceState&, const ChanceState&) = default;
};

struct FoldTerminalState {
  BettingData data;
  Player folded = Player::kA;

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

struct PreflopBoard {
  friend bool operator==(const PreflopBoard&, const PreflopBoard&) = default;
};

class FlopBoard {
 public:
  absl::Span<const Card> cards() const { return cards_; }
  CardMask mask() const { return mask_; }

  friend bool operator==(const FlopBoard&, const FlopBoard&) = default;

 private:
  FlopBoard(std::array<Card, 3> cards, CardMask mask)
      : cards_(cards), mask_(mask) {}

  friend FlopBoard DealFlop(const PreflopBoard&,
                            std::array<Card, 3> cards) noexcept;
  friend absl::StatusOr<FlopBoard> MakeFlop(std::array<Card, 3> cards);

  std::array<Card, 3> cards_ = {};
  CardMask mask_ = 0;
};

class TurnBoard {
 public:
  absl::Span<const Card> cards() const {
    return absl::Span<const Card>(cards_.data(), 4);
  }
  CardMask mask() const { return mask_; }

  friend bool operator==(const TurnBoard&, const TurnBoard&) = default;

 private:
  TurnBoard(std::array<Card, 5> cards, CardMask mask)
      : cards_(cards), mask_(mask) {}

  friend TurnBoard DealTurn(const FlopBoard&, Card card) noexcept;
  friend absl::StatusOr<TurnBoard> MakeTurn(const FlopBoard&, Card card);

  std::array<Card, 5> cards_ = {};
  CardMask mask_ = 0;
};

class RiverBoard {
 public:
  absl::Span<const Card> cards() const { return cards_; }
  CardMask mask() const { return mask_; }

  friend bool operator==(const RiverBoard&, const RiverBoard&) = default;

 private:
  RiverBoard(std::array<Card, 5> cards, CardMask mask)
      : cards_(cards), mask_(mask) {}

  friend RiverBoard DealRiver(const TurnBoard&, Card card) noexcept;
  friend absl::StatusOr<RiverBoard> MakeRiver(const TurnBoard&, Card card);

  std::array<Card, 5> cards_ = {};
  CardMask mask_ = 0;
};

using Board = std::variant<PreflopBoard, FlopBoard, TurnBoard, RiverBoard>;

FlopBoard DealFlop(const PreflopBoard& board,
                   std::array<Card, 3> cards) noexcept;
TurnBoard DealTurn(const FlopBoard& board, Card card) noexcept;
RiverBoard DealRiver(const TurnBoard& board, Card card) noexcept;
absl::StatusOr<FlopBoard> MakeFlop(std::array<Card, 3> cards);
absl::StatusOr<TurnBoard> MakeTurn(const FlopBoard& board, Card card);
absl::StatusOr<RiverBoard> MakeRiver(const TurnBoard& board, Card card);

absl::Span<const Card> BoardCards(const Board& board) noexcept;
CardMask BoardMask(const Board& board) noexcept;
uint8_t BoardCount(const Board& board) noexcept;
bool BoardContains(const Board& board, Card card) noexcept;

struct ExactPublicState {
  BettingState betting;
  Board board = PreflopBoard{};
};

inline Chips Pot(const BettingData& state) noexcept {
  return state.total_committed[0] + state.total_committed[1];
}

inline Chips HighestStreetCommitment(const BettingData& state) noexcept {
  return std::max(state.street_committed[0], state.street_committed[1]);
}

inline Chips ToCall(const BettingData& state, Player player) noexcept {
  return HighestStreetCommitment(state) - state.street_committed[Index(player)];
}

inline Chips MaxContestableAdditional(const BettingData& state,
                                      Player player) noexcept {
  return std::min(state.stack[Index(player)],
                  ToCall(state, player) + state.stack[Index(Opponent(player))]);
}

inline bool AnyPlayerAllIn(const BettingData& state) noexcept {
  return state.stack[0] == 0 || state.stack[1] == 0;
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
         (state.pending_action_mask & ~kAllPlayersMask) == 0;
}

struct SolverTransition {
  GameAction action;
  BettingState child;
};

using SolverTransitions = absl::InlinedVector<SolverTransition, 8>;

const ComboInfo& GetComboInfo(ComboId combo_id);
CardMask ComboMask(ComboId combo_id);
std::optional<ComboId> MaybeCardsToComboId(Card first, Card second);
ComboId CardsToComboId(Card first, Card second);

int CardsForNextStreet(StreetKind street);
int BoardCardsForStreet(StreetKind street);
absl::InlinedVector<Card, 5> SampleStreetCards(
    StreetKind street,
    const Board& board,
    CardMask known_private_cards,
    std::mt19937& rng);

ExactPublicState MakeInitialState(
    const BettingRules& rules,
    std::array<Chips, kPlayerCount> stacks,
    std::array<Chips, kPlayerCount> blinds);
SolverTransitions GenerateTransitions(const SolverConfig& config,
                                      const DecisionState& state);
absl::StatusOr<BettingState> TryApplyAction(const DecisionState& state,
                                            const GameAction& action);
BettingState AdvanceBettingStreet(const ChanceState& state,
                                  const BettingRules& rules);
ExactPublicState ApplyChance(const ExactPublicState& state,
                             absl::Span<const Card> cards,
                             const BettingRules& rules);
double TerminalUtility(const FoldTerminalState& state,
                       Player evaluated_player) noexcept;
double TerminalUtility(const ShowdownState& state,
                       const RiverBoard& board,
                       HoleCards player_a,
                       HoleCards player_b) noexcept;
bool IsTerminal(const ExactPublicState& state);

}  // namespace poker
