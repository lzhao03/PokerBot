#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

namespace poker {

using CardId = uint8_t;
using CardMask = uint64_t;
using BoardBucketId = uint64_t;
using PublicObservationId = uint64_t;
using PrivateObservationId = uint64_t;
using Chips = int32_t;
using ComboId = uint16_t;

constexpr int kDeckCardCount = 52;
constexpr int kMaxBoardCards = 5;
constexpr int kPlayerCount = 2;
constexpr int kComboCount = 1326;

struct ComboInfo {
  CardId card0 = 0;
  CardId card1 = 0;
  CardMask mask = 0;
};

enum class Player : uint8_t {
  kA = 0,
  kB = 1,
};

inline int PlayerIndex(Player player) {
  return static_cast<int>(player);
}

enum class SuitKind : uint8_t {
  kHearts = 0,
  kDiamonds = 1,
  kClubs = 2,
  kSpades = 3,
};

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
  // Maximum number of info sets to allocate. 0 means unlimited.
  // Training throws when the limit is reached.
  int max_info_sets = 0;
};

inline int SuitIndex(SuitKind suit) {
  return static_cast<int>(suit);
}

inline int RankFromCardId(CardId card_id) {
  return 2 + static_cast<int>(card_id % 13);
}

inline SuitKind SuitFromCardId(CardId card_id) {
  return static_cast<SuitKind>(card_id / 13);
}

inline CardId MakeCardId(int rank, SuitKind suit) {
  if (rank == 1) {
    rank = 14;
  }
  const int rank_index = rank - 2;
  const int suit_index = SuitIndex(suit);
  if (rank_index < 0 || rank_index >= 13 || suit_index < 0 ||
      suit_index >= 4) {
    throw std::invalid_argument("Invalid card");
  }
  return static_cast<CardId>(suit_index * 13 + rank_index);
}

inline CardMask CardBit(CardId card_id) {
  return CardMask{1} << card_id;
}

inline int EncodedCard(CardId card_id) {
  return RankFromCardId(card_id) * 8 + 1 + SuitIndex(SuitFromCardId(card_id));
}

inline bool IsPlayer(int player) {
  return player == 0 || player == 1;
}

inline int Opponent(int player) {
  return 1 - player;
}

struct BettingRules {
  Chips minimum_bet = 0;
};

constexpr uint8_t PlayerBit(int player) {
  return static_cast<uint8_t>(1u << player);
}

constexpr uint8_t kAllPlayersMask =
    static_cast<uint8_t>((1u << kPlayerCount) - 1u);

struct BettingState {
  std::array<Chips, kPlayerCount> stack = {0, 0};
  std::array<Chips, kPlayerCount> total_committed = {0, 0};
  std::array<Chips, kPlayerCount> street_committed = {0, 0};
  Chips last_full_raise = 0;
  StreetKind street = StreetKind::kPreflop;
  int8_t player_to_act = 0;
  int8_t folded_player = -1;
  uint8_t pending_action_mask = kAllPlayersMask;

  friend bool operator==(const BettingState&, const BettingState&) = default;
};

class BoardRunout {
 public:
  static BoardRunout Preflop() { return BoardRunout(); }

  void deal_flop(absl::Span<const CardId> cards) {
    if (count_ != 0) {
      throw std::logic_error("flop requires a preflop runout");
    }
    if (cards.size() != 3) {
      throw std::invalid_argument("flop requires exactly three cards");
    }

    CardMask dealt_mask = 0;
    for (CardId card : cards) {
      const CardMask bit = CardBit(card);
      if ((dealt_mask & bit) != 0) {
        throw std::invalid_argument("duplicate board card");
      }
      dealt_mask |= bit;
    }

    std::copy(cards.begin(), cards.end(), cards_.begin());
    std::sort(cards_.begin(), cards_.begin() + 3);
    mask_ = dealt_mask;
    count_ = 3;
  }

  void deal_turn(CardId card) {
    if (count_ != 3) {
      throw std::logic_error("turn requires a dealt flop");
    }
    deal_card(card, 3);
  }

  void deal_river(CardId card) {
    if (count_ != 4) {
      throw std::logic_error("river requires a dealt turn");
    }
    deal_card(card, 4);
  }

  absl::Span<const CardId> cards() const {
    return absl::Span<const CardId>(cards_.data(), count_);
  }

  CardMask mask() const { return mask_; }
  uint8_t count() const { return count_; }

  bool contains(CardId card) const {
    return (mask_ & CardBit(card)) != 0;
  }

  bool operator==(const BoardRunout&) const = default;

 private:
  BoardRunout() = default;

  void deal_card(CardId card, size_t index) {
    if (contains(card)) {
      throw std::invalid_argument("duplicate board card");
    }
    cards_[index] = card;
    mask_ |= CardBit(card);
    ++count_;
  }

  std::array<CardId, kMaxBoardCards> cards_ = {};
  CardMask mask_ = 0;
  uint8_t count_ = 0;
};

struct ExactPublicState {
  BettingState betting;
  BoardRunout board = BoardRunout::Preflop();
};

inline Chips Pot(const BettingState& state) noexcept {
  return state.total_committed[0] + state.total_committed[1];
}

inline Chips HighestStreetCommitment(const BettingState& state) noexcept {
  return std::max(state.street_committed[0], state.street_committed[1]);
}

inline Chips ToCall(const BettingState& state, int player) noexcept {
  return HighestStreetCommitment(state) - state.street_committed[player];
}

inline Chips MaxContestableAdditional(const BettingState& state,
                                      int player) noexcept {
  return std::min(state.stack[player],
                  ToCall(state, player) + state.stack[Opponent(player)]);
}

inline bool AnyPlayerAllIn(const BettingState& state) noexcept {
  return state.stack[0] == 0 || state.stack[1] == 0;
}

inline bool IsValidBettingState(const BettingState& state) noexcept {
  const bool completed_non_fold =
      state.folded_player < 0 && state.player_to_act == -1;
  return state.stack[0] >= 0 && state.stack[1] >= 0 &&
         state.total_committed[0] >= 0 &&
         state.total_committed[1] >= 0 &&
         state.street_committed[0] >= 0 &&
         state.street_committed[1] >= 0 &&
         state.street_committed[0] <= state.total_committed[0] &&
         state.street_committed[1] <= state.total_committed[1] &&
         state.last_full_raise > 0 &&
         (state.pending_action_mask & ~kAllPlayersMask) == 0 &&
         state.folded_player >= -1 && state.folded_player < kPlayerCount &&
         state.player_to_act >= -1 && state.player_to_act < kPlayerCount &&
         (!completed_non_fold ||
          state.street_committed[0] == state.street_committed[1]);
}

using SolverActions = absl::InlinedVector<GameAction, 8>;

const ComboInfo& GetComboInfo(ComboId combo_id);
CardMask ComboMask(ComboId combo_id);
std::optional<ComboId> MaybeCardsToComboId(CardId first, CardId second);
ComboId CardsToComboId(CardId first, CardId second);

int CardsForNextStreet(StreetKind street);
int BoardCardsForStreet(StreetKind street);
absl::InlinedVector<CardId, 5> SampleStreetCards(
    StreetKind street,
    const BoardRunout& board,
    CardMask known_private_cards,
    std::mt19937& rng);

ExactPublicState MakeInitialState(
    const BettingRules& rules,
    std::array<Chips, kPlayerCount> stacks,
    std::array<Chips, kPlayerCount> blinds);
SolverActions GetSolverActions(const SolverConfig& config,
                               const BettingState& state);
BettingState ApplyAction(const BettingState& state,
                         const GameAction& action);
BettingState AdvanceBettingStreet(const BettingState& state,
                                  const BettingRules& rules);
ExactPublicState ApplyChance(const ExactPublicState& state,
                             absl::Span<const CardId> cards,
                             const BettingRules& rules);
double TerminalUtility(const ExactPublicState& state,
                       ComboId player0_hand,
                       ComboId player1_hand);
bool IsTerminal(const ExactPublicState& state);
bool IsBettingRoundOver(const BettingState& state) noexcept;

}  // namespace poker
