#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "absl/types/span.h"
#include "src/poker.h"

namespace poker {

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

}  // namespace poker
