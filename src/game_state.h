#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "absl/types/span.h"
#include "src/poker_types.h"

namespace poker {

using Chips = int32_t;

constexpr uint8_t PlayerBit(int player) {
  return static_cast<uint8_t>(1u << player);
}

constexpr uint8_t kAllPlayersMask =
    static_cast<uint8_t>((1u << kPlayerCount) - 1u);

struct BettingState {
  std::array<Chips, kPlayerCount> stack = {0, 0};
  std::array<Chips, kPlayerCount> committed = {0, 0};
  StreetKind street = StreetKind::kPreflop;
  int8_t player_to_act = 0;
  int8_t folded_player = -1;
  uint8_t pending_action_mask = kAllPlayersMask;

  friend bool operator==(const BettingState&, const BettingState&) = default;
};

struct Board {
  std::array<CardId, kMaxBoardCards> cards = {};
  CardMask mask = 0;
  uint8_t count = 0;

  absl::Span<const CardId> span() const {
    return absl::Span<const CardId>(cards.data(), count);
  }

  bool contains(CardId card) const {
    return (mask & CardBit(card)) != 0;
  }

  void add(CardId card) {
    if (count >= cards.size()) {
      throw std::logic_error("board is full");
    }
    if (contains(card)) {
      throw std::invalid_argument("duplicate board card");
    }
    cards[static_cast<size_t>(count)] = card;
    ++count;
    mask |= CardBit(card);
  }
};

struct ExactGameState {
  BettingState betting;
  Board board;
};

inline Chips Pot(const BettingState& state) noexcept {
  return state.committed[0] + state.committed[1];
}

inline Chips ToCall(const BettingState& state, int player) noexcept {
  return std::max(Chips{0},
                  state.committed[Opponent(player)] -
                      state.committed[player]);
}

inline bool AnyPlayerAllIn(const BettingState& state) noexcept {
  return state.stack[0] == 0 || state.stack[1] == 0;
}

inline void ValidateBettingState(const BettingState& state) {
  assert(state.stack[0] >= 0);
  assert(state.stack[1] >= 0);
  assert(state.committed[0] >= 0);
  assert(state.committed[1] >= 0);
  assert((state.pending_action_mask & ~kAllPlayersMask) == 0);
  assert(state.folded_player >= -1 && state.folded_player < kPlayerCount);
  assert(state.player_to_act >= -1 && state.player_to_act < kPlayerCount);
}

}  // namespace poker
