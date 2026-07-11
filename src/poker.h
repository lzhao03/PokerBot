#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "absl/container/inlined_vector.h"

namespace poker {

class BoardRunout;

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

}  // namespace poker
