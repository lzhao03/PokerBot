#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace poker {

using CardId = uint8_t;
using CardMask = uint64_t;
using BoardBucketId = uint64_t;
using PublicObservationId = uint64_t;
using PrivateObservationId = uint32_t;
using Chips = int32_t;

constexpr int kDeckCardCount = 52;
constexpr int kMaxBoardCards = 5;
constexpr int kPlayerCount = 2;

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

// Maximum number of legal actions at any decision node.
// With 3 bet sizes: fold/call + 3 raises + all-in = 6 (facing bet),
// or check + 3 bets + all-in = 5 (no bet). 8 gives headroom.
inline constexpr int kMaxActionsPerNode = 8;

struct SolverConfig {
  std::vector<double> bet_sizes;
  int starting_stack_size = 100;
  int max_depth = 0;
  bool enable_logging = false;
  int small_blind = 1;
  int big_blind = 2;
  int chance_samples = 1;
  std::vector<double> preflop_bet_sizes;
  std::vector<double> flop_bet_sizes;
  std::vector<double> turn_bet_sizes;
  std::vector<double> river_bet_sizes;
  bool regret_only_training = false;
  // Maximum number of info sets to allocate. 0 means unlimited.
  // Once this limit is reached, new (combo, node) pairs fall back to
  // uniform strategy rather than allocating new entries. Controls peak memory.
  int max_info_sets = 0;
  // Maximum number of graph nodes to allocate. 0 means
  // unlimited. Once this limit is reached, new public states are skipped
  // instead of expanding the sampled branch. Controls peak traversal memory.
  int max_public_states = 0;
  // Number of threads to use for parallel training. 0 or 1 = single-threaded.
  int num_training_threads = 0;
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

}  // namespace poker
