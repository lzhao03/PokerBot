#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/game_state.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/solver_stats.h"
#include "src/training_range.h"

namespace poker {

using HistoryId = uint32_t;
inline constexpr HistoryId kInvalidHistoryId =
    std::numeric_limits<HistoryId>::max();

enum class HistoryNodeKind : uint8_t {
  kDecision,
  kChance,
  kTerminal,
};

struct HistoryEdge {
  GameAction action;
  HistoryId child = kInvalidHistoryId;
};

struct HistoryNode {
  BettingState state;
  uint32_t action_begin = 0;
  uint8_t action_count = 0;
  HistoryId chance_child = kInvalidHistoryId;
  HistoryNodeKind kind = HistoryNodeKind::kDecision;
};

struct HistoryTree {
  HistoryId root = kInvalidHistoryId;
  std::vector<HistoryNode> nodes;
  std::vector<HistoryEdge> edges;
};

struct Position {
  HistoryId history = kInvalidHistoryId;
  BoardRunout board = BoardRunout::Preflop();
  PublicObservationId public_observation = initial_public_observation();
};

struct InfoSetKey {
  HistoryId history = kInvalidHistoryId;
  PublicObservationId public_observation = 0;
  PrivateObservationId private_observation = 0;

  friend bool operator==(const InfoSetKey&, const InfoSetKey&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const InfoSetKey& key) {
    return H::combine(std::move(h), key.history, key.public_observation,
                      key.private_observation);
  }
};

struct InfoSetRow {
  uint32_t action_offset = 0;
  uint8_t action_count = 0;

  friend bool operator==(const InfoSetRow&, const InfoSetRow&) = default;
};

struct CfrState {
  absl::flat_hash_map<InfoSetKey, InfoSetRow> rows;
  std::vector<float> regret_sum;
  std::vector<float> strategy_sum;
  uint64_t iterations = 0;
  double cumulative_root_utility = 0.0;
};

struct CFRSolverTestAccess;

class CFRSolver {
 public:
  using TraversalStats = poker::TraversalStats;
  using TrainingRunStats = poker::TrainingRunStats;

  explicit CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config,
            const ExactPublicState& initial_state);

  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);

  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);

  double get_expected_value(int player_id) const;
  uint64_t get_iterations_run() const { return state_.iterations; }
  int64_t get_cfr_update_count() const { return cfr_update_count_; }
  size_t get_info_set_count() const { return state_.rows.size(); }
  size_t get_history_count() const { return history_.nodes.size(); }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  void reset_traversal_stats() { traversal_stats_ = {}; }
  TrainingRunStats get_last_training_run_stats() const {
    return last_training_run_stats_;
  }
  static bool traversal_stats_enabled();

 private:
  friend struct CFRSolverTestAccess;

  struct Deal {
    std::array<ComboId, kPlayerCount> hands = {};
    CardMask blocked_mask = 0;

    ComboId hand(int player) const {
      return hands[static_cast<size_t>(player)];
    }
  };

  struct TraversalFrame {
    std::array<double, kPlayerCount> reach = {1.0, 1.0};
    std::array<PrivateObservationId, kPlayerCount> private_observations = {};
    uint16_t decision_depth = 0;
  };

  Deal traversal_deal(RangeDeal deal) const;
  Position root_position() const;
  Position action_child(Position position, int action_index) const;
  Position sample_chance_child(Position position, const Deal& deal);
  std::array<PrivateObservationId, kPlayerCount>
  private_observations_for_position(const Deal& deal,
                                    const Position& position) const;
  void advance_private_observations(TraversalFrame& frame,
                                    const Deal& deal,
                                    const Position& child) const;
  double traverse(Position position,
                  const Deal& deal,
                  TraversalFrame frame,
                  int update_player,
                  uint64_t iteration);
  double evaluate_position(Position position,
                           const Deal& deal,
                           TraversalFrame frame);
  InfoSetRow find_or_create_row(InfoSetKey key, uint8_t action_count);
  const InfoSetRow* find_row(InfoSetKey key, uint8_t action_count) const;
  void log_training_summary() const;

  SolverConfig config_;
  BettingRules betting_rules_;
  ExactPublicState initial_state_;
  std::mt19937 rng_;
  BettingAbstraction betting_abstraction_;
  HistoryTree history_;
  CfrState state_;
  int64_t cfr_update_count_ = 0;
  TraversalStats traversal_stats_;
  TrainingRunStats last_training_run_stats_;
};

}  // namespace poker
