#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "src/card_abstraction.h"
#include "src/poker.h"

namespace poker {

struct ComboRange {
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  void add(ComboId combo, float weight = 1.0f) {
    if (weight > 0.0f) {
      if (weights[combo.index()] == 0.0f) {
        active[active_count++] = combo;
      }
      weights[combo.index()] += weight;
    }
  }

  size_t count() const { return active_count; }
  float weight(ComboId combo) const { return weights[combo.index()]; }
};

absl::StatusOr<ComboRange> ParseRange(std::string_view text);
ComboRange UniformRange();
ComboRange SingleComboRange(ComboId combo, float weight = 1.0f);

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

struct SolverStats {
  uint64_t decision_visits = 0;
  uint64_t chance_samples = 0;
  uint64_t terminal_visits = 0;
};

enum class StrategySource : uint8_t {
  kCurrent,
  kAverage,
};

struct Deal {
  std::array<HoleCards, kPlayerCount> hands = {};
  CardMask blocked_mask = 0;

  const HoleCards& hand(Player player) const {
    return hands[static_cast<size_t>(PlayerIndex(player))];
  }
};

struct CFRSolverTestAccess;

class CFRSolver {
 public:
  explicit CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config,
            const ExactPublicState& initial_state);

  void run(uint64_t iterations, const ComboRange& player_a_range,
           const ComboRange& player_b_range);

  double evaluate_strategy(ComboId player_a_hand,
                           ComboId player_b_hand,
                           StrategySource source);
  double evaluate_strategy(int samples, const ComboRange& player_a_range,
                           const ComboRange& player_b_range,
                           StrategySource source);

  double get_expected_value(int player_id) const;
  uint64_t get_iterations_run() const { return state_.iterations; }
  uint64_t get_cfr_update_count() const { return stats_.decision_visits; }
  size_t get_info_set_count() const { return state_.rows.size(); }
  size_t get_history_count() const { return history_.nodes.size(); }
  size_t get_regret_bytes() const {
    return state_.regret_sum.size() * sizeof(float);
  }
  size_t get_strategy_bytes() const {
    return state_.strategy_sum.size() * sizeof(float);
  }
  SolverStats get_stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  friend struct CFRSolverTestAccess;

  struct TraversalFrame {
    std::array<double, kPlayerCount> reach = {1.0, 1.0};
    std::array<PrivateObservationId, kPlayerCount> private_observations = {};
  };

  enum class TraversalMode : uint8_t {
    kTrain,
    kEvaluateCurrent,
    kEvaluateAverage,
  };

  struct TraversalContext {
    const Deal& deal;
    TraversalMode mode = TraversalMode::kTrain;
    int update_player = -1;
    uint64_t iteration = 0;
  };

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
                  TraversalFrame frame,
                  TraversalContext& context);
  InfoSetRow find_or_create_row(InfoSetKey key, uint8_t action_count);
  const InfoSetRow* find_row(InfoSetKey key, uint8_t action_count) const;
  void log_training_summary() const;

  SolverConfig config_;
  BettingRules betting_rules_;
  ExactPublicState initial_state_;
  std::mt19937 rng_;
  HistoryTree history_;
  CfrState state_;
  SolverStats stats_;
};

}  // namespace poker
