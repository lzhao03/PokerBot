#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "src/hand_range.h"
#include "src/poker_types.h"

namespace poker {

class CFRSolver {
 public:
  struct TraversalStats {
    int64_t cfr_updates = 0;
    int64_t preflop_updates = 0;
    int64_t flop_updates = 0;
    int64_t turn_updates = 0;
    int64_t river_updates = 0;
    int max_decision_depth = 0;
    int64_t child_nodes_created = 0;
    int64_t chance_samples = 0;
    int64_t terminal_utility_calls = 0;
    int64_t fold_utility_calls = 0;
    int64_t showdown_utility_calls = 0;
    int64_t action_entry_touches = 0;
    int64_t atomic_regret_update_retries = 0;
    int64_t betting_history_transition_hits = 0;
    int64_t betting_history_transition_misses = 0;
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  struct TrainingRunStats {
    bool public_state_prebuild_complete = false;
    bool betting_history_transition_prebuild_complete = false;
    bool action_transition_prebuild_complete = false;
    bool chance_transition_prebuild_complete = false;
    bool info_set_prebuild_complete = false;
    bool private_bucket_prebuild_complete = false;
    bool frozen_info_set_lookup_prebuild_complete = false;
    int64_t prebuild_public_states = 0;
    int64_t prebuild_betting_histories = 0;
    int64_t prebuild_betting_history_transitions = 0;
    int64_t missing_betting_history_transitions = 0;
    int64_t prebuild_action_transitions = 0;
    int64_t missing_action_transitions = 0;
    int64_t prebuild_chance_transitions = 0;
    int64_t missing_chance_transitions = 0;
    int64_t prebuild_info_sets = 0;
    int64_t prebuild_action_entries = 0;
    int64_t prebuild_private_bucket_rows = 0;
    int64_t prebuild_frozen_info_set_lookup_rows = 0;
    double prebuild_seconds = 0.0;
    double info_set_prebuild_seconds = 0.0;
    int warmup_iterations = 0;
    int frozen_iterations = 0;
    double warmup_seconds = 0.0;
    double frozen_seconds = 0.0;
    int64_t warmup_cfr_updates = 0;
    int64_t frozen_cfr_updates = 0;
  };

  CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config, const GameState& initial_state);
  ~CFRSolver();

  CFRSolver(const CFRSolver&) = delete;
  CFRSolver& operator=(const CFRSolver&) = delete;
  CFRSolver(CFRSolver&&) noexcept;
  CFRSolver& operator=(CFRSolver&&) noexcept;

  void run(int iterations, ComboId player_a_hand, ComboId player_b_hand);
  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);

  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);

  double get_expected_value(int player_id) const;
  int get_iterations_run() const;
  int64_t get_cfr_update_count() const;
  size_t get_info_set_count() const;
  size_t get_public_state_count() const;
  TraversalStats get_traversal_stats() const;
  TrainingRunStats get_last_training_run_stats() const;
  void add_traversal_stats(const TraversalStats& stats);
  UtilityCacheStats get_utility_cache_stats() const;
  static bool traversal_stats_enabled();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace poker
