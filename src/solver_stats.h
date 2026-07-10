#pragma once

#include <algorithm>
#include <cstdint>

#include "src/build_flags.h"
#include "src/poker_types.h"

namespace poker {

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
  int64_t transition_hits = 0;
  int64_t transition_misses = 0;

  void add(const TraversalStats& other) {
    cfr_updates += other.cfr_updates;
    preflop_updates += other.preflop_updates;
    flop_updates += other.flop_updates;
    turn_updates += other.turn_updates;
    river_updates += other.river_updates;
    child_nodes_created += other.child_nodes_created;
    chance_samples += other.chance_samples;
    terminal_utility_calls += other.terminal_utility_calls;
    fold_utility_calls += other.fold_utility_calls;
    showdown_utility_calls += other.showdown_utility_calls;
    action_entry_touches += other.action_entry_touches;
    atomic_regret_update_retries += other.atomic_regret_update_retries;
    transition_hits += other.transition_hits;
    transition_misses += other.transition_misses;
    max_decision_depth =
        std::max(max_decision_depth, other.max_decision_depth);
  }

  void record_action_entries(int64_t count = 1) {
    if constexpr (kTraversalStatsEnabled) {
      action_entry_touches += count;
    }
  }

  void record_decision(StreetKind street, int depth) {
    if constexpr (kTraversalStatsEnabled) {
      ++cfr_updates;
      max_decision_depth = std::max(max_decision_depth, depth);
      switch (street) {
        case StreetKind::kPreflop:
          ++preflop_updates;
          break;
        case StreetKind::kFlop:
          ++flop_updates;
          break;
        case StreetKind::kTurn:
          ++turn_updates;
          break;
        case StreetKind::kRiver:
          ++river_updates;
          break;
      }
    }
  }

  void record_chance_samples(int64_t count) {
    if constexpr (kTraversalStatsEnabled) {
      chance_samples += count;
    }
  }

  void record_terminal(bool showdown) {
    if constexpr (kTraversalStatsEnabled) {
      ++terminal_utility_calls;
      if (showdown) {
        ++showdown_utility_calls;
      } else {
        ++fold_utility_calls;
      }
    }
  }

  void record_child_node_created() {
    if constexpr (kTraversalStatsEnabled) {
      ++child_nodes_created;
    }
  }

  void record_transition_hit() {
    if constexpr (kTraversalStatsEnabled) {
      ++transition_hits;
    }
  }

  void record_transition_miss() {
    if constexpr (kTraversalStatsEnabled) {
      ++transition_misses;
    }
  }

  void record_atomic_retries(int64_t count) {
    if constexpr (kCasRetryStatsEnabled) {
      atomic_regret_update_retries += count;
    }
  }
};

struct TrainingRunStats {
  bool public_state_prebuild_complete = false;
  bool action_transition_prebuild_complete = false;
  bool chance_transition_prebuild_complete = false;
  bool info_set_prebuild_complete = false;
  bool frozen_info_set_lookup_prebuild_complete = false;
  int64_t prebuild_public_states = 0;
  int64_t prebuild_action_transitions = 0;
  int64_t missing_action_transitions = 0;
  int64_t prebuild_chance_transitions = 0;
  int64_t missing_chance_transitions = 0;
  int64_t prebuild_info_sets = 0;
  int64_t prebuild_action_entries = 0;
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

}  // namespace poker
