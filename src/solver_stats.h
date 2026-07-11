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
  int64_t chance_samples = 0;
  int64_t terminal_utility_calls = 0;
  int64_t fold_utility_calls = 0;
  int64_t showdown_utility_calls = 0;
  int64_t action_entry_touches = 0;

  void add(const TraversalStats& other) {
    cfr_updates += other.cfr_updates;
    preflop_updates += other.preflop_updates;
    flop_updates += other.flop_updates;
    turn_updates += other.turn_updates;
    river_updates += other.river_updates;
    chance_samples += other.chance_samples;
    terminal_utility_calls += other.terminal_utility_calls;
    fold_utility_calls += other.fold_utility_calls;
    showdown_utility_calls += other.showdown_utility_calls;
    action_entry_touches += other.action_entry_touches;
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

};

struct TrainingRunStats {
  uint64_t iterations = 0;
  int64_t decision_visits = 0;
  int64_t chance_samples = 0;
  int64_t terminal_visits = 0;
  double seconds = 0.0;
};

}  // namespace poker
