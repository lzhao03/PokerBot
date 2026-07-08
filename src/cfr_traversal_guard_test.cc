#include "src/cfr_solver.h"

#include "src/combo.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace poker {
namespace {

struct Metrics {
  double value = 0.0;
  int64_t info_sets = 0;
  int64_t public_states = 0;
  int64_t cfr_updates = 0;
  int64_t action_entry_touches = 0;
};

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

void ExpectMetrics(const Metrics& actual,
                   const Metrics& expected,
                   const char* name) {
  constexpr double kTolerance = 1e-9;
  Expect(std::isfinite(actual.value), name);
  const bool ok =
      std::fabs(actual.value - expected.value) <= kTolerance &&
      actual.info_sets == expected.info_sets &&
      actual.public_states == expected.public_states &&
      actual.cfr_updates == expected.cfr_updates &&
      actual.action_entry_touches == expected.action_entry_touches;
  if (ok) {
    return;
  }

  std::cerr << std::setprecision(17)
            << "FAILED: " << name << "\n"
            << "  actual value=" << actual.value
            << " info_sets=" << actual.info_sets
            << " public_states=" << actual.public_states
            << " cfr_updates=" << actual.cfr_updates
            << " action_entry_touches=" << actual.action_entry_touches
            << "\n"
            << "  expected value=" << expected.value
            << " info_sets=" << expected.info_sets
            << " public_states=" << expected.public_states
            << " cfr_updates=" << expected.cfr_updates
            << " action_entry_touches=" << expected.action_entry_touches
            << "\n";
  std::exit(1);
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

HandRange ExactRange(ComboId combo_id) {
  HandRange range;
  range.add_combo(combo_id, 1.0);
  return range;
}

SolverConfig BaseConfig() {
  SolverConfig config;
  config.starting_stack_size = 12;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.chance_samples = 1;
  return config;
}

Metrics TrainingMetrics(const CFRSolver& solver) {
  const CFRSolver::TraversalStats stats = solver.get_traversal_stats();
  return Metrics{
      solver.get_expected_value(0),
      static_cast<int64_t>(solver.get_info_set_count()),
      static_cast<int64_t>(solver.get_public_state_count()),
      solver.get_cfr_update_count(),
      stats.action_entry_touches,
  };
}

#if POKER_COARSE_PUBLIC_BUCKETS

void CheckFrozenRegretOnlyGuard() {
  SolverConfig config = BaseConfig();
  config.starting_stack_size = 18;
  config.bet_sizes = {0.5, 1.0};
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.warmup_iterations = 1;
  config.num_training_threads = 1;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        13, SuitKind::kHearts));
  HandRange player_b = ExactRange(Combo(12, SuitKind::kClubs,
                                        11, SuitKind::kClubs));

  solver.run(4, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.frozen_iterations == 3,
         "frozen guard should run frozen iterations");
  Expect(stats.frozen_cfr_updates > stats.frozen_iterations,
         "frozen guard should do real CFR work");
  ExpectMetrics(TrainingMetrics(solver),
                Metrics{1.0997248478223594, 822, 2870, 120, 2482},
                "frozen regret-only traversal");
}

#else

HandRange TwoComboRange(ComboId first, ComboId second) {
  HandRange range;
  range.add_combo(first, 1.0);
  range.add_combo(second, 1.0);
  return range;
}

void CheckWarmupFullCfrGuard() {
  SolverConfig config = BaseConfig();
  config.max_depth = 1;
  config.regret_only_training = false;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kHearts,
                                        13, SuitKind::kSpades));

  solver.run(3, player_a, player_b);
  ExpectMetrics(TrainingMetrics(solver),
                Metrics{-1.0 / 12.0, 1, 5, 3, 60},
                "warmup full CFR traversal");
}

void CheckEvaluateStrategyGuard() {
  SolverConfig config = BaseConfig();
  config.max_depth = 1;
  config.regret_only_training = true;
  CFRSolver solver(config);
  const ComboId player_a_combo = Combo(14, SuitKind::kHearts,
                                       2, SuitKind::kHearts);
  const ComboId player_b_combo = Combo(13, SuitKind::kClubs,
                                       2, SuitKind::kClubs);
  HandRange player_a = ExactRange(player_a_combo);
  HandRange player_b = ExactRange(player_b_combo);

  solver.run(3, player_a, player_b);
  const int64_t before_touches =
      solver.get_traversal_stats().action_entry_touches;
  const double value =
      solver.evaluate_strategy(player_a_combo, player_b_combo);
  const int64_t after_touches =
      solver.get_traversal_stats().action_entry_touches;
  ExpectMetrics(Metrics{
                    value,
                    static_cast<int64_t>(solver.get_info_set_count()),
                    static_cast<int64_t>(solver.get_public_state_count()),
                    solver.get_cfr_update_count(),
                    after_touches - before_touches,
                },
                Metrics{0.046839674465274539, 1, 232, 3, 4},
                "evaluate strategy traversal");
}

void CheckMaxDepthRangeConditionedGuard() {
  SolverConfig config = BaseConfig();
  config.max_depth = 2;
  config.regret_only_training = true;
  CFRSolver solver(config);
  HandRange player_a = TwoComboRange(
      Combo(14, SuitKind::kHearts, 14, SuitKind::kSpades),
      Combo(13, SuitKind::kHearts, 13, SuitKind::kSpades));
  HandRange player_b = TwoComboRange(
      Combo(12, SuitKind::kClubs, 12, SuitKind::kDiamonds),
      Combo(11, SuitKind::kClubs, 11, SuitKind::kDiamonds));

  solver.run(4, player_a, player_b);
  ExpectMetrics(TrainingMetrics(solver),
                Metrics{4.4166666666666661, 4, 33, 16, 132},
                "max-depth range-conditioned traversal");
}

#endif

}  // namespace
}  // namespace poker

int main() {
#if POKER_COARSE_PUBLIC_BUCKETS
  poker::CheckFrozenRegretOnlyGuard();
#else
  poker::CheckWarmupFullCfrGuard();
  poker::CheckEvaluateStrategyGuard();
  poker::CheckMaxDepthRangeConditionedGuard();
#endif
  return 0;
}
