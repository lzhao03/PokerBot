#include "src/cfr_solver.h"

#include "src/combo.h"
#include "doctest/doctest.h"

#include <cmath>

namespace poker {
namespace {

struct Metrics {
  double value = 0.0;
  int64_t info_sets = 0;
  int64_t public_states = 0;
  int64_t cfr_updates = 0;
  int64_t action_entry_touches = 0;
};

void CheckMetrics(const Metrics& actual,
                  const Metrics& expected,
                  const char* name) {
  CAPTURE(name);
  CHECK(std::isfinite(actual.value));
  CHECK(actual.value == doctest::Approx(expected.value).epsilon(1e-9));
  CHECK(actual.info_sets == expected.info_sets);
  CHECK(actual.public_states == expected.public_states);
  CHECK(actual.cfr_updates == expected.cfr_updates);
  CHECK(actual.action_entry_touches == expected.action_entry_touches);
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

TEST_CASE("frozen regret-only traversal guard") {
  SolverConfig config = BaseConfig();
  config.starting_stack_size = 18;
  config.bet_sizes = {0.5, 1.0};
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 1;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        13, SuitKind::kHearts));
  HandRange player_b = ExactRange(Combo(12, SuitKind::kClubs,
                                        11, SuitKind::kClubs));

  solver.run(4, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  CHECK(stats.warmup_iterations == 0);
  CHECK(stats.frozen_iterations == 4);
  CHECK(stats.frozen_cfr_updates > stats.frozen_iterations);
  CheckMetrics(TrainingMetrics(solver),
               Metrics{0.66192917573519394, 822, 2870, 120, 2480},
               "frozen regret-only traversal");
}

#else

HandRange TwoComboRange(ComboId first, ComboId second) {
  HandRange range;
  range.add_combo(first, 1.0);
  range.add_combo(second, 1.0);
  return range;
}

TEST_CASE("warmup full CFR traversal guard") {
  SolverConfig config = BaseConfig();
  config.max_depth = 1;
  config.regret_only_training = false;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kHearts,
                                        13, SuitKind::kSpades));

  solver.run(3, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 3);
  CHECK(stats.warmup_iterations == 3);
  CHECK(stats.frozen_iterations == 0);
  CheckMetrics(TrainingMetrics(solver),
               Metrics{-1.0 / 12.0, 1, 5, 3, 60},
               "warmup full CFR traversal");
}

TEST_CASE("evaluate strategy traversal guard") {
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
  CheckMetrics(Metrics{
                   value,
                   static_cast<int64_t>(solver.get_info_set_count()),
                   static_cast<int64_t>(solver.get_public_state_count()),
                   solver.get_cfr_update_count(),
                   after_touches - before_touches,
               },
               Metrics{0.046839674465274539, 1, 232, 3, 4},
               "evaluate strategy traversal");
}

TEST_CASE("max-depth range-conditioned traversal guard") {
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
  CheckMetrics(TrainingMetrics(solver),
               Metrics{4.4166666666666661, 4, 33, 16, 132},
               "max-depth range-conditioned traversal");
}

#endif

}  // namespace
}  // namespace poker
