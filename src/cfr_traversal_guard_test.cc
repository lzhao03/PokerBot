#include "src/cfr_solver.h"

#include "src/combo.h"
#include "doctest/doctest.h"

#include <cmath>

namespace poker {
namespace {

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
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_expected_value(0) == doctest::Approx(-1.0 / 12.0));
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(solver.get_traversal_stats().max_decision_depth <= config.max_depth);
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
  const CFRSolver::TrainingRunStats run_stats =
      solver.get_last_training_run_stats();
  const CFRSolver::TraversalStats traversal_stats =
      solver.get_traversal_stats();
  CHECK(solver.get_iterations_run() == 4);
  CHECK(run_stats.warmup_iterations == 4);
  CHECK(run_stats.frozen_iterations == 0);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(traversal_stats.max_decision_depth == config.max_depth - 1);
  CHECK(traversal_stats.max_decision_depth < config.max_depth);
}

}  // namespace
}  // namespace poker
