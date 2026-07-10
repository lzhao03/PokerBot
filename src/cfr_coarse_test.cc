#include "src/cfr_solver.h"
#include "src/combo.h"

#include "doctest/doctest.h"

#include <cmath>

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "cfr_coarse_test must be compiled with coarse public buckets"
#endif

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

HandRange ExactRange(ComboId combo) {
  HandRange range;
  range.add_combo(combo, 1.0);
  return range;
}

TEST_CASE("coarse training prebuilds and stays in fixed storage") {
  SolverConfig config;
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.max_depth = 0;
  config.chance_samples = 1;
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 1;

  CFRSolver solver(config);
  const HandRange player_a = ExactRange(
      Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts));
  const HandRange player_b = ExactRange(
      Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs));
  solver.run(4, player_a, player_b);

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 4);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(stats.public_state_prebuild_complete);
  CHECK(stats.action_transition_prebuild_complete);
  CHECK(stats.chance_transition_prebuild_complete);
  CHECK(stats.info_set_prebuild_complete);
  CHECK(stats.frozen_info_set_lookup_prebuild_complete);
  CHECK(stats.warmup_iterations == 0);
  CHECK(stats.frozen_iterations == 4);
  CHECK(stats.frozen_cfr_updates > stats.frozen_iterations);
  CHECK(stats.missing_action_transitions == 0);
  CHECK(stats.missing_chance_transitions == 0);

  solver.run(2, player_a, player_b);
  const CFRSolver::TrainingRunStats second =
      solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 6);
  CHECK(second.warmup_iterations == 0);
  CHECK(second.frozen_iterations == 2);
}

}  // namespace
}  // namespace poker
