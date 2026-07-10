#include "src/cfr_solver.h"

#include "src/combo.h"
#include "doctest/doctest.h"

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "cfr_board_bucket_test must be compiled with coarse public buckets"
#endif

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

SolverConfig CoarseConfig() {
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
  return config;
}

TEST_CASE("coarse training reaches fixed-storage phase") {
  SolverConfig config = CoarseConfig();
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        13, SuitKind::kHearts));
  HandRange player_b = ExactRange(Combo(12, SuitKind::kClubs,
                                        11, SuitKind::kClubs));

  solver.run(4, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
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
  const CFRSolver::TrainingRunStats second_stats =
      solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 6);
  CHECK(second_stats.warmup_iterations == 0);
  CHECK(second_stats.frozen_iterations == 2);
}

}  // namespace
}  // namespace poker
