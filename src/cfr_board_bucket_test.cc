#include "src/cfr_solver.h"
#include "src/combo.h"

#include "doctest/doctest.h"

#include <cmath>

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "cfr_board_bucket_test must use coarse public buckets"
#endif

ComboId H(int r0, SuitKind s0, int r1, SuitKind s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

HandRange R(ComboId hand) {
  HandRange range;
  range.add_combo(hand, 1.0);
  return range;
}

TEST_CASE("coarse mode reaches and reuses fixed storage") {
  SolverConfig config;
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.max_depth = 1;
  config.chance_samples = 1;
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 2;
  CFRSolver solver(config);
  const HandRange a = R(H(14, SuitKind::kHearts, 13, SuitKind::kHearts));
  const HandRange b = R(H(12, SuitKind::kClubs, 11, SuitKind::kClubs));

  solver.run(4, a, b);
  const auto first = solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 4);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(first.public_state_prebuild_complete);
  CHECK(first.action_transition_prebuild_complete);
  CHECK(first.chance_transition_prebuild_complete);
  CHECK(first.info_set_prebuild_complete);
  CHECK(first.frozen_info_set_lookup_prebuild_complete);
  CHECK(first.warmup_iterations == 0);
  CHECK(first.frozen_iterations == 4);
  CHECK(first.missing_action_transitions == 0);
  CHECK(first.missing_chance_transitions == 0);

  solver.run(2, a, b);
  const auto second = solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 6);
  CHECK(second.warmup_iterations == 0);
  CHECK(second.frozen_iterations == 2);
}

}  // namespace
}  // namespace poker
