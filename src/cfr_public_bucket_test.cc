#include "src/cfr_solver.h"

#include "src/combo.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "cfr_public_bucket_test must be compiled with coarse public buckets"
#endif

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
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

SolverConfig CoarseConfig() {
  SolverConfig config;
  config.starting_stack_size = 18;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {0.5, 1.0};
  config.max_depth = 0;
  config.chance_samples = 1;
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.warmup_iterations = 1;
  config.num_training_threads = 1;
  return config;
}

void CheckCoarseTrainingReachesFrozenPhase() {
  SolverConfig config = CoarseConfig();
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        13, SuitKind::kHearts));
  HandRange player_b = ExactRange(Combo(12, SuitKind::kClubs,
                                        11, SuitKind::kClubs));

  solver.run(4, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.public_state_prebuild_complete,
         "coarse public states should prebuild");
  Expect(stats.betting_history_transition_prebuild_complete,
         "coarse betting transitions should validate");
  Expect(stats.action_transition_prebuild_complete,
         "coarse action transitions should validate");
  Expect(stats.chance_transition_prebuild_complete,
         "coarse chance transitions should validate");
  Expect(stats.info_set_prebuild_complete,
         "coarse infosets should prebuild");
  Expect(stats.private_bucket_prebuild_complete,
         "coarse private bucket rows should prebuild");
  Expect(stats.frozen_info_set_lookup_prebuild_complete,
         "coarse frozen lookups should prebuild");
  Expect(stats.frozen_iterations == 3,
         "coarse run should use frozen iterations after warmup");
  Expect(stats.frozen_cfr_updates > stats.frozen_iterations,
         "full-depth coarse frozen traversal should do real work");
  Expect(stats.missing_betting_history_transitions == 0,
         "coarse betting validation should have no misses");
  Expect(stats.missing_action_transitions == 0,
         "coarse action validation should have no misses");
  Expect(stats.missing_chance_transitions == 0,
         "coarse chance validation should have no misses");
}

void CheckCoarseEvaluationIsFinite() {
  SolverConfig config = CoarseConfig();
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kDiamonds,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kDiamonds,
                                        13, SuitKind::kSpades));

  solver.run(3, player_a, player_b);
  const double value = solver.evaluate_strategy(3, player_a, player_b);
  Expect(std::isfinite(value), "coarse evaluation should be finite");
  Expect(solver.get_public_state_count() > 0,
         "coarse run should build public states");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckCoarseTrainingReachesFrozenPhase();
  poker::CheckCoarseEvaluationIsFinite();
  return 0;
}
