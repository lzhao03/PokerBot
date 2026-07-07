#include "src/cfr_solver.h"

#include "src/combo.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

SolverConfig SmallConfig() {
  SolverConfig config;
  config.starting_stack_size = 12;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.chance_samples = 1;
  config.regret_only_training = true;
  return config;
}

HandRange ExactRange(ComboId combo_id) {
  HandRange range;
  range.add_combo(combo_id, 1.0);
  return range;
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

void CheckRangeTrainingUpdatesCounters() {
  SolverConfig config = SmallConfig();
  config.max_depth = 1;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kHearts,
                                        13, SuitKind::kSpades));

  solver.run(3, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(solver.get_iterations_run() == 3, "range run should record iterations");
  Expect(solver.get_cfr_update_count() > 0, "range run should visit CFR nodes");
  Expect(solver.get_info_set_count() > 0, "range run should allocate infosets");
  Expect(stats.warmup_iterations + stats.frozen_iterations == 3,
         "run stats should account for every iteration");
}

void CheckPublicStateCapPreventsFrozenPhase() {
  SolverConfig config = SmallConfig();
  config.max_public_states = 1;
  config.max_info_sets = 500000;
  config.warmup_iterations = 1;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kDiamonds,
                                        12, SuitKind::kDiamonds));
  HandRange player_b = ExactRange(Combo(11, SuitKind::kSpades,
                                        10, SuitKind::kSpades));

  solver.run(5, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(!stats.public_state_prebuild_complete,
         "public-state cap should stop prebuild completion");
  Expect(stats.frozen_iterations == 0,
         "incomplete prebuild should skip frozen training");
  Expect(solver.get_public_state_count() <= 1,
         "public-state cap should bound allocation");
}

void CheckEvaluateStrategyReturnsFiniteValue() {
  SolverConfig config = SmallConfig();
  config.max_depth = 1;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        2, SuitKind::kHearts));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kClubs,
                                        2, SuitKind::kClubs));

  solver.run(3, player_a, player_b);
  const double exact_value = solver.evaluate_strategy(
      Combo(14, SuitKind::kHearts, 2, SuitKind::kHearts),
      Combo(13, SuitKind::kClubs, 2, SuitKind::kClubs));
  const double range_value = solver.evaluate_strategy(3, player_a, player_b);
  Expect(std::isfinite(exact_value), "exact evaluation should be finite");
  Expect(std::isfinite(range_value), "range evaluation should be finite");
}

void CheckFixedTerminalHandUtility() {
  SolverConfig config = SmallConfig();
  GameState terminal;
  terminal.stack[0] = 0;
  terminal.stack[1] = 0;
  terminal.pot = 10;
  terminal.street = StreetKind::kRiver;
  terminal.player_to_act = -1;
  terminal.folded_player = 1;
  terminal.player_contribution = {5, 5};
  terminal.player_contribution_count = 2;

  CFRSolver solver(config, terminal);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kHearts,
                                        13, SuitKind::kSpades));
  solver.run(1, player_a, player_b);
  Expect(solver.get_iterations_run() == 1,
         "fixed terminal run should still count the iteration");
  Expect(solver.get_expected_value(0) == 5.0,
         "fold utility should award half the pot to player A");
}

void CheckInvalidRangesAreRejected() {
  SolverConfig config = SmallConfig();
  CFRSolver solver(config);
  HandRange empty;
  HandRange player_b = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));

  bool threw = false;
  try {
    solver.run(1, empty, player_b);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "empty ranges should be rejected");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckRangeTrainingUpdatesCounters();
  poker::CheckPublicStateCapPreventsFrozenPhase();
  poker::CheckEvaluateStrategyReturnsFiniteValue();
  poker::CheckFixedTerminalHandUtility();
  poker::CheckInvalidRangesAreRejected();
  return 0;
}
