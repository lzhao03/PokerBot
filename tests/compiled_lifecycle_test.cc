#include "tests/compiled_solver_test_support.h"

#include <cmath>

#include "doctest/doctest.h"

namespace poker {
namespace {

TEST_CASE("compiled solver topology remains fixed during training") {
  const SolverConfig config = test::CompiledConfig();
  const ExactPublicState state = test::TurnRoot();
  const HandRange a_range = test::ExactRange(test::kPlayerAHand);
  const HandRange b_range = test::ExactRange(test::kPlayerBHand);
  CFRSolver solver(config, state);
  CFRSolverTestAccess::prepare(solver, a_range, b_range);

  const auto prebuild = solver.get_last_training_run_stats();
  CHECK(prebuild.public_state_prebuild_complete);
  CHECK(prebuild.action_transition_prebuild_complete);
  CHECK(prebuild.chance_transition_prebuild_complete);
  CHECK(prebuild.info_set_prebuild_complete);
  CHECK(prebuild.frozen_info_set_lookup_prebuild_complete);
  CHECK(prebuild.missing_action_transitions == 0);
  CHECK(prebuild.missing_chance_transitions == 0);

  CFRSolverTestAccess::freeze(solver);
  const auto before = CFRSolverTestAccess::snapshot(
      solver, test::kPlayerAHand, test::kPlayerBHand);
  CFRSolverTestAccess::seed(solver, test::kTrainingSeed);
  solver.run(test::kTrainingIterations, a_range, b_range);
  const auto after = CFRSolverTestAccess::snapshot(
      solver, test::kPlayerAHand, test::kPlayerBHand);
  const auto near = [](double value) {
    return doctest::Approx(value).epsilon(1e-4);
  };

  CHECK(after.root_node_id == before.root_node_id);
  CHECK(after.node_count == before.node_count);
  CHECK(after.betting_node_count == before.betting_node_count);
  CHECK(after.action_transition_count == before.action_transition_count);
  CHECK(after.chance_transition_count == before.chance_transition_count);
  CHECK(after.info_set_count == before.info_set_count);
  CHECK(after.action_entry_count == before.action_entry_count);
  CHECK(after.root_actions == before.root_actions);
  CHECK(after.selected_info_set_offsets == before.selected_info_set_offsets);
  CHECK(solver.get_iterations_run() == test::kTrainingIterations);
  CHECK(solver.get_last_training_run_stats().frozen_iterations ==
        test::kTrainingIterations);
  CHECK(std::isfinite(solver.get_expected_value(0)));

  CHECK(after.root_node_id == 0);
  CHECK(after.node_count == 2085);
  CHECK(after.betting_node_count == 64);
  CHECK(after.action_transition_count == 1748);
  CHECK(after.chance_transition_count == 336);
  CHECK(after.info_set_count == 744);
  CHECK(after.action_entry_count == 11890);
  const std::vector<GameAction> expected_actions = {
      {ActionKind::kCheck, 0},
      {ActionKind::kBet, 4},
      {ActionKind::kAllIn, 6},
  };
  CHECK(after.root_actions == expected_actions);
  const std::vector<test::InfoSetOffset> expected_offsets = {
      {0, 960, 0},
      {1, 584, 16},
      {2, 584, 32},
      {3, 584, 48},
  };
  CHECK(after.selected_info_set_offsets == expected_offsets);

  CHECK(after.regrets[0] == near(1.5223));
  CHECK(after.regrets[16] == near(1.98499));
  CHECK(after.regrets[17] == near(2.30055));
  CHECK(after.regrets[33] == near(0.23602));
  CHECK(after.regrets[34] == near(1.36963));
  CHECK(after.regrets[49] == near(1.66667));
  CHECK(after.strategy_sums[0] == near(10.5773));
  CHECK(after.strategy_sums[1] == near(0.756052));
  CHECK(after.strategy_sums[2] == near(0.666667));
  CHECK(after.strategy_sums[16] == near(1.8421));
  CHECK(after.strategy_sums[17] == near(3.25657));
  CHECK(after.strategy_sums[18] == near(3.90134));
  CHECK(after.strategy_sums[32] == near(0.333333));
  CHECK(after.strategy_sums[33] == near(1.57677));
  CHECK(after.strategy_sums[34] == near(7.08989));
  CHECK(after.strategy_sums[48] == near(0.5));
  CHECK(after.strategy_sums[49] == near(8.5));
  CHECK(CFRSolverTestAccess::cumulative_utility(solver) ==
        near(21.1626));

  CFRSolverTestAccess::seed(solver, test::kEvaluationSeed);
  const double evaluation = solver.evaluate_strategy(
      test::kPlayerAHand, test::kPlayerBHand);
  CHECK(evaluation == near(3.26835));
}

}  // namespace
}  // namespace poker
