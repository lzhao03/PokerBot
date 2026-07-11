#include "tests/compiled_solver_test_support.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

TEST_CASE("freezing transfers topology and retains cumulative storage") {
  const SolverConfig config = test::CompiledConfig();
  const HandRange a_range = test::ExactRange(test::kPlayerAHand);
  const HandRange b_range = test::ExactRange(test::kPlayerBHand);
  CFRSolver solver(config, test::TurnRoot());
  CFRSolverTestAccess::prepare(solver, a_range, b_range);

  const auto mutable_identity =
      CFRSolverTestAccess::storage_identity(solver);
  REQUIRE(mutable_identity.mutable_tables != nullptr);
  CHECK(mutable_identity.frozen_tables == mutable_identity.mutable_tables);
  REQUIRE(mutable_identity.cfr_state != nullptr);

  CFRSolverTestAccess::freeze(solver);
  const auto frozen_identity =
      CFRSolverTestAccess::storage_identity(solver);
  CHECK(frozen_identity.mutable_tables == nullptr);
  CHECK(frozen_identity.frozen_tables == mutable_identity.mutable_tables);
  CHECK(frozen_identity.cfr_state == mutable_identity.cfr_state);
  const auto before = CFRSolverTestAccess::snapshot(
      solver, test::kPlayerAHand, test::kPlayerBHand);

  CFRSolverTestAccess::run_fixed(
      solver, test::kTrainingIterations, test::kTrainingSeed, a_range,
      b_range);
  const auto after_identity =
      CFRSolverTestAccess::storage_identity(solver);
  const auto after = CFRSolverTestAccess::snapshot(
      solver, test::kPlayerAHand, test::kPlayerBHand);
  CHECK(after_identity.mutable_tables == nullptr);
  CHECK(after_identity.frozen_tables == frozen_identity.frozen_tables);
  CHECK(after_identity.cfr_state == frozen_identity.cfr_state);
  CHECK(after.node_count == before.node_count);
  CHECK(after.info_set_count == before.info_set_count);
  CHECK(after.action_transition_count == before.action_transition_count);
  CHECK(after.chance_transition_count == before.chance_transition_count);
  CHECK(after.regrets != before.regrets);
  CHECK(after.strategy_sums != before.strategy_sums);
}

TEST_CASE("CFR states sharing compiled tables remain independent") {
  const SolverConfig config = test::CompiledConfig();
  const ExactPublicState root = test::TurnRoot();
  const HandRange a_range = test::ExactRange(test::kPlayerAHand);
  const HandRange b_range = test::ExactRange(test::kPlayerBHand);
  CFRSolver compiled(config, root);
  CFRSolverTestAccess::prepare(compiled, a_range, b_range);
  CFRSolverTestAccess::freeze(compiled);

  auto first_state = CFRSolverTestAccess::copy_cfr_state(compiled);
  auto second_state = CFRSolverTestAccess::copy_cfr_state(compiled);
  CHECK(first_state->regret_sum == second_state->regret_sum);
  CHECK(first_state->strategy_sum == second_state->strategy_sum);
  CHECK(first_state->iterations == second_state->iterations);
  CHECK(first_state->cumulative_root_utility ==
        second_state->cumulative_root_utility);

  CFRSolver first(config, root);
  CFRSolver second(config, root);
  CFRSolverTestAccess::bind_frozen(first, compiled, first_state);
  CFRSolverTestAccess::bind_frozen(second, compiled, second_state);
  const auto first_identity = CFRSolverTestAccess::storage_identity(first);
  const auto second_identity = CFRSolverTestAccess::storage_identity(second);
  CHECK(first_identity.frozen_tables == second_identity.frozen_tables);
  CHECK(first_identity.cfr_state != second_identity.cfr_state);

  CFRSolverTestAccess::run_fixed(
      first, test::kTrainingIterations, test::kTrainingSeed, a_range,
      b_range);
  CHECK(first_state->regret_sum != second_state->regret_sum);
  CHECK(first_state->strategy_sum != second_state->strategy_sum);
  CHECK(first_state->iterations == test::kTrainingIterations);
  CHECK(second_state->iterations == 0);
  CHECK(first_state->cumulative_root_utility != 0.0);
  CHECK(second_state->cumulative_root_utility == 0.0);
}

}  // namespace
}  // namespace poker
