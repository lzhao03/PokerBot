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
  REQUIRE(mutable_identity.cumulative != nullptr);

  CFRSolverTestAccess::freeze(solver);
  const auto frozen_identity =
      CFRSolverTestAccess::storage_identity(solver);
  CHECK(frozen_identity.mutable_tables == nullptr);
  CHECK(frozen_identity.frozen_tables == mutable_identity.mutable_tables);
  CHECK(frozen_identity.cumulative == mutable_identity.cumulative);
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
  CHECK(after_identity.cumulative == frozen_identity.cumulative);
  CHECK(after.node_count == before.node_count);
  CHECK(after.info_set_count == before.info_set_count);
  CHECK(after.action_transition_count == before.action_transition_count);
  CHECK(after.chance_transition_count == before.chance_transition_count);
  CHECK(after.regrets != before.regrets);
  CHECK(after.strategy_sums != before.strategy_sums);
}

}  // namespace
}  // namespace poker
