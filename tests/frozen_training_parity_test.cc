#include "tests/compiled_solver_test_support.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

TEST_CASE("complete mutable and frozen traversal produce identical updates") {
  const SolverConfig config = test::CompiledConfig();
  const ExactPublicState state = test::TurnRoot();
  const HandRange a_range = test::ExactRange(test::kPlayerAHand);
  const HandRange b_range = test::ExactRange(test::kPlayerBHand);
  CFRSolver growing(config, state);
  CFRSolver frozen(config, state);
  CFRSolverTestAccess::prepare(growing, a_range, b_range);
  CFRSolverTestAccess::prepare(frozen, a_range, b_range);
  CFRSolverTestAccess::freeze(frozen);

  const uint32_t worker_seed =
      CFRSolverTestAccess::first_worker_seed(test::kTrainingSeed);
  CFRSolverTestAccess::run_growing(
      growing, test::kTrainingIterations, worker_seed, a_range, b_range);
  CFRSolverTestAccess::run_fixed(
      frozen, test::kTrainingIterations, test::kTrainingSeed, a_range,
      b_range);

  const auto growing_snapshot = CFRSolverTestAccess::snapshot(
      growing, test::kPlayerAHand, test::kPlayerBHand);
  const auto frozen_snapshot = CFRSolverTestAccess::snapshot(
      frozen, test::kPlayerAHand, test::kPlayerBHand);
  CHECK(growing_snapshot.root_node_id == frozen_snapshot.root_node_id);
  CHECK(growing_snapshot.node_count == frozen_snapshot.node_count);
  CHECK(growing_snapshot.betting_node_count ==
        frozen_snapshot.betting_node_count);
  CHECK(growing_snapshot.action_transition_count ==
        frozen_snapshot.action_transition_count);
  CHECK(growing_snapshot.chance_transition_count ==
        frozen_snapshot.chance_transition_count);
  CHECK(growing_snapshot.info_set_count == frozen_snapshot.info_set_count);
  CHECK(growing_snapshot.selected_info_set_offsets ==
        frozen_snapshot.selected_info_set_offsets);
  CHECK(growing_snapshot.regrets == frozen_snapshot.regrets);
  CHECK(growing_snapshot.strategy_sums == frozen_snapshot.strategy_sums);
  CHECK(CFRSolverTestAccess::cumulative_utility(growing) ==
        CFRSolverTestAccess::cumulative_utility(frozen));
  CHECK(growing.get_cfr_update_count() == frozen.get_cfr_update_count());
}

}  // namespace
}  // namespace poker
