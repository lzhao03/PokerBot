#include "src/cfr_solver.h"
#include "src/combo.h"

#include "doctest/doctest.h"

#include <cmath>

namespace poker {

struct CFRSolverTestAccess {
  static const StrategyTables& tables(const CFRSolver& solver) {
    return solver.tables();
  }

  static PrivateInfoSetId private_id(CFRSolver& solver,
                                     ComboId hand,
                                     const BoardRunout& board,
                                     NodeId node_id) {
    return solver.private_info_set_id_for_runout(hand, board, node_id);
  }
};

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

SolverConfig Config() {
  SolverConfig config;
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.max_depth = 2;
  config.chance_samples = 1;
  config.regret_only_training = true;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 2;
  return config;
}

TEST_CASE("coarse public and private abstractions require legacy opt-in") {
  const SolverConfig config = Config();
  CHECK_THROWS_WITH_AS(
      CFRSolver{config},
      "coarse public + coarse private abstraction does not provide "
      "exhaustive history-aware private observation support",
      std::invalid_argument);
}

TEST_CASE("legacy coarse private identity uses the current street bucket") {
  SolverConfig config = Config();
  config.recall_policy = RecallPolicy::kAllowLegacyImperfectRecall;
  CFRSolver solver(config);
  const HandRange a = R(H(14, SuitKind::kHearts, 13, SuitKind::kHearts));
  const HandRange b = R(H(12, SuitKind::kClubs, 11, SuitKind::kClubs));

  solver.run(4, a, b);
  const auto first = solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 4);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(first.public_state_prebuild_complete);
  CHECK(first.info_set_prebuild_complete);
  CHECK(first.frozen_info_set_lookup_prebuild_complete);
  CHECK(first.warmup_iterations == 0);
  CHECK(first.frozen_iterations == 4);
  CHECK(first.missing_action_transitions == 0);
  CHECK(first.missing_chance_transitions == 0);

  const StrategyTables& tables = CFRSolverTestAccess::tables(solver);
  const NodeId root_id = tables.root_node_id;
  REQUIRE(root_id < tables.nodes.size());
  const ComboId hand = H(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const PrivateInfoSetId private_id = CFRSolverTestAccess::private_id(
      solver, hand, BoardRunout::Preflop(), root_id);
  CHECK(private_id == observe_private_street(
                          hand, StreetKind::kPreflop,
                          BoardRunout::Preflop()).value);
  CHECK(private_id != initial_private_observation(hand));
}

}  // namespace
}  // namespace poker
