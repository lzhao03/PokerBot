#include "src/build_flags.h"
#include "src/cfr_solver.h"
#include "src/combo.h"
#include "src/game_rules.h"

#include "doctest/doctest.h"

#include <array>
#include <cmath>

namespace poker {
namespace {

#if POKER_COARSE_PUBLIC_BUCKETS == POKER_COARSE_PRIVATE_BUCKETS
#error "recall_policy_test requires one coarse abstraction"
#endif

HandRange Range(int first_rank, int second_rank, SuitKind suit) {
  HandRange range;
  range.add_combo(
      CardsToComboId(MakeCardId(first_rank, suit),
                     MakeCardId(second_rank, suit)),
      1.0);
  return range;
}

TEST_CASE("mixed abstractions support strict prebuilt training") {
  SolverConfig config;
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.max_depth = 1;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 2;
  config.regret_only_training = true;

  const BettingRules rules{config.big_blind};
  ExactPublicState state = MakeInitialState(rules, {8, 8}, {1, 2});
  state.betting = ApplyAction(
      state.betting, {ActionKind::kCall, 2});
  state.betting = ApplyAction(
      state.betting, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      MakeCardId(2, SuitKind::kDiamonds),
      MakeCardId(7, SuitKind::kSpades),
      MakeCardId(9, SuitKind::kDiamonds),
  };
  state = ApplyChance(state, flop, rules);

  CFRSolver solver(config, state);
  solver.run(2, Range(14, 13, SuitKind::kHearts),
             Range(12, 11, SuitKind::kClubs));

  const auto stats = solver.get_last_training_run_stats();
  CHECK(solver.get_iterations_run() == 2);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(stats.public_state_prebuild_complete);
  CHECK(stats.info_set_prebuild_complete);
  CHECK(stats.frozen_info_set_lookup_prebuild_complete);
  CHECK(stats.frozen_iterations == 2);
}

}  // namespace
}  // namespace poker
