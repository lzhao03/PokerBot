#include "src/cfr_solver.h"

#include "src/combo.h"
#include "doctest/doctest.h"

#include <stdexcept>

namespace poker {
namespace {

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

TEST_CASE("public-state cap prevents fixed-storage training") {
  SolverConfig config = SmallConfig();
  config.max_public_states = 1;
  config.max_info_sets = 500000;
  CFRSolver solver(config);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kDiamonds,
                                        12, SuitKind::kDiamonds));
  HandRange player_b = ExactRange(Combo(11, SuitKind::kSpades,
                                        10, SuitKind::kSpades));

  solver.run(5, player_a, player_b);
  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  CHECK(!stats.public_state_prebuild_complete);
  CHECK(stats.warmup_iterations == 5);
  CHECK(stats.frozen_iterations == 0);
  CHECK(solver.get_public_state_count() <= 1);
}

TEST_CASE("fixed terminal run counts iteration and utility") {
  SolverConfig config = SmallConfig();
  ExactGameState terminal;
  terminal.betting.stack = {0, 0};
  terminal.betting.street = StreetKind::kRiver;
  terminal.betting.player_to_act = -1;
  terminal.betting.folded_player = 1;
  terminal.betting.committed = {5, 5};

  CFRSolver solver(config, terminal);
  HandRange player_a = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));
  HandRange player_b = ExactRange(Combo(13, SuitKind::kHearts,
                                        13, SuitKind::kSpades));
  solver.run(1, player_a, player_b);
  CHECK(solver.get_iterations_run() == 1);
  CHECK(solver.get_expected_value(0) == doctest::Approx(5.0));
}

TEST_CASE("empty ranges are rejected") {
  SolverConfig config = SmallConfig();
  CFRSolver solver(config);
  HandRange empty;
  HandRange player_b = ExactRange(Combo(14, SuitKind::kHearts,
                                        14, SuitKind::kSpades));

  CHECK_THROWS_AS(solver.run(1, empty, player_b), std::invalid_argument);
}

TEST_CASE("invalid initial betting state is rejected") {
  SolverConfig config = SmallConfig();
  ExactGameState invalid;
  invalid.betting.stack[0] = -1;

  CHECK_THROWS_AS(CFRSolver(config, invalid), std::invalid_argument);
}

}  // namespace
}  // namespace poker
