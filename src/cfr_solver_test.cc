#include "src/cfr_solver.h"
#include "src/combo.h"
#include "src/strategy_store.h"

#include "doctest/doctest.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace poker {
namespace {

using S = SuitKind;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

HandRange R(ComboId hand) {
  HandRange range;
  range.add_combo(hand, 1.0);
  return range;
}

SolverConfig Config() {
  SolverConfig config;
  config.starting_stack_size = 12;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.chance_samples = 1;
  config.max_depth = 2;
  config.max_info_sets = 500000;
  config.max_public_states = 200000;
  config.num_training_threads = 1;
  return config;
}

const ComboId kA = H(14, S::kHearts, 14, S::kSpades);
const ComboId kB = H(13, S::kClubs, 13, S::kDiamonds);

TEST_CASE("solver lifecycle respects training, limits, and root contracts") {
  SUBCASE("evaluation does not update training state") {
    SolverConfig config = Config();
    config.regret_only_training = false;
    CFRSolver solver(config);
    solver.run(4, R(kA), R(kB));
    const auto run = solver.get_last_training_run_stats();
    CHECK(solver.get_iterations_run() == 4);
    CHECK(run.warmup_iterations + run.frozen_iterations == 4);
    CHECK(std::isfinite(solver.get_expected_value(0)));
    CHECK(solver.get_cfr_update_count() > 0);
    CHECK(solver.get_traversal_stats().max_decision_depth <= config.max_depth);

    const int iterations = solver.get_iterations_run();
    const size_t info_sets = solver.get_info_set_count();
    const int64_t updates = solver.get_cfr_update_count();
    CHECK(std::isfinite(solver.evaluate_strategy(kA, kB)));
    CHECK(solver.get_iterations_run() == iterations);
    CHECK(solver.get_info_set_count() == info_sets);
    CHECK(solver.get_cfr_update_count() == updates);
  }

  SUBCASE("public-state cap forces mutable warmup") {
    SolverConfig config = Config();
    config.max_public_states = 1;
    config.regret_only_training = true;
    CFRSolver solver(config);
    solver.run(3, R(kA), R(kB));
    const auto run = solver.get_last_training_run_stats();
    CHECK_FALSE(run.public_state_prebuild_complete);
    CHECK(run.warmup_iterations == 3);
    CHECK(run.frozen_iterations == 0);
    CHECK(solver.get_public_state_count() <= 1);
  }

  SUBCASE("terminal and invalid roots") {
    SolverConfig config = Config();
    ExactPublicState terminal;
    terminal.betting.stack = {0, 0};
    terminal.betting.committed = {5, 5};
    terminal.betting.street = StreetKind::kRiver;
    terminal.betting.player_to_act = -1;
    terminal.betting.folded_player = 1;
    CFRSolver solver(config, terminal);
    solver.run(1, R(kA), R(kB));
    CHECK(solver.get_iterations_run() == 1);
    CHECK(solver.get_expected_value(0) == doctest::Approx(5.0));

    ExactPublicState invalid;
    invalid.betting.stack[0] = -1;
    CHECK_THROWS_AS(CFRSolver(config, invalid), std::invalid_argument);
  }
}

TEST_CASE("strategy storage performs regret matching, averaging, and freezing") {
  SolverConfig config;
  SolverStorage storage;
  TraversalStats stats;
  storage.mutable_tables->nodes.resize(1);
  storage.mutable_tables->betting_nodes.resize(1);
  storage.mutable_tables->nodes[0].betting_node_id = 0;
  auto& node = storage.mutable_tables->betting_nodes[0];
  node.kind = StrategyTables::NodeKind::kDecision;
  node.state.player_to_act = 0;
  node.action_count = 3;
  StrategyStore store(config, storage, &stats);

  double probabilities[3] = {};
  store.regret_matching_or_uniform(std::nullopt, 3, RegretLoadMode::kPlain,
                                   absl::Span<double>(probabilities));
  for (double probability : probabilities)
    CHECK(probability == doctest::Approx(1.0 / 3.0));

  const auto created = store.get_or_create({0, 0}, 3);
  REQUIRE(created.has_value());
  ActionBlock block = *created;
  const RegretUpdateOptions plain{RegretUpdateMode::kPlain, false};
  block.add_cfr_plus_regret(0, 1.0f, plain);
  block.add_cfr_plus_regret(0, -2.0f, plain);
  block.add_cfr_plus_regret(1, 2.0f, plain);
  block.add_cfr_plus_regret(2, 6.0f, plain);
  block.regret_matching(RegretLoadMode::kPlain,
                        absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.0));
  CHECK(probabilities[1] == doctest::Approx(0.25));
  CHECK(probabilities[2] == doctest::Approx(0.75));

  const double strategy[3] = {0.2, 0.3, 0.5};
  block.add_average_strategy(absl::Span<const double>(strategy), 2.0,
                             RegretUpdateMode::kPlain);
  block.average_strategy(false, 0.5, absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.2));
  CHECK(probabilities[1] == doctest::Approx(0.3));
  CHECK(probabilities[2] == doctest::Approx(0.5));

  REQUIRE(store.prebuild_frozen_info_set_action_offsets());
  const auto frozen = store.find_frozen(0, 0, 3);
  REQUIRE(frozen.has_value());
  double frozen_probabilities[3] = {};
  frozen->regret_matching(RegretLoadMode::kPlain,
                          absl::Span<double>(frozen_probabilities));
  for (int i = 0; i < 3; ++i)
    CHECK(frozen_probabilities[i] == doctest::Approx(
          i == 0 ? 0.0 : i == 1 ? 0.25 : 0.75));
}

}  // namespace
}  // namespace poker
