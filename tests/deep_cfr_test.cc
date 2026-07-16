#include "src/deep_cfr.h"

#include <cmath>

#include "doctest/doctest.h"

namespace poker {
namespace {

DeepCfrConfig TinyDeepConfig() {
  DeepCfrConfig config;
  config.advantage_memory_capacity = 128;
  config.strategy_memory_capacity = 128;
  config.inference_cache_capacity = 128;
  config.traversals_per_player = 8;
  config.training_steps = 4;
  config.batch_size = 16;
  config.hidden_size = 16;
  return config;
}

SolveSpec TinySolveSpec() {
  SolverConfig config;
  config.bet_abstraction = SmallBettingConfig();
  config.card_abstraction.public_mode = PublicCardMode::CompactTexture;
  config.card_abstraction.recall_mode = RecallMode::CurrentBucketOnly;
  const ComboRange range = UniformComboRange();
  const ExactPublicState root = MakeInitialState(
      config.betting_rules, {8, 8}, {1, 2});
  return {config, root, {range, range}};
}

TEST_CASE("Deep CFR trains bounded neural memories") {
  auto solver = DeepCfrSolver::Create(TinySolveSpec(), TinyDeepConfig());
  REQUIRE(solver.ok());
  REQUIRE(solver->run(2).ok());

  const DeepCfrStats& stats = solver->stats();
  CHECK(stats.iterations == 2);
  CHECK(stats.traversals == 32);
  CHECK(stats.advantage_samples[0] > 0);
  CHECK(stats.advantage_samples[1] > 0);
  CHECK(stats.advantage_samples[0] <= 128);
  CHECK(stats.advantage_samples[1] <= 128);
  CHECK(stats.strategy_samples > 0);
  CHECK(stats.strategy_samples <= 128);
  CHECK(std::isfinite(stats.advantage_loss[0]));
  CHECK(std::isfinite(stats.advantage_loss[1]));
  CHECK(std::isfinite(stats.strategy_loss));
  CHECK(stats.network_evaluations > 0);

  const auto value = solver->evaluate_average(4);
  REQUIRE(value.ok());
  CHECK(std::isfinite(*value));
}

TEST_CASE("Deep CFR rejects an empty reservoir") {
  DeepCfrConfig config = TinyDeepConfig();
  config.advantage_memory_capacity = 0;
  CHECK_FALSE(DeepCfrSolver::Create(TinySolveSpec(), config).ok());
}

TEST_CASE("Deep CFR run boundaries do not change training") {
  auto whole = DeepCfrSolver::Create(TinySolveSpec(), TinyDeepConfig());
  auto split = DeepCfrSolver::Create(TinySolveSpec(), TinyDeepConfig());
  REQUIRE(whole.ok());
  REQUIRE(split.ok());
  REQUIRE(whole->run(2).ok());
  REQUIRE(split->run(1).ok());
  REQUIRE(split->run(1).ok());

  CHECK(whole->stats().advantage_samples ==
        split->stats().advantage_samples);
  CHECK(whole->stats().strategy_samples == split->stats().strategy_samples);
  CHECK(whole->stats().advantage_loss == split->stats().advantage_loss);
  CHECK(whole->stats().strategy_loss == split->stats().strategy_loss);

  const auto whole_value = whole->evaluate_average(4);
  const auto split_value = split->evaluate_average(4);
  REQUIRE(whole_value.ok());
  REQUIRE(split_value.ok());
  CHECK(*whole_value == *split_value);
}

}  // namespace
}  // namespace poker
