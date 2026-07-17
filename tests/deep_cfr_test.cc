#include "src/deep_cfr.h"

#include <cmath>
#include <filesystem>

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
  config.policy_training_steps = 4;
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
  CHECK_FALSE(solver->game().config.external_sampling);
  CHECK(solver->game().config.accumulate_average_strategy);
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
  CHECK(stats.policy_parameter_bytes > 0);

  const auto value = solver->evaluate_average(4);
  REQUIRE(value.ok());
  CHECK(std::isfinite(*value));
  const auto value_against_uniform =
      solver->evaluate_average_against_uniform(Player::A, 4);
  REQUIRE(value_against_uniform.ok());
  CHECK(std::isfinite(*value_against_uniform));

  const auto path =
      std::filesystem::temp_directory_path() / "poker_deep_cfr_test.pt";
  REQUIRE(solver->save_average_model(path).ok());
  CHECK(std::filesystem::file_size(path) > 0);

  auto loaded = DeepCfrSolver::Create(TinySolveSpec(), TinyDeepConfig());
  REQUIRE(loaded.ok());
  REQUIRE(loaded->load_average_model(path).ok());
  const auto loaded_value = loaded->evaluate_average(4);
  REQUIRE(loaded_value.ok());
  CHECK(std::isfinite(*loaded_value));

  Policy uniform;
  uniform.model = loaded->game().model;
  const auto match = loaded->evaluate_against_policy(
      Player::A, uniform, DeepCfrStrategy::Average, 4);
  REQUIRE(match.ok());
  CHECK(std::isfinite(match->policy_player_value));
  CHECK(match->opponent_policy_lookups > 0);
  CHECK(match->missing_opponent_lookups == match->opponent_policy_lookups);
  std::filesystem::remove(path);
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
