#include "src/neural_policy.h"

#include <cmath>
#include <filesystem>
#include <numeric>
#include <vector>

#include "doctest/doctest.h"
#include "src/bet_abstraction.h"
#include "src/evaluation.h"

namespace poker {
namespace {

absl::StatusOr<CompiledGame> TinyGame() {
  SolverConfig config;
  config.bet_abstraction = SmallBettingConfig();
  config.card_abstraction.public_mode = PublicCardMode::CompactTexture;
  config.card_abstraction.recall_mode = RecallMode::CurrentBucketOnly;
  const ComboRange range = UniformComboRange();
  return CompileGame({config,
                      MakeInitialState(config.betting_rules, {8, 8}, {1, 2}),
                      {range, range}});
}

TEST_CASE("tabular policies fit the shared neural policy format") {
  auto game = TinyGame();
  REQUIRE(game.ok());
  const HistoryNode& root = game->history.nodes.front();
  const InfoSetKey key{
      game->root.public_state.observation(), game->root.history,
      ObservePrivate(ComboId{}, game->root.public_state)};

  Policy teacher;
  teacher.model = game->model;
  teacher.rows.emplace(key, 0);
  teacher.probabilities.assign(root.child_count, 0.0f);
  teacher.probabilities.front() = 1.0f;

  const auto fitted = FitNeuralPolicy(
      *game, teacher,
      {.seed = 7,
       .steps = 100,
       .batch_size = 16,
       .hidden_size = 16,
       .learning_rate = 1e-2});
  REQUIRE(fitted.ok());
  CHECK(fitted->samples == 1);
  CHECK(std::isfinite(fitted->loss));
  CHECK(fitted->policy.model() == game->model);
  CHECK(fitted->policy.parameter_bytes() > 0);

  std::vector<float> probabilities(root.child_count);
  REQUIRE(fitted->policy.strategy(*game, key, probabilities));
  CHECK(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f) ==
        doctest::Approx(1.0f));
  CHECK(probabilities.front() > 0.9f);
  const auto value = EstimateExpectedValue(
      *game, fitted->policy, fitted->policy, 2, 11);
  REQUIRE(value.ok());
  CHECK(std::isfinite(value->mean));
  const auto sampled_value = EstimateExpectedValue(
      *game, fitted->policy, fitted->policy, 2, 11, false, true);
  REQUIRE(sampled_value.ok());
  CHECK(std::isfinite(sampled_value->mean));
  const auto exploitability = EstimateExploitability(
      *game, fitted->policy, {2, 2, 11});
  REQUIRE(exploitability.ok());
  CHECK(std::isfinite(exploitability->exploitability));

  const auto path =
      std::filesystem::temp_directory_path() / "poker_neural_policy_test.pt";
  REQUIRE(SaveNeuralPolicy(fitted->policy, path).ok());
  const auto loaded = LoadNeuralPolicy(path, game->model);
  REQUIRE(loaded.ok());
  std::vector<float> loaded_probabilities(root.child_count);
  REQUIRE(loaded->strategy(*game, key, loaded_probabilities));
  CHECK(loaded_probabilities == probabilities);
  CHECK_FALSE(LoadNeuralPolicy(path, ModelFingerprint{1}).ok());
  std::filesystem::remove(path);
}

TEST_CASE("neural fitting rejects a policy for another game") {
  auto game = TinyGame();
  REQUIRE(game.ok());
  Policy policy;
  policy.model = ModelFingerprint{1};
  CHECK_FALSE(FitNeuralPolicy(*game, policy, {}).ok());
}

}  // namespace
}  // namespace poker
