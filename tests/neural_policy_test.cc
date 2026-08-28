#include "src/neural_policy.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "src/bet_abstraction.h"
#include "src/neural_evaluation.h"

namespace poker {
namespace {

SolveSpec TinySpec() {
  SolverConfig config;
  config.bet_abstraction = SmallBettingConfig();
  config.card_abstraction.public_mode = PublicCardMode::CompactTexture;
  config.card_abstraction.recall_mode = RecallMode::CurrentBucketOnly;
  const ComboRange range = UniformComboRange();
  return {config,
          MakeInitialState(config.betting_rules, {8, 8}, {1, 2}),
          {range, range}};
}

TEST_CASE("tabular policies fit the shared neural policy format") {
  const SolveSpec spec = TinySpec();
  const SolverConfig& config = spec.config;
  auto deals = DealDistribution::Create(spec.ranges[0], spec.ranges[1]);
  REQUIRE(deals.ok());
  const HistoryTree history = BuildHistoryTree(
      spec.root.betting, config.betting_rules, config.bet_abstraction);
  const PublicPosition initial_public(
      config.card_abstraction, spec.root.board);
  const ModelFingerprint model =
      ModelFingerprintFor(config, spec.root, spec.ranges);
  const HistoryNode& root = history.nodes.front();
  const InfoSetKey key{
      initial_public.observation(), HistoryId{},
      ObservePrivate(ComboId{}, initial_public)};

  Policy teacher;
  teacher.model = model;
  teacher.rows.emplace(key, 0);
  teacher.probabilities.assign(root.child_count, 0.0f);
  teacher.probabilities.front() = 1.0f;

  const auto fitted = FitNeuralPolicy(
      history, config.card_abstraction, model, teacher,
      {.seed = 7,
       .steps = 100,
       .batch_size = 16,
       .hidden_size = 16,
       .learning_rate = 1e-2});
  REQUIRE(fitted.ok());
  CHECK(fitted->samples == 1);
  CHECK(std::isfinite(fitted->loss));
  CHECK(fitted->policy.model() == model);
  CHECK(fitted->policy.parameter_bytes() > 0);

  std::vector<float> probabilities(root.child_count);
  REQUIRE(fitted->policy.strategy(
      history, config.card_abstraction, model, key, probabilities));
  CHECK(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f) ==
        doctest::Approx(1.0f));
  CHECK(probabilities.front() > 0.9f);

  const StrategyLookup unavailable = MakeStrategyLookup(
      history, config.card_abstraction, ModelFingerprint{1}, fitted->policy);
  std::vector<float> unavailable_probabilities(
      root.child_count, std::numeric_limits<float>::quiet_NaN());
  CHECK_FALSE(unavailable(key, unavailable_probabilities));
  CHECK_FALSE(unavailable(key, unavailable_probabilities));
  for (float probability : unavailable_probabilities) {
    CHECK(probability == doctest::Approx(1.0f / root.child_count));
  }
  const auto value = EstimateExpectedValue(
      config, *deals, history, initial_public, model, fitted->policy,
      fitted->policy, 2, 11);
  REQUIRE(value.ok());
  CHECK(std::isfinite(value->mean));
  const auto sampled_value = EstimateExpectedValue(
      config, *deals, history, initial_public, model, fitted->policy,
      fitted->policy, 2, 11, false, true);
  REQUIRE(sampled_value.ok());
  CHECK(std::isfinite(sampled_value->mean));
  const auto exploitability = EstimateExploitability(
      config, *deals, history, initial_public, model, fitted->policy,
      {2, 2, 11});
  REQUIRE(exploitability.ok());
  CHECK(std::isfinite(exploitability->exploitability));

  const auto path =
      std::filesystem::temp_directory_path() / "poker_neural_policy_test.pt";
  REQUIRE(SaveNeuralPolicy(fitted->policy, path).ok());
  const auto loaded = LoadNeuralPolicy(path, model);
  REQUIRE(loaded.ok());
  std::vector<float> loaded_probabilities(root.child_count);
  REQUIRE(loaded->strategy(
      history, config.card_abstraction, model, key, loaded_probabilities));
  CHECK(loaded_probabilities == probabilities);
  CHECK_FALSE(LoadNeuralPolicy(path, ModelFingerprint{1}).ok());

  const auto portable_path =
      std::filesystem::temp_directory_path() / "poker_neural_policy_test.pnn";
  REQUIRE(SavePortableNeuralPolicy(fitted->policy, portable_path).ok());
  std::ifstream portable_input(portable_path, std::ios::binary);
  const std::vector<char> bytes(
      (std::istreambuf_iterator<char>(portable_input)),
      std::istreambuf_iterator<char>());
  REQUIRE(bytes.size() == 32 + fitted->policy.parameter_bytes());
  CHECK(std::string(bytes.begin(), bytes.begin() + 4) == "PNN1");
  CHECK(bytes[4] == 1);
  CHECK(bytes[8] == kNeuralFeatureSchemaVersion);
  std::filesystem::remove(path);
  std::filesystem::remove(portable_path);
}

TEST_CASE("neural fitting rejects a policy for another game") {
  const SolveSpec spec = TinySpec();
  const SolverConfig& config = spec.config;
  const HistoryTree history = BuildHistoryTree(
      spec.root.betting, config.betting_rules, config.bet_abstraction);
  const ModelFingerprint model =
      ModelFingerprintFor(config, spec.root, spec.ranges);
  Policy policy;
  policy.model = ModelFingerprint{1};
  CHECK_FALSE(FitNeuralPolicy(
      history, config.card_abstraction, model, policy, {}).ok());
}

}  // namespace
}  // namespace poker
