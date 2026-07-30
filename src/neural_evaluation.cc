#include "src/neural_evaluation.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"

namespace poker {

StrategyLookup MakeStrategyLookup(const CompiledGame& game,
                                  const NeuralPolicy& policy) {
  constexpr size_t kCacheCapacity = 1'000'000;
  auto cache = std::make_shared<
      absl::flat_hash_map<InfoSetKey, NeuralActionVector>>();
  cache->reserve(100'000);
  return [&game, &policy, cache](InfoSetKey key, std::span<float> output) {
    const auto found = cache->find(key);
    if (found != cache->end()) {
      std::copy_n(found->second.begin(), output.size(), output.begin());
      return true;
    }
    NeuralActionVector probabilities = {};
    const bool available = policy.strategy(
        game, key,
        std::span<float>(probabilities.data(), output.size()));
    std::copy_n(probabilities.begin(), output.size(), output.begin());
    if (available && cache->size() < kCacheCapacity) {
      cache->emplace(key, probabilities);
    }
    return available;
  };
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const NeuralPolicy& player_a,
    const NeuralPolicy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (player_a.model() != game.model || player_b.model() != game.model) {
    return absl::FailedPreconditionError(
        "neural policy model does not match game");
  }
  return EstimateExpectedValue(
      game, MakeStrategyLookup(game, player_a),
      MakeStrategyLookup(game, player_b), samples, seed,
      measure_reach_coverage, sample_actions);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CompiledGame& game,
    const NeuralPolicy& policy,
    const BestResponseConfig& config) {
  if (policy.model() != game.model) {
    return absl::FailedPreconditionError(
        "neural policy model does not match game");
  }
  return EstimateExploitabilityParallel(
      game, [&] { return MakeStrategyLookup(game, policy); }, config);
}

}  // namespace poker
