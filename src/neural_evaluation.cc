#include "src/neural_evaluation.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"

namespace poker {

StrategyLookup MakeStrategyLookup(
    const HistoryTree& history, const CardAbstractionConfig& card_abstraction,
    ModelFingerprint model, const NeuralPolicy& policy) {
  constexpr size_t kCacheCapacity = 1'000'000;
  auto cache = std::make_shared<absl::flat_hash_map<InfoSetKey, NeuralActionVector>>();
  cache->reserve(100'000);
  return [&history, card_abstraction, model, &policy, cache](InfoSetKey key,
                                                             std::span<float> output) {
    const auto found = cache->find(key);
    if (found != cache->end()) {
      std::copy_n(found->second.begin(), output.size(), output.begin());
      return true;
    }
    NeuralActionVector probabilities = {};
    const bool available = policy.strategy(
        history, card_abstraction, model, key,
        std::span<float>(probabilities.data(), output.size()));
    std::copy_n(probabilities.begin(), output.size(), output.begin());
    if (available && cache->size() < kCacheCapacity) cache->emplace(key, probabilities);
    return available;
  };
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    const NeuralPolicy& player_a, const NeuralPolicy& player_b, uint64_t samples, uint64_t seed,
    bool measure_reach_coverage, bool sample_actions) {
  if (player_a.model() != model || player_b.model() != model) {
    return absl::FailedPreconditionError("neural policy model does not match game");
  }
  return EstimateExpectedValue(
      solver_config, deals, history, initial_public,
      MakeStrategyLookup(history, solver_config.card_abstraction, model, player_a),
      MakeStrategyLookup(history, solver_config.card_abstraction, model, player_b),
      samples, seed, measure_reach_coverage, sample_actions);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config, const DealDistribution& deals,
    const HistoryTree& history, const PublicPosition& initial_public, ModelFingerprint model,
    const NeuralPolicy& policy, const BestResponseConfig& config) {
  if (policy.model() != model) {
    return absl::FailedPreconditionError("neural policy model does not match game");
  }
  return EstimateExploitabilityParallel(
      solver_config, deals, history, initial_public, model,
      [&] {
        return MakeStrategyLookup(history, solver_config.card_abstraction, model, policy);
      },
      config);
}

}  // namespace poker
