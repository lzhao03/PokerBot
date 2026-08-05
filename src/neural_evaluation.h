#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "src/evaluation.h"
#include "src/neural_policy.h"

namespace poker {

StrategyLookup MakeStrategyLookup(const HistoryTree& history,
                                  const CardAbstractionConfig& card_abstraction,
                                  ModelFingerprint model,
                                  const NeuralPolicy& policy);
StrategyLookup MakeStrategyLookup(HistoryTree&& history,
                                  const CardAbstractionConfig& card_abstraction,
                                  ModelFingerprint model,
                                  const NeuralPolicy& policy) = delete;
StrategyLookup MakeStrategyLookup(const HistoryTree& history,
                                  const CardAbstractionConfig& card_abstraction,
                                  ModelFingerprint model,
                                  NeuralPolicy&& policy) = delete;

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const NeuralPolicy& player_a,
    const NeuralPolicy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false,
    bool sample_actions = false);

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const NeuralPolicy& policy,
    const BestResponseConfig& config =
        BestResponseConfig{.external_sampling = true});

}  // namespace poker
