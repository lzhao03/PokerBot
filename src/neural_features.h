#pragma once

#include <array>
#include <cstddef>

#include "src/card_abstraction.h"
#include "src/history.h"

namespace poker {

inline constexpr uint32_t kNeuralFeatureSchemaVersion = 2;
inline constexpr size_t kNeuralFeatureCount =
    32 + 16 + 16 + 64 + 4 * kPrivateBucketCount + 15;

using NeuralFeatureVector = std::array<float, kNeuralFeatureCount>;
using NeuralActionVector = std::array<float, kMaxActionsPerNode>;

NeuralFeatureVector EncodeNeuralFeatures(
    HistoryId history,
    PublicObservationId public_observation,
    PrivateObservationId private_observation,
    const HistoryNode& node,
    const CardAbstractionConfig& cards);

}  // namespace poker
