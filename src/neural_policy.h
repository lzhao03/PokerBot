#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "src/solver.h"

namespace poker {

inline constexpr uint32_t kNeuralFeatureSchemaVersion = 2;
inline constexpr size_t kNeuralFeatureCount =
    32 + 16 + 16 + 64 + 4 * 36 + 15;

using NeuralFeatureVector = std::array<float, kNeuralFeatureCount>;
using NeuralActionVector = std::array<float, kMaxActionsPerNode>;

struct NeuralSample {
  InfoSetKey key;
  NeuralActionVector target = {};
  float weight = 1.0f;
};

enum class NeuralTarget : uint8_t {
  Advantage,
  AveragePolicy,
  CurrentPolicy,
};

struct NeuralTrainingConfig {
  uint64_t seed = 1;
  int steps = 100;
  int batch_size = 256;
  double learning_rate = 1e-3;
};

class NeuralNetwork {
 public:
  explicit NeuralNetwork(int hidden_size);
  ~NeuralNetwork();
  NeuralNetwork(NeuralNetwork&&) noexcept;
  NeuralNetwork& operator=(NeuralNetwork&&) noexcept;

  NeuralNetwork(const NeuralNetwork&) = delete;
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  int hidden_size() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend size_t NeuralParameterBytes(const NeuralNetwork& network);
  friend NeuralActionVector PredictNeuralNetwork(
      NeuralNetwork& network,
      const CompiledGame& game,
      InfoSetKey key,
      std::array<std::vector<float>, 2>& hidden);
  friend float FitNeuralNetwork(
      NeuralNetwork& network,
      const CompiledGame& game,
      std::span<const NeuralSample> samples,
      const NeuralTrainingConfig& config,
      NeuralTarget target,
      const std::array<NeuralNetwork*, kPlayerCount>& current_policy_sources);
  friend void SaveNeuralNetwork(const NeuralNetwork& network,
                                const std::filesystem::path& path,
                                ModelFingerprint model);
  friend void LoadNeuralNetwork(NeuralNetwork& network,
                                const std::filesystem::path& path,
                                ModelFingerprint expected_model);
};

void SetNeuralSeed(uint64_t seed);

NeuralFeatureVector EncodeNeuralFeatures(
    InfoSetKey key,
    const HistoryNode& node,
    const SolverConfig& config);

size_t NeuralParameterBytes(const NeuralNetwork& network);

void FillUniform(std::span<float> probabilities);
void RegretMatch(std::span<const float> advantages,
                 std::span<float> probabilities);
void Softmax(std::span<const float> logits,
             std::span<float> probabilities);

NeuralActionVector PredictNeuralNetwork(
    NeuralNetwork& network,
    const CompiledGame& game,
    InfoSetKey key,
    std::array<std::vector<float>, 2>& hidden);

float FitNeuralNetwork(
    NeuralNetwork& network,
    const CompiledGame& game,
    std::span<const NeuralSample> samples,
    const NeuralTrainingConfig& config,
    NeuralTarget target,
    const std::array<NeuralNetwork*, kPlayerCount>& current_policy_sources =
        {});

void SaveNeuralNetwork(const NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint model);
void LoadNeuralNetwork(NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint expected_model);

}  // namespace poker
