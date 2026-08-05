#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/neural_features.h"
#include "src/solver.h"

namespace poker {

struct NeuralSample {
  InfoSetKey key;
  NeuralActionVector target = {};
  float weight = 1.0f;
};

enum class NeuralTarget : uint8_t {
  Advantage,
  AveragePolicy,
};

struct NeuralTrainingConfig {
  uint64_t seed = 1;
  int steps = 100;
  int batch_size = 256;
  int hidden_size = 128;
  double learning_rate = 1e-3;
};

class NeuralPolicy;

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

  friend class NeuralPolicy;
  friend NeuralActionVector PredictNeuralNetwork(
      const NeuralNetwork& network,
      const CompiledGame& game,
      InfoSetKey key,
      std::array<std::vector<float>, 2>& hidden);
  friend float FitNeuralNetwork(
      NeuralNetwork& network,
      const CompiledGame& game,
      std::span<const NeuralSample> samples,
      const NeuralTrainingConfig& config,
      NeuralTarget target);
  friend absl::Status SaveNeuralPolicy(
      const NeuralPolicy& policy,
      const std::filesystem::path& path);
  friend absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
      const std::filesystem::path& path,
      ModelFingerprint expected_model);
  friend absl::Status SavePortableNeuralPolicy(
      const NeuralPolicy& policy,
      const std::filesystem::path& path);
};

void SetNeuralSeed(uint64_t seed);
void UseSingleThreadedNeuralRuntime();

NeuralFeatureVector EncodeNeuralFeatures(
    InfoSetKey key,
    const HistoryNode& node,
    const SolverConfig& config);

void FillUniform(std::span<float> probabilities);
void RegretMatch(std::span<const float> advantages,
                 std::span<float> probabilities);
void Softmax(std::span<const float> logits,
             std::span<float> probabilities);

NeuralActionVector PredictNeuralNetwork(
    const NeuralNetwork& network,
    const CompiledGame& game,
    InfoSetKey key,
    std::array<std::vector<float>, 2>& hidden);

float FitNeuralNetwork(
    NeuralNetwork& network,
    const CompiledGame& game,
    std::span<const NeuralSample> samples,
    const NeuralTrainingConfig& config,
    NeuralTarget target);

struct NeuralPolicyFitResult;

class NeuralPolicy {
 public:
  NeuralPolicy(NeuralNetwork network, ModelFingerprint model);
  NeuralPolicy(NeuralPolicy&&) noexcept;
  NeuralPolicy& operator=(NeuralPolicy&&) noexcept;

  NeuralPolicy(const NeuralPolicy&) = delete;
  NeuralPolicy& operator=(const NeuralPolicy&) = delete;

  bool strategy(const CompiledGame& game,
                ModelFingerprint model,
                InfoSetKey key,
                std::span<float> probabilities) const;
  size_t parameter_bytes() const;
  ModelFingerprint model() const noexcept { return model_; }

 private:
  NeuralNetwork network_;
  ModelFingerprint model_;

  friend struct NeuralPolicyFitResult;
  friend absl::StatusOr<NeuralPolicyFitResult> FitNeuralPolicy(
      const CompiledGame& game,
      ModelFingerprint model,
      const Policy& teacher,
      const NeuralTrainingConfig& config);
  friend absl::Status SaveNeuralPolicy(
      const NeuralPolicy& policy,
      const std::filesystem::path& path);
  friend absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
      const std::filesystem::path& path,
      ModelFingerprint expected_model);
  friend absl::Status SavePortableNeuralPolicy(
      const NeuralPolicy& policy,
      const std::filesystem::path& path);
};

struct NeuralPolicyFitResult {
  NeuralPolicy policy;
  float loss = 0.0f;
  size_t samples = 0;
};

absl::StatusOr<NeuralPolicyFitResult> FitNeuralPolicy(
    const CompiledGame& game,
    ModelFingerprint model,
    const Policy& teacher,
    const NeuralTrainingConfig& config);

absl::Status SaveNeuralPolicy(const NeuralPolicy& policy,
                              const std::filesystem::path& path);
absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
    const std::filesystem::path& path,
    ModelFingerprint expected_model);
absl::Status SavePortableNeuralPolicy(
    const NeuralPolicy& policy,
    const std::filesystem::path& path);

}  // namespace poker
