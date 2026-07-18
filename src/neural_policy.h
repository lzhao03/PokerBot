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
};

struct NeuralTrainingConfig {
  uint64_t seed = 1;
  int steps = 100;
  int batch_size = 256;
  int hidden_size = 128;
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

void SaveNeuralNetwork(const NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint model);
void LoadNeuralNetwork(NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint expected_model);

struct NeuralPolicyFitResult;

class NeuralPolicy {
 public:
  NeuralPolicy(NeuralNetwork network, ModelFingerprint model);
  NeuralPolicy(NeuralPolicy&&) noexcept;
  NeuralPolicy& operator=(NeuralPolicy&&) noexcept;

  NeuralPolicy(const NeuralPolicy&) = delete;
  NeuralPolicy& operator=(const NeuralPolicy&) = delete;

  bool strategy(const CompiledGame& game,
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
      const Policy& teacher,
      const NeuralTrainingConfig& config);
  friend absl::Status SaveNeuralPolicy(
      const NeuralPolicy& policy,
      const std::filesystem::path& path);
  friend absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
      const std::filesystem::path& path,
      ModelFingerprint expected_model);
};

struct NeuralPolicyFitResult {
  NeuralPolicy policy;
  float loss = 0.0f;
  size_t samples = 0;
};

absl::StatusOr<NeuralPolicyFitResult> FitNeuralPolicy(
    const CompiledGame& game,
    const Policy& teacher,
    const NeuralTrainingConfig& config);

absl::Status SaveNeuralPolicy(const NeuralPolicy& policy,
                              const std::filesystem::path& path);
absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
    const std::filesystem::path& path,
    ModelFingerprint expected_model);

}  // namespace poker
