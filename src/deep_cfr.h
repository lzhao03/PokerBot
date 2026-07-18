#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/evaluation.h"

namespace poker {

struct DeepCfrConfig {
  uint64_t seed = 1;
  size_t advantage_memory_capacity = 100'000;
  size_t strategy_memory_capacity = 100'000;
  size_t inference_cache_capacity = 100'000;
  size_t policy_cache_capacity = 1'000'000;
  int traversals_per_player = 1024;
  int training_steps = 750;
  int policy_training_steps = 2500;
  int batch_size = 256;
  int hidden_size = 256;
  double learning_rate = 1e-3;
};

struct DeepCfrStats {
  uint64_t iterations = 0;
  uint64_t traversals = 0;
  std::array<size_t, kPlayerCount> advantage_samples = {};
  size_t strategy_samples = 0;
  std::array<float, kPlayerCount> advantage_loss = {};
  float strategy_loss = 0.0f;
  uint64_t network_evaluations = 0;
  uint64_t cache_hits = 0;
  size_t policy_parameter_bytes = 0;
  SolverStats traversal;
};

struct DeepCfrMatchResult {
  double policy_player_value = 0.0;
  double standard_error = 0.0;
  uint64_t opponent_policy_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
};

enum class DeepCfrStrategy : uint8_t {
  Current,
  Average,
};

class DeepCfrSolver {
 public:
  static absl::StatusOr<DeepCfrSolver> Create(
      SolveSpec spec,
      DeepCfrConfig config = {});

  ~DeepCfrSolver();
  DeepCfrSolver(DeepCfrSolver&&) noexcept;
  DeepCfrSolver& operator=(DeepCfrSolver&&) noexcept;

  DeepCfrSolver(const DeepCfrSolver&) = delete;
  DeepCfrSolver& operator=(const DeepCfrSolver&) = delete;

  absl::Status run(uint64_t iterations);
  absl::StatusOr<double> evaluate_current(int samples);
  absl::StatusOr<double> evaluate_average(int samples);
  absl::StatusOr<double> evaluate_average_against_uniform(
      Player policy_player,
      int samples);
  absl::StatusOr<DeepCfrMatchResult> evaluate_against_policy(
      Player policy_player,
      const Policy& opponent,
      DeepCfrStrategy strategy,
      int samples);
  absl::StatusOr<ExploitabilityEstimate> estimate_exploitability(
      const BestResponseConfig& config);
  absl::Status load_average_model(const std::filesystem::path& path);
  absl::Status save_average_model(const std::filesystem::path& path) const;

  const DeepCfrStats& stats() const noexcept;
  const CompiledGame& game() const noexcept;

 private:
  struct Impl;

  explicit DeepCfrSolver(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace poker
