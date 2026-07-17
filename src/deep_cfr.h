#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/solver.h"

namespace poker {

struct DeepCfrConfig {
  uint64_t seed = 1;
  size_t advantage_memory_capacity = 100'000;
  size_t strategy_memory_capacity = 100'000;
  size_t inference_cache_capacity = 100'000;
  int traversals_per_player = 100;
  int training_steps = 100;
  int batch_size = 256;
  int hidden_size = 128;
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
  SolverStats traversal;
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

  const DeepCfrStats& stats() const noexcept;
  const CompiledGame& game() const noexcept;

 private:
  struct Impl;

  explicit DeepCfrSolver(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace poker
