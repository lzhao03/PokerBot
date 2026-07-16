#include <cmath>
#include <cstdint>
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "src/deep_cfr.h"

ABSL_FLAG(uint64_t, iterations, 2, "Deep CFR outer iterations");
ABSL_FLAG(int, traversals_per_player, 64,
          "External-sampling traversals per player and iteration");
ABSL_FLAG(int, training_steps, 50, "Optimizer steps per network fit");
ABSL_FLAG(int, batch_size, 128, "Neural training batch size");
ABSL_FLAG(int, hidden_size, 32, "Width of both hidden layers");
ABSL_FLAG(double, learning_rate, 1e-3, "Adam learning rate");
ABSL_FLAG(uint64_t, memory_capacity, 4096,
          "Capacity of each reservoir");
ABSL_FLAG(uint64_t, cache_capacity, 4096,
          "Maximum cached neural strategies");
ABSL_FLAG(int, starting_stack, 8, "Starting stack in chips");
ABSL_FLAG(int, evaluation_samples, 64,
          "Deals sampled for average-policy evaluation");
ABSL_FLAG(uint64_t, seed, 1, "Training seed");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the optional LibTorch Deep CFR solver.");
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_iterations) == 0 ||
      absl::GetFlag(FLAGS_starting_stack) < 2 ||
      absl::GetFlag(FLAGS_evaluation_samples) <= 0) {
    std::cerr << "iterations and evaluation samples must be positive; "
                 "stack must be at least two\n";
    return 1;
  }

  poker::SolverConfig solver_config;
  solver_config.bet_abstraction = poker::SmallBettingConfig();
  solver_config.card_abstraction.public_mode =
      poker::PublicCardMode::CompactTexture;
  solver_config.card_abstraction.private_kind =
      poker::PrivateAbstractionKind::Handcrafted36;
  solver_config.card_abstraction.recall_mode =
      poker::RecallMode::CurrentBucketOnly;

  poker::DeepCfrConfig deep_config;
  deep_config.seed = absl::GetFlag(FLAGS_seed);
  deep_config.advantage_memory_capacity =
      absl::GetFlag(FLAGS_memory_capacity);
  deep_config.strategy_memory_capacity =
      absl::GetFlag(FLAGS_memory_capacity);
  deep_config.inference_cache_capacity =
      absl::GetFlag(FLAGS_cache_capacity);
  deep_config.traversals_per_player =
      absl::GetFlag(FLAGS_traversals_per_player);
  deep_config.training_steps = absl::GetFlag(FLAGS_training_steps);
  deep_config.batch_size = absl::GetFlag(FLAGS_batch_size);
  deep_config.hidden_size = absl::GetFlag(FLAGS_hidden_size);
  deep_config.learning_rate = absl::GetFlag(FLAGS_learning_rate);

  const poker::ComboRange range = poker::UniformComboRange();
  const poker::ExactPublicState root = poker::MakeInitialState(
      solver_config.betting_rules,
      {absl::GetFlag(FLAGS_starting_stack),
       absl::GetFlag(FLAGS_starting_stack)},
      {1, 2});
  auto solver = poker::DeepCfrSolver::Create(
      {solver_config, root, {range, range}}, deep_config);
  if (!solver.ok()) {
    std::cerr << solver.status() << '\n';
    return 1;
  }
  const absl::Status trained = solver->run(absl::GetFlag(FLAGS_iterations));
  if (!trained.ok()) {
    std::cerr << trained << '\n';
    return 1;
  }
  const auto value =
      solver->evaluate_average(absl::GetFlag(FLAGS_evaluation_samples));
  if (!value.ok()) {
    std::cerr << value.status() << '\n';
    return 1;
  }

  const poker::DeepCfrStats& stats = solver->stats();
  std::cout << "iterations=" << stats.iterations << '\n'
            << "traversals=" << stats.traversals << '\n'
            << "advantage_samples_a=" << stats.advantage_samples[0] << '\n'
            << "advantage_samples_b=" << stats.advantage_samples[1] << '\n'
            << "strategy_samples=" << stats.strategy_samples << '\n'
            << "advantage_loss_a=" << stats.advantage_loss[0] << '\n'
            << "advantage_loss_b=" << stats.advantage_loss[1] << '\n'
            << "strategy_loss=" << stats.strategy_loss << '\n'
            << "network_evaluations=" << stats.network_evaluations << '\n'
            << "cache_hits=" << stats.cache_hits << '\n'
            << "average_value=" << *value << '\n';
  return std::isfinite(*value) ? 0 : 1;
}
