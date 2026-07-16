#include <cmath>
#include <cstdint>
#include <iostream>

#include "src/deep_cfr.h"

int main() {
  poker::SolverConfig solver_config;
  solver_config.bet_abstraction = poker::SmallBettingConfig();
  solver_config.card_abstraction.public_mode =
      poker::PublicCardMode::CompactTexture;
  solver_config.card_abstraction.private_kind =
      poker::PrivateAbstractionKind::Handcrafted36;
  solver_config.card_abstraction.recall_mode =
      poker::RecallMode::CurrentBucketOnly;

  poker::DeepCfrConfig deep_config;
  deep_config.advantage_memory_capacity = 4096;
  deep_config.strategy_memory_capacity = 4096;
  deep_config.inference_cache_capacity = 4096;
  deep_config.traversals_per_player = 64;
  deep_config.training_steps = 50;
  deep_config.batch_size = 128;
  deep_config.hidden_size = 32;

  const poker::ComboRange range = poker::UniformComboRange();
  const poker::ExactPublicState root = poker::MakeInitialState(
      solver_config.betting_rules, {8, 8}, {1, 2});
  auto solver = poker::DeepCfrSolver::Create(
      {solver_config, root, {range, range}}, deep_config);
  if (!solver.ok()) {
    std::cerr << solver.status() << '\n';
    return 1;
  }
  const absl::Status trained = solver->run(2);
  if (!trained.ok()) {
    std::cerr << trained << '\n';
    return 1;
  }
  const auto value = solver->evaluate_average(64);
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
