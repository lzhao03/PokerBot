#include "src/deep_cfr.h"
#include "src/policy_codec.h"
#include "src/solver.h"

#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

ABSL_FLAG(int, iterations, 100, "CFR iterations");
ABSL_FLAG(std::string, algorithm, "tabular", "tabular or deep");
ABSL_FLAG(int, starting_stack, 100, "starting stack in chips");
ABSL_FLAG(int, small_blind, 1, "small blind in chips");
ABSL_FLAG(int, big_blind, 2, "big blind in chips");
ABSL_FLAG(int, chance_samples, 1, "chance samples per chance node");
ABSL_FLAG(int, max_info_sets, 500000, "maximum infosets");
ABSL_FLAG(int, threads, 1, "training worker threads after infoset prefill");
ABSL_FLAG(int64_t, max_memory_mb, 4096,
          "hard memory limit in MB; 0 is unlimited");
ABSL_FLAG(bool, accumulate_average_strategy, true,
          "store the average strategy");
ABSL_FLAG(bool, external_sampling, false,
          "sample opponent actions during training");
ABSL_FLAG(std::string, public_abstraction, "texture",
          "exact, texture, or compact_texture");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36",
          "exact or handcrafted36");
ABSL_FLAG(std::string, private_recall, "auto",
          "auto, current, or history");
ABSL_FLAG(std::string, betting_abstraction, "default",
          "default or small_betting");
ABSL_FLAG(std::vector<std::string>, pot_fractions, {},
          "override pot fractions after calling for every street");
ABSL_FLAG(std::vector<std::string>, preflop_pot_fractions, {},
          "preflop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, flop_pot_fractions, {},
          "flop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, turn_pot_fractions, {},
          "turn pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, river_pot_fractions, {},
          "river pot fractions after calling");
ABSL_FLAG(int, deep_traversals_per_player, 64,
          "Deep CFR traversals per player and iteration");
ABSL_FLAG(int, deep_training_steps, 50,
          "Deep CFR optimizer steps per network fit");
ABSL_FLAG(int, deep_policy_training_steps, 500,
          "Deep CFR optimizer steps for the final average policy");
ABSL_FLAG(int, deep_batch_size, 128, "Deep CFR neural training batch size");
ABSL_FLAG(int, deep_hidden_size, 32,
          "Deep CFR width of both hidden layers");
ABSL_FLAG(double, deep_learning_rate, 1e-3, "Deep CFR Adam learning rate");
ABSL_FLAG(uint64_t, deep_memory_capacity, 4096,
          "Deep CFR capacity of each reservoir");
ABSL_FLAG(uint64_t, deep_cache_capacity, 4096,
          "Deep CFR maximum cached neural strategies");
ABSL_FLAG(int, deep_evaluation_samples, 64,
          "Deals sampled for Deep CFR average-policy evaluation");
ABSL_FLAG(uint64_t, deep_seed, 1, "Deep CFR training seed");
ABSL_FLAG(std::string, deep_model_output, "",
          "output path for the trained Deep CFR average model");
ABSL_FLAG(std::string, deep_model_input, "",
          "trained Deep CFR average model to load");
ABSL_FLAG(std::string, deep_opponent_policy, "",
          "tabular policy to evaluate against the Deep CFR model");

namespace {

void SetMemoryLimit(int64_t megabytes) {
  if (megabytes <= 0) {
    return;
  }
  const rlim_t bytes = static_cast<rlim_t>(megabytes) * 1024ULL * 1024ULL;
  const rlimit limit{bytes, bytes};
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    std::cerr << "Warning: failed to set memory limit to " << megabytes
              << " MB: " << std::strerror(errno) << "\n";
  }
}

absl::StatusOr<std::vector<double>> ParsePotFractions(
    const std::vector<std::string>& values) {
  std::vector<double> sizes;
  sizes.reserve(values.size());
  for (const std::string& value : values) {
    double size = 0.0;
    if (!absl::SimpleAtod(value, &size)) {
      return absl::InvalidArgumentError("invalid pot fraction: " + value);
    }
    sizes.push_back(size);
  }
  return sizes;
}

absl::StatusOr<poker::SolverConfig> ConfigFromFlags() {
  poker::SolverConfig config;
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::Chips small_blind = absl::GetFlag(FLAGS_small_blind);
  const poker::Chips big_blind = absl::GetFlag(FLAGS_big_blind);
  if (stack <= 0 || small_blind <= 0 || big_blind < small_blind ||
      stack < big_blind) {
    return absl::InvalidArgumentError("invalid stack or blind configuration");
  }
  config.betting_rules.minimum_bet = big_blind;
  config.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  config.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  config.accumulate_average_strategy =
      absl::GetFlag(FLAGS_accumulate_average_strategy);
  config.external_sampling = absl::GetFlag(FLAGS_external_sampling);

  const std::string public_abstraction =
      absl::GetFlag(FLAGS_public_abstraction);
  if (public_abstraction == "exact") {
    config.card_abstraction.public_mode =
        poker::PublicCardMode::ExactCanonical;
  } else if (public_abstraction == "compact_texture") {
    config.card_abstraction.public_mode =
        poker::PublicCardMode::CompactTexture;
  } else if (public_abstraction != "texture") {
    return absl::InvalidArgumentError("invalid public abstraction");
  }

  const std::string private_abstraction =
      absl::GetFlag(FLAGS_private_abstraction);
  if (private_abstraction != "exact" &&
      private_abstraction != "handcrafted36") {
    return absl::InvalidArgumentError("invalid private abstraction");
  }
  config.card_abstraction.private_kind = private_abstraction == "exact"
      ? poker::PrivateAbstractionKind::ExactCanonical
      : poker::PrivateAbstractionKind::Handcrafted36;

  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall != "auto" && recall != "current" && recall != "history") {
    return absl::InvalidArgumentError("invalid private recall mode");
  }
  const bool retain_bucket_history =
      recall == "history" ||
      (recall == "auto" &&
       config.card_abstraction.private_kind ==
           poker::PrivateAbstractionKind::Handcrafted36);
  config.card_abstraction.recall_mode = retain_bucket_history
          ? poker::RecallMode::BucketHistory
          : poker::RecallMode::CurrentBucketOnly;

  const std::string betting_abstraction =
      absl::GetFlag(FLAGS_betting_abstraction);
  if (betting_abstraction == "small_betting") {
    config.bet_abstraction = poker::SmallBettingConfig();
  } else if (betting_abstraction != "default") {
    return absl::InvalidArgumentError("invalid betting abstraction");
  }
  const auto global_fractions = absl::GetFlag(FLAGS_pot_fractions);
  if (!global_fractions.empty()) {
    const auto fractions = ParsePotFractions(global_fractions);
    if (!fractions.ok()) return fractions.status();
    config.bet_abstraction.pot_fractions.fill(*fractions);
  }
  const std::array overrides = {
      absl::GetFlag(FLAGS_preflop_pot_fractions),
      absl::GetFlag(FLAGS_flop_pot_fractions),
      absl::GetFlag(FLAGS_turn_pot_fractions),
      absl::GetFlag(FLAGS_river_pot_fractions),
  };
  for (size_t street = 0; street < overrides.size(); ++street) {
    if (overrides[street].empty()) continue;
    const auto override = ParsePotFractions(overrides[street]);
    if (!override.ok()) return override.status();
    config.bet_abstraction.pot_fractions[street] = *override;
  }
  return poker::SolverConfig::Create(std::move(config));
}

void PrintRunSummary(const poker::TabularCfrSolver& solver,
                     const poker::SolverConfig& config,
                     double seconds) {
  const size_t info_sets = solver.info_set_count();
  const size_t history_nodes = solver.history_count();
  const uint64_t visits = solver.stats().decision_visits;

  std::cout << "iterations=" << solver.iterations() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets << "\n";
  std::cout << "info_set_cap_hit="
            << (info_sets >= static_cast<size_t>(config.max_info_sets))
            << "\n";
  std::cout << "player_a_ev=" << solver.expected_value(poker::Player::A)
            << "\n";
  std::cout << "seconds=" << seconds << "\n";
  std::cout << "history_nodes=" << history_nodes << "\n";
  std::cout << "decision_visits=" << visits << "\n";
  if (seconds > 0.0) {
    std::cout << "decision_visits_per_second="
              << visits / seconds << "\n";
  }
}

int RunTabular(poker::SolveSpec spec, uint64_t iterations, int threads) {
  auto solver = poker::TabularCfrSolver::Create(std::move(spec));
  if (!solver.ok()) {
    std::cerr << "Error: " << solver.status() << "\n";
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  solver->run(iterations, threads);
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - start;

  PrintRunSummary(*solver, solver->game().config, elapsed.count());
  std::cout << "threads=" << threads << "\n";
  return 0;
}

int RunDeep(poker::SolveSpec spec, uint64_t iterations) {
  poker::DeepCfrConfig config;
  config.seed = absl::GetFlag(FLAGS_deep_seed);
  config.advantage_memory_capacity =
      absl::GetFlag(FLAGS_deep_memory_capacity);
  config.strategy_memory_capacity =
      absl::GetFlag(FLAGS_deep_memory_capacity);
  config.inference_cache_capacity =
      absl::GetFlag(FLAGS_deep_cache_capacity);
  config.traversals_per_player =
      absl::GetFlag(FLAGS_deep_traversals_per_player);
  config.training_steps = absl::GetFlag(FLAGS_deep_training_steps);
  config.policy_training_steps =
      absl::GetFlag(FLAGS_deep_policy_training_steps);
  config.batch_size = absl::GetFlag(FLAGS_deep_batch_size);
  config.hidden_size = absl::GetFlag(FLAGS_deep_hidden_size);
  config.learning_rate = absl::GetFlag(FLAGS_deep_learning_rate);

  auto solver = poker::DeepCfrSolver::Create(std::move(spec), config);
  if (!solver.ok()) {
    std::cerr << "Error: " << solver.status() << '\n';
    return 1;
  }
  const std::string model_input = absl::GetFlag(FLAGS_deep_model_input);
  if (!model_input.empty()) {
    const absl::Status loaded = solver->load_average_model(model_input);
    if (!loaded.ok()) {
      std::cerr << "Error: " << loaded << '\n';
      return 1;
    }
  }
  const auto start = std::chrono::steady_clock::now();
  if (iterations > 0) {
    const absl::Status trained = solver->run(iterations);
    if (!trained.ok()) {
      std::cerr << "Error: " << trained << '\n';
      return 1;
    }
  }
  const auto value =
      solver->evaluate_average(absl::GetFlag(FLAGS_deep_evaluation_samples));
  if (!value.ok()) {
    std::cerr << "Error: " << value.status() << '\n';
    return 1;
  }
  const auto value_as_a = solver->evaluate_average_against_uniform(
      poker::Player::A, absl::GetFlag(FLAGS_deep_evaluation_samples));
  const auto value_as_b = solver->evaluate_average_against_uniform(
      poker::Player::B, absl::GetFlag(FLAGS_deep_evaluation_samples));
  if (!value_as_a.ok() || !value_as_b.ok()) {
    std::cerr << "Error: uniform-opponent evaluation failed\n";
    return 1;
  }
  const std::string model_output = absl::GetFlag(FLAGS_deep_model_output);
  if (!model_output.empty()) {
    const absl::Status saved = solver->save_average_model(model_output);
    if (!saved.ok()) {
      std::cerr << "Error: " << saved << '\n';
      return 1;
    }
  }
  const std::string opponent_path =
      absl::GetFlag(FLAGS_deep_opponent_policy);
  if (!opponent_path.empty()) {
    const auto opponent = poker::LoadPolicy(opponent_path);
    if (!opponent.ok()) {
      std::cerr << "Error: " << opponent.status() << '\n';
      return 1;
    }
    const auto as_a = solver->evaluate_average_against_policy(
        poker::Player::A, *opponent,
        absl::GetFlag(FLAGS_deep_evaluation_samples));
    const auto as_b = solver->evaluate_average_against_policy(
        poker::Player::B, *opponent,
        absl::GetFlag(FLAGS_deep_evaluation_samples));
    if (!as_a.ok() || !as_b.ok()) {
      std::cerr << "Error: tabular opponent evaluation failed\n";
      return 1;
    }
    std::cout << "average_vs_tabular_as_a=" << as_a->policy_player_value
              << '\n'
              << "average_vs_tabular_as_a_se=" << as_a->standard_error
              << '\n'
              << "average_vs_tabular_as_b=" << as_b->policy_player_value
              << '\n'
              << "average_vs_tabular_as_b_se=" << as_b->standard_error
              << '\n'
              << "tabular_policy_lookups="
              << as_a->opponent_policy_lookups +
                     as_b->opponent_policy_lookups
              << '\n'
              << "missing_tabular_policy_lookups="
              << as_a->missing_opponent_lookups +
                     as_b->missing_opponent_lookups
              << '\n';
  }
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - start;

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
            << "policy_parameter_bytes=" << stats.policy_parameter_bytes << '\n'
            << "average_value=" << *value << '\n'
            << "average_vs_uniform_as_a=" << *value_as_a << '\n'
            << "average_vs_uniform_as_b=" << *value_as_b << '\n'
            << "seconds=" << elapsed.count() << '\n';
  return std::isfinite(*value) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the heads-up poker solver.");
  absl::ParseCommandLine(argc, argv);
  const std::string algorithm = absl::GetFlag(FLAGS_algorithm);
  if (algorithm != "tabular" && algorithm != "deep") {
    std::cerr << "Error: --algorithm must be tabular or deep\n";
    return 1;
  }
  const int iterations = absl::GetFlag(FLAGS_iterations);
  if (iterations < 0 ||
      (iterations == 0 &&
       (algorithm != "deep" ||
        absl::GetFlag(FLAGS_deep_model_input).empty()))) {
    std::cerr << "Error: --iterations must be positive unless loading a Deep "
                 "CFR model\n";
    return 1;
  }
  const int64_t memory_limit_mb = absl::GetFlag(FLAGS_max_memory_mb);
  if (memory_limit_mb < 0) {
    std::cerr << "Error: --max_memory_mb must be non-negative\n";
    return 1;
  }
  const int threads = absl::GetFlag(FLAGS_threads);
  if (threads <= 0) {
    std::cerr << "Error: --threads must be positive\n";
    return 1;
  }
  if (algorithm == "deep" && threads != 1) {
    std::cerr << "Error: Deep CFR currently requires --threads=1\n";
    return 1;
  }
  const auto config = ConfigFromFlags();
  if (!config.ok()) {
    std::cerr << "Error: " << config.status() << "\n";
    return 1;
  }
  const poker::ComboRange range = poker::UniformComboRange();
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::Chips small_blind = absl::GetFlag(FLAGS_small_blind);
  const poker::ExactPublicState root = poker::MakeInitialState(
      config->betting_rules, {stack, stack},
      {small_blind, config->betting_rules.minimum_bet});
  SetMemoryLimit(memory_limit_mb);
  poker::SolveSpec spec{*config, root, {range, range}};
  return algorithm == "deep"
      ? RunDeep(std::move(spec), static_cast<uint64_t>(iterations))
      : RunTabular(std::move(spec), static_cast<uint64_t>(iterations), threads);
}
