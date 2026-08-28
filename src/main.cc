#include "src/deep_cfr.h"
#include "src/evaluation.h"
#include "src/neural_evaluation.h"
#include "src/neural_policy.h"
#include "src/policy_codec.h"
#include "src/solver.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
ABSL_FLAG(int64_t, max_memory_mb, 4096, "hard memory limit in MB; 0 is unlimited");
ABSL_FLAG(bool, external_sampling, false, "sample opponent actions during training");
ABSL_FLAG(std::string, public_abstraction, "texture", "exact, texture, or compact_texture");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36", "exact or handcrafted36");
ABSL_FLAG(std::string, private_recall, "auto", "auto, current, or history");
ABSL_FLAG(std::string, betting_abstraction, "default", "default or small_betting");
ABSL_FLAG(std::vector<std::string>, pot_fractions, {},
          "override pot fractions after calling for every street");
ABSL_FLAG(std::vector<std::string>, preflop_pot_fractions, {}, "preflop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, flop_pot_fractions, {}, "flop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, turn_pot_fractions, {}, "turn pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, river_pot_fractions, {}, "river pot fractions after calling");
ABSL_FLAG(std::string, policy_output, "", "output path for the trained compact tabular policy");
ABSL_FLAG(std::string, neural_policy_output, "", "output path for a fitted neural policy");
ABSL_FLAG(std::string, portable_neural_policy_output, "",
          "output path for browser-compatible neural policy weights");
ABSL_FLAG(std::string, neural_policy_input, "", "neural policy to load for Deep CFR evaluation");
ABSL_FLAG(std::string, neural_opponent_policy, "", "second neural policy for seat-swapped evaluation");
ABSL_FLAG(int, neural_steps, 2500, "optimizer steps for fitting the final neural policy");
ABSL_FLAG(int, neural_batch_size, 256, "batch size for neural training");
ABSL_FLAG(int, neural_hidden_size, 256, "hidden width for neural models");
ABSL_FLAG(double, neural_learning_rate, 1e-3, "learning rate for tabular policy neural approximation");
ABSL_FLAG(uint64_t, neural_seed, 1, "random seed for neural training");
ABSL_FLAG(int, deep_traversals_per_player, 1024, "Deep CFR traversals per player and iteration");
ABSL_FLAG(int, deep_training_steps, 750, "Deep CFR optimizer steps per network fit");
ABSL_FLAG(uint64_t, deep_memory_capacity, 100000, "Deep CFR capacity of each reservoir");
ABSL_FLAG(uint64_t, deep_cache_capacity, 4096, "Deep CFR maximum cached advantage-network predictions");
ABSL_FLAG(uint64_t, deep_policy_cache_capacity, 1'000'000,
          "Deep CFR maximum cached policy-network predictions");
ABSL_FLAG(int, evaluation_samples, 64, "deals sampled for neural policy evaluation");
ABSL_FLAG(uint64_t, best_response_iterations, 0,
          "one-sided CFR iterations per neural best response; 0 disables");
ABSL_FLAG(bool, best_response_external_sampling, true,
          "sample fixed-opponent actions during neural best responses");
ABSL_FLAG(uint64_t, deep_checkpoint_interval, 0,
          "save a numbered Deep CFR model every N iterations; 0 disables");
ABSL_FLAG(std::string, deep_opponent_policy, "", "tabular policy to evaluate against the Deep CFR model");

namespace {

std::filesystem::path CheckpointPath(const std::string& output, uint64_t iterations) {
  const std::filesystem::path path(output);
  return path.parent_path() /
         (path.stem().string() + "_i" + std::to_string(iterations) + path.extension().string());
}

void SetMemoryLimit(int64_t megabytes) {
  if (megabytes <= 0) return;
  const rlim_t bytes = static_cast<rlim_t>(megabytes) * 1024ULL * 1024ULL;
  const rlimit limit{bytes, bytes};
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    std::cerr << "Warning: failed to set memory limit to " << megabytes << " MB: " << std::strerror(errno) << "\n";
  }
}

absl::StatusOr<std::vector<double>> ParsePotFractions(const std::vector<std::string>& values) {
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
  config.external_sampling = absl::GetFlag(FLAGS_external_sampling);

  const std::string public_abstraction = absl::GetFlag(FLAGS_public_abstraction);
  if (public_abstraction == "exact") {
    config.card_abstraction.public_mode = poker::PublicCardMode::ExactCanonical;
  } else if (public_abstraction == "compact_texture") {
    config.card_abstraction.public_mode = poker::PublicCardMode::CompactTexture;
  } else if (public_abstraction != "texture") {
    return absl::InvalidArgumentError("invalid public abstraction");
  }

  const std::string private_abstraction = absl::GetFlag(FLAGS_private_abstraction);
  if (private_abstraction != "exact" && private_abstraction != "handcrafted36") {
    return absl::InvalidArgumentError("invalid private abstraction");
  }
  config.card_abstraction.private_kind = private_abstraction == "exact"
      ? poker::PrivateAbstractionKind::ExactCanonical
      : poker::PrivateAbstractionKind::Handcrafted36;

  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall != "auto" && recall != "current" && recall != "history") {
    return absl::InvalidArgumentError("invalid private recall mode");
  }
  const bool retain_bucket_history = recall == "history" ||
      (recall == "auto" && config.card_abstraction.private_kind == poker::PrivateAbstractionKind::Handcrafted36);
  config.card_abstraction.recall_mode = retain_bucket_history
      ? poker::RecallMode::BucketHistory
      : poker::RecallMode::CurrentBucketOnly;

  const std::string betting_abstraction = absl::GetFlag(FLAGS_betting_abstraction);
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
  return config;
}

void PrintRunSummary(const poker::TabularCfrSolver& solver, const poker::SolverConfig& config, double seconds) {
  const size_t info_sets = solver.info_set_count();
  const size_t history_nodes = solver.history_count();
  const uint64_t visits = solver.stats().decision_visits;

  std::printf("iterations=%" PRIu64 "\n", solver.iterations());
  std::printf("info_sets=%zu\n", info_sets);
  std::printf("max_info_sets=%d\n", config.max_info_sets);
  std::printf("info_set_cap_hit=%d\n", info_sets >= static_cast<size_t>(config.max_info_sets));
  std::printf("player_a_ev=%g\n", solver.expected_value(poker::Player::A));
  std::printf("seconds=%g\n", seconds);
  std::printf("history_nodes=%zu\n", history_nodes);
  std::printf("decision_visits=%" PRIu64 "\n", visits);
  if (seconds > 0.0) std::printf("decision_visits_per_second=%g\n", visits / seconds);
}

absl::Status RunTabular(poker::SolveSpec spec, uint64_t iterations, int threads) {
  auto solver = poker::TabularCfrSolver::Create(std::move(spec));
  if (!solver.ok()) return solver.status();
  const auto start = std::chrono::steady_clock::now();
  solver->run(iterations, threads);
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

  PrintRunSummary(*solver, solver->config(), elapsed.count());
  std::printf("threads=%d\n", threads);

  const std::string policy_output = absl::GetFlag(FLAGS_policy_output);
  const std::string neural_output = absl::GetFlag(FLAGS_neural_policy_output);
  if (policy_output.empty() && neural_output.empty()) return absl::OkStatus();

  const auto policy = solver->extract_average_policy();
  if (!policy_output.empty()) {
    const absl::Status saved = poker::SavePolicy(policy, policy_output);
    if (!saved.ok()) return saved;
  }
  if (!neural_output.empty()) {
    const auto fitted = poker::FitNeuralPolicy(
        solver->history(), solver->config().card_abstraction, solver->model(), policy,
        {.seed = absl::GetFlag(FLAGS_neural_seed),
         .steps = absl::GetFlag(FLAGS_neural_steps),
         .batch_size = absl::GetFlag(FLAGS_neural_batch_size),
         .hidden_size = absl::GetFlag(FLAGS_neural_hidden_size),
         .learning_rate = absl::GetFlag(FLAGS_neural_learning_rate)});
    if (!fitted.ok()) return fitted.status();
    const absl::Status saved = poker::SaveNeuralPolicy(fitted->policy, neural_output);
    if (!saved.ok()) return saved;
    std::printf("neural_policy_samples=%zu\n", fitted->samples);
    std::printf("neural_policy_loss=%g\n", fitted->loss);
    std::printf("neural_policy_parameter_bytes=%zu\n", fitted->policy.parameter_bytes());
    const auto value = poker::EstimateExpectedValue(
        solver->config(), solver->deals(), solver->history(), solver->initial_public(), solver->model(),
        fitted->policy, fitted->policy,
        static_cast<uint64_t>(absl::GetFlag(FLAGS_evaluation_samples)),
        absl::GetFlag(FLAGS_neural_seed));
    if (!value.ok()) return value.status();
    std::printf("neural_policy_value=%g\n", value->mean);
    std::printf("neural_policy_value_se=%g\n", value->standard_error);
    const uint64_t response_iterations = absl::GetFlag(FLAGS_best_response_iterations);
    if (response_iterations > 0) {
      poker::BestResponseConfig response_config{
          response_iterations,
          static_cast<uint64_t>(absl::GetFlag(FLAGS_evaluation_samples)),
          absl::GetFlag(FLAGS_neural_seed)};
      response_config.external_sampling = absl::GetFlag(FLAGS_best_response_external_sampling);
      const auto exploitability = poker::EstimateExploitability(
          solver->config(), solver->deals(), solver->history(), solver->initial_public(), solver->model(),
          fitted->policy, response_config);
      if (!exploitability.ok()) return exploitability.status();
      std::printf("neural_nash_conv=%g\n", exploitability->nash_conv);
      std::printf("neural_exploitability=%g\n", exploitability->exploitability);
    }
  }
  return absl::OkStatus();
}

absl::Status RunDeep(poker::SolveSpec spec, uint64_t iterations) {
  poker::DeepCfrConfig config;
  config.seed = absl::GetFlag(FLAGS_neural_seed);
  config.advantage_memory_capacity = absl::GetFlag(FLAGS_deep_memory_capacity);
  config.strategy_memory_capacity = absl::GetFlag(FLAGS_deep_memory_capacity);
  config.inference_cache_capacity = absl::GetFlag(FLAGS_deep_cache_capacity);
  config.policy_cache_capacity = absl::GetFlag(FLAGS_deep_policy_cache_capacity);
  config.traversals_per_player = absl::GetFlag(FLAGS_deep_traversals_per_player);
  config.training_steps = absl::GetFlag(FLAGS_deep_training_steps);
  config.policy_training_steps = absl::GetFlag(FLAGS_neural_steps);
  config.batch_size = absl::GetFlag(FLAGS_neural_batch_size);
  config.hidden_size = absl::GetFlag(FLAGS_neural_hidden_size);
  config.learning_rate = absl::GetFlag(FLAGS_neural_learning_rate);

  auto solver = poker::DeepCfrSolver::Create(std::move(spec), config);
  if (!solver.ok()) return solver.status();
  const std::string model_input = absl::GetFlag(FLAGS_neural_policy_input);
  if (!model_input.empty()) {
    const absl::Status loaded = solver->load_average_model(model_input);
    if (!loaded.ok()) return loaded;
  }
  const std::string model_output = absl::GetFlag(FLAGS_neural_policy_output);
  const uint64_t checkpoint_interval = absl::GetFlag(FLAGS_deep_checkpoint_interval);
  if (checkpoint_interval > 0 && model_output.empty()) {
    return absl::InvalidArgumentError("--deep_checkpoint_interval requires --neural_policy_output");
  }
  const auto start = std::chrono::steady_clock::now();
  uint64_t trained_iterations = 0;
  while (trained_iterations < iterations) {
    const uint64_t batch = checkpoint_interval == 0
        ? iterations
        : std::min(checkpoint_interval, iterations - trained_iterations);
    const absl::Status trained = solver->run(batch);
    if (!trained.ok()) return trained;
    trained_iterations += batch;
    if (checkpoint_interval > 0) {
      const auto path = CheckpointPath(model_output, trained_iterations);
      const absl::Status saved = solver->save_average_model(path);
      if (!saved.ok()) return saved;
      std::printf("deep_checkpoint=%s\n", path.string().c_str());
      std::fflush(stdout);
    }
  }
  const auto value = solver->evaluate_average(absl::GetFlag(FLAGS_evaluation_samples));
  if (!value.ok()) return value.status();
  const auto value_as_a = solver->evaluate_average_against_uniform(
      poker::Player::A, absl::GetFlag(FLAGS_evaluation_samples));
  if (!value_as_a.ok()) return value_as_a.status();
  const auto value_as_b = solver->evaluate_average_against_uniform(
      poker::Player::B, absl::GetFlag(FLAGS_evaluation_samples));
  if (!value_as_b.ok()) return value_as_b.status();
  const uint64_t response_iterations = absl::GetFlag(FLAGS_best_response_iterations);
  if (response_iterations > 0) {
    poker::BestResponseConfig response_config{
        response_iterations,
        static_cast<uint64_t>(absl::GetFlag(FLAGS_evaluation_samples)),
        absl::GetFlag(FLAGS_neural_seed)};
    response_config.external_sampling = absl::GetFlag(FLAGS_best_response_external_sampling);
    const auto estimate = solver->estimate_exploitability(response_config);
    if (!estimate.ok()) return estimate.status();
    const auto& a = estimate->player_a_response;
    const auto& b = estimate->player_b_response;
    std::printf("deep_best_response_a_value=%g\n", a.value);
    std::printf("deep_best_response_a_se=%g\n", a.standard_error);
    std::printf("deep_best_response_b_value=%g\n", b.value);
    std::printf("deep_best_response_b_se=%g\n", b.standard_error);
    std::printf("deep_best_response_a_info_sets=%zu\n", a.response_policy.rows.size());
    std::printf("deep_best_response_b_info_sets=%zu\n", b.response_policy.rows.size());
    std::printf("deep_policy_lookups=%" PRIu64 "\n", a.opponent_policy_lookups + b.opponent_policy_lookups);
    std::printf("deep_response_policy_lookups=%" PRIu64 "\n", a.response_policy_lookups + b.response_policy_lookups);
    std::printf("deep_missing_response_lookups=%" PRIu64 "\n",
                a.missing_response_lookups + b.missing_response_lookups);
    std::printf("deep_nash_conv=%g\n", estimate->nash_conv);
    std::printf("deep_exploitability=%g\n", estimate->exploitability);
    std::printf("deep_missing_policy_lookups=%" PRIu64 "\n",
                a.missing_opponent_lookups + b.missing_opponent_lookups);
  }
  if (!model_output.empty() && checkpoint_interval == 0) {
    const absl::Status saved = solver->save_average_model(model_output);
    if (!saved.ok()) return saved;
  }
  const std::string portable_output = absl::GetFlag(FLAGS_portable_neural_policy_output);
  if (!portable_output.empty()) {
    if (solver->average_policy() == nullptr) {
      return absl::FailedPreconditionError("no average neural policy to export");
    }
    const absl::Status saved = poker::SavePortableNeuralPolicy(*solver->average_policy(), portable_output);
    if (!saved.ok()) return saved;
  }
  const std::string opponent_path = absl::GetFlag(FLAGS_deep_opponent_policy);
  if (!opponent_path.empty()) {
    const auto opponent = poker::LoadPolicy(opponent_path);
    if (!opponent.ok()) return opponent.status();
    const auto as_a = solver->evaluate_against_policy(
        poker::Player::A, *opponent, poker::DeepCfrStrategy::Average,
        absl::GetFlag(FLAGS_evaluation_samples));
    if (!as_a.ok()) return as_a.status();
    const auto as_b = solver->evaluate_against_policy(
        poker::Player::B, *opponent, poker::DeepCfrStrategy::Average,
        absl::GetFlag(FLAGS_evaluation_samples));
    if (!as_b.ok()) return as_b.status();
    std::printf("average_vs_tabular_as_a=%g\n", as_a->policy_player_value);
    std::printf("average_vs_tabular_as_a_se=%g\n", as_a->standard_error);
    std::printf("average_vs_tabular_as_b=%g\n", as_b->policy_player_value);
    std::printf("average_vs_tabular_as_b_se=%g\n", as_b->standard_error);
    std::printf("tabular_policy_lookups=%" PRIu64 "\n",
                as_a->opponent_policy_lookups + as_b->opponent_policy_lookups);
    std::printf("missing_tabular_policy_lookups=%" PRIu64 "\n",
                as_a->missing_opponent_lookups + as_b->missing_opponent_lookups);
    std::printf("tabular_policy_lookups_when_deep_is_a=%" PRIu64 "\n", as_a->opponent_policy_lookups);
    std::printf("missing_tabular_lookups_when_deep_is_a=%" PRIu64 "\n", as_a->missing_opponent_lookups);
    std::printf("tabular_policy_lookups_when_deep_is_b=%" PRIu64 "\n", as_b->opponent_policy_lookups);
    std::printf("missing_tabular_lookups_when_deep_is_b=%" PRIu64 "\n", as_b->missing_opponent_lookups);
    if (iterations > 0) {
      const auto current_as_a = solver->evaluate_against_policy(
          poker::Player::A, *opponent, poker::DeepCfrStrategy::Current,
          absl::GetFlag(FLAGS_evaluation_samples));
      const auto current_as_b = solver->evaluate_against_policy(
          poker::Player::B, *opponent, poker::DeepCfrStrategy::Current,
          absl::GetFlag(FLAGS_evaluation_samples));
      if (current_as_a.ok() && current_as_b.ok()) {
        std::printf("current_vs_tabular_as_a=%g\n", current_as_a->policy_player_value);
        std::printf("current_vs_tabular_as_b=%g\n", current_as_b->policy_player_value);
      }
    }
  }
  const std::string neural_opponent_path = absl::GetFlag(FLAGS_neural_opponent_policy);
  if (!neural_opponent_path.empty()) {
    const auto opponent = poker::LoadNeuralPolicy(neural_opponent_path, solver->model());
    if (!opponent.ok()) return opponent.status();
    if (solver->average_policy() == nullptr) {
      return absl::FailedPreconditionError("no average neural policy to evaluate");
    }
    const uint64_t samples = static_cast<uint64_t>(absl::GetFlag(FLAGS_evaluation_samples));
    const uint64_t seed = absl::GetFlag(FLAGS_neural_seed);
    const auto as_a = poker::EstimateExpectedValue(
        solver->solver_config(), solver->deals(), solver->history(),
        solver->initial_public(), solver->model(), *solver->average_policy(),
        *opponent, samples, seed, false, true);
    if (!as_a.ok()) return as_a.status();
    const auto as_b = poker::EstimateExpectedValue(
        solver->solver_config(), solver->deals(), solver->history(),
        solver->initial_public(), solver->model(), *opponent,
        *solver->average_policy(), samples, seed, false, true);
    if (!as_b.ok()) return as_b.status();
    std::printf("neural_vs_neural_as_a=%g\n", as_a->mean);
    std::printf("neural_vs_neural_as_a_se=%g\n", as_a->standard_error);
    std::printf("neural_vs_neural_as_b=%g\n", -as_b->mean);
    std::printf("neural_vs_neural_as_b_se=%g\n", as_b->standard_error);
  }
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

  const poker::DeepCfrStats& stats = solver->stats();
  std::printf("iterations=%" PRIu64 "\n", stats.iterations);
  std::printf("traversals=%" PRIu64 "\n", stats.traversals);
  std::printf("advantage_samples_a=%zu\n", stats.advantage_samples[0]);
  std::printf("advantage_samples_b=%zu\n", stats.advantage_samples[1]);
  std::printf("strategy_samples=%zu\n", stats.strategy_samples);
  std::printf("advantage_loss_a=%g\n", stats.advantage_loss[0]);
  std::printf("advantage_loss_b=%g\n", stats.advantage_loss[1]);
  std::printf("strategy_loss=%g\n", stats.strategy_loss);
  std::printf("network_evaluations=%" PRIu64 "\n", stats.network_evaluations);
  std::printf("cache_hits=%" PRIu64 "\n", stats.cache_hits);
  std::printf("policy_parameter_bytes=%zu\n", stats.policy_parameter_bytes);
  std::printf("average_value=%g\n", *value);
  std::printf("average_vs_uniform_as_a=%g\n", *value_as_a);
  std::printf("average_vs_uniform_as_b=%g\n", *value_as_b);
  std::printf("seconds=%g\n", elapsed.count());
  return std::isfinite(*value)
             ? absl::OkStatus()
             : absl::InternalError("Deep CFR evaluation was not finite");
}

absl::Status RunFromFlags() {
  const std::string algorithm = absl::GetFlag(FLAGS_algorithm);
  if (algorithm != "tabular" && algorithm != "deep") {
    return absl::InvalidArgumentError("--algorithm must be tabular or deep");
  }
  const int iterations = absl::GetFlag(FLAGS_iterations);
  if (iterations < 0 ||
      (iterations == 0 &&
       (algorithm != "deep" ||
        absl::GetFlag(FLAGS_neural_policy_input).empty()))) {
    return absl::InvalidArgumentError(
        "--iterations must be positive unless loading a Deep CFR model");
  }
  if (absl::GetFlag(FLAGS_evaluation_samples) <= 0) {
    return absl::InvalidArgumentError("--evaluation_samples must be positive");
  }
  const int64_t memory_limit_mb = absl::GetFlag(FLAGS_max_memory_mb);
  if (memory_limit_mb < 0) {
    return absl::InvalidArgumentError("--max_memory_mb must be non-negative");
  }
  const int threads = absl::GetFlag(FLAGS_threads);
  if (threads <= 0) {
    return absl::InvalidArgumentError("--threads must be positive");
  }
  if (algorithm == "deep" && threads != 1) {
    return absl::InvalidArgumentError("Deep CFR currently requires --threads=1");
  }
  const auto config = ConfigFromFlags();
  if (!config.ok()) return config.status();
  const poker::ComboRange range = poker::UniformComboRange();
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::Chips small_blind = absl::GetFlag(FLAGS_small_blind);
  const poker::ExactPublicState root = poker::MakeInitialState(
      config->betting_rules, {stack, stack}, {small_blind, config->betting_rules.minimum_bet});
  SetMemoryLimit(memory_limit_mb);
  poker::SolveSpec spec{*config, root, {range, range}};
  return algorithm == "deep"
      ? RunDeep(std::move(spec), static_cast<uint64_t>(iterations))
      : RunTabular(std::move(spec), static_cast<uint64_t>(iterations), threads);
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the heads-up poker solver.");
  absl::ParseCommandLine(argc, argv);
  const absl::Status status = RunFromFlags();
  if (status.ok()) return 0;
  std::cerr << "Error: " << status.message() << '\n';
  return 1;
}
