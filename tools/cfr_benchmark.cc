#include "src/solver.h"
#include "src/evaluation.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"

#ifndef POKER_BENCHMARK_PROD_DEFAULTS
#define POKER_BENCHMARK_PROD_DEFAULTS 0
#endif

constexpr bool kProdBenchmarkDefaults =
    POKER_BENCHMARK_PROD_DEFAULTS != 0;

constexpr int kDefaultIterations = kProdBenchmarkDefaults ? 1 : 100;
constexpr int kDefaultEvalSamples = kProdBenchmarkDefaults ? 1 : 100;
constexpr const char* kDefaultRange = "premium";

ABSL_FLAG(int, iterations, kDefaultIterations, "CFR iterations");
ABSL_FLAG(int, eval_samples, kDefaultEvalSamples, "evaluation samples");
ABSL_FLAG(std::string, range, kDefaultRange,
          "premium, all, or a poker range");
ABSL_FLAG(double, training_seconds, 0.0,
          "train for this wall-clock duration; 0 uses iterations");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36",
          "handcrafted36 or equity");
ABSL_FLAG(std::string, private_recall, "auto",
          "auto, current, or history");
ABSL_FLAG(std::string, equity_model, "", "equity model path");
ABSL_FLAG(uint64_t, evaluation_seed, 1, "policy evaluation seed");
ABSL_FLAG(uint64_t, best_response_iterations, 0,
          "approximate best-response iterations; 0 disables it");
ABSL_FLAG(int, starting_stack, 100, "starting stack in chips");
ABSL_FLAG(int, max_info_sets, 500000, "maximum infosets");
ABSL_FLAG(int, chance_samples, 1, "chance samples per chance node");

namespace {

absl::StatusOr<poker::ComboRange> BenchmarkRange(std::string_view text) {
  if (text == "premium") {
    return poker::ParseRange("AA,KK,QQ,JJ,AKs,AQs,AKo");
  }
  if (text == "all") {
    return poker::UniformComboRange();
  }
  return poker::ParseRange(text);
}

double Rate(double count, double seconds) {
  return seconds > 0.0 ? count / seconds : 0.0;
}

absl::StatusOr<poker::SolverConfig> BenchmarkConfig() {
  poker::SolverConfigOptions options;
  options.starting_stack = absl::GetFlag(FLAGS_starting_stack);
  options.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  options.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  const std::string kind = absl::GetFlag(FLAGS_private_abstraction);
  if (kind == "handcrafted36") {
    options.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::Handcrafted36;
  } else if (kind == "equity") {
    options.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::EquityPotential;
    const std::string path = absl::GetFlag(FLAGS_equity_model);
    if (path.empty()) {
      return absl::InvalidArgumentError(
          "--equity_model is required for equity abstraction");
    }
    auto model = poker::LoadEquityBucketModel(path);
    if (!model.ok()) return model.status();
    options.card_abstraction.equity_model = std::move(*model);
  } else {
    return absl::InvalidArgumentError("invalid private abstraction");
  }
  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall == "auto") {
    options.card_abstraction.recall_mode =
        kind == "equity" ? poker::RecallMode::CurrentBucketOnly
                         : poker::RecallMode::BucketHistory;
  } else if (recall == "current") {
    options.card_abstraction.recall_mode =
        poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "history") {
    options.card_abstraction.recall_mode =
        poker::RecallMode::BucketHistory;
  } else {
    return absl::InvalidArgumentError("invalid private recall mode");
  }
  return poker::SolverConfig::Create(std::move(options));
}

template <typename Function>
double Measure(std::string_view name, Function function) {
  const auto start = std::chrono::steady_clock::now();
  const auto result = function();
  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << name << '\t' << seconds << '\t' << result << '\n';
  return seconds;
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Benchmark the heads-up poker CFR solver.");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const auto config_result = BenchmarkConfig();
  if (!config_result.ok()) {
    std::cerr << "Error: " << config_result.status() << '\n';
    return 1;
  }
  const poker::SolverConfig config = *config_result;

  const std::string range = absl::GetFlag(FLAGS_range);
  const auto parsed_range = BenchmarkRange(range);
  if (!parsed_range.ok()) {
    std::cerr << "Error: " << parsed_range.status() << '\n';
    return 1;
  }
  const poker::ComboRange a_range = *parsed_range;
  const poker::ComboRange b_range = *parsed_range;
  const poker::Chips stack = config.starting_stack();
  const poker::ExactPublicState root = poker::MakeInitialState(
      poker::BettingRules{config.big_blind()}, {stack, stack},
      {config.small_blind(), config.big_blind()});

  std::cout << "case\tseconds\tresult\n";
  Measure("range_expand", [&] { return a_range.count(); });

  std::unique_ptr<poker::CFRSolver> solver;
  std::string build_error;
  Measure("build_history", [&] {
    auto result = poker::CFRSolver::Create(
        {config, root, {a_range, b_range}});
    if (!result.ok()) {
      build_error = result.status().ToString();
      return size_t{0};
    }
    solver = std::move(*result);
    return solver->get_history_count();
  });
  if (!build_error.empty()) {
    std::cerr << "Error: " << build_error << '\n';
    return 1;
  }
  const double training_seconds = Measure("train_range", [&] {
    const double seconds = absl::GetFlag(FLAGS_training_seconds);
    if (seconds <= 0.0) {
      solver->run(absl::GetFlag(FLAGS_iterations));
    } else {
      const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::duration<double>(seconds);
      while (std::chrono::steady_clock::now() < deadline) {
        if (solver->run(1).stop_reason ==
            poker::TrainingStopReason::InfoSetLimit) {
          break;
        }
      }
    }
    return solver->get_expected_value(poker::Player::A);
  });
  const auto training = solver->get_stats();
  std::cout << "iterations\t" << solver->get_iterations_run() << '\n'
            << "decision_visits\t" << training.decision_visits << '\n'
            << "decision_visits_per_second\t"
            << Rate(training.decision_visits, training_seconds) << '\n'
            << "chance_samples\t" << training.chance_samples << '\n'
            << "terminal_visits\t" << training.terminal_visits << '\n'
            << "infosets\t" << solver->get_info_set_count() << '\n'
            << "history_nodes\t" << solver->get_history_count() << '\n'
            << "regret_bytes\t" << solver->get_regret_bytes() << '\n'
            << "strategy_bytes\t" << solver->get_strategy_bytes() << '\n';
  std::cout << "equity_cache_entries\t"
            << solver->card_abstraction().cache_size() << '\n';

  solver->reset_stats();
  Measure("evaluate_range", [&] {
    if (!config.accumulate_average_strategy()) {
      return solver->evaluate_current(
          absl::GetFlag(FLAGS_eval_samples));
    }
    const auto value = solver->evaluate_average(
        absl::GetFlag(FLAGS_eval_samples));
    return value.ok() ? *value : 0.0;
  });

  const auto policy = solver->extract_average_policy();
  if (policy.ok()) {
    const auto profile = poker::EstimateExpectedValue(
        *solver, *policy, *policy,
        static_cast<uint64_t>(absl::GetFlag(FLAGS_eval_samples)),
        absl::GetFlag(FLAGS_evaluation_seed));
    if (profile.ok()) {
      std::cout << "policy_ev\t" << profile->mean << '\n'
                << "policy_standard_error\t" << profile->standard_error
                << '\n'
                << "policy_lookups\t" << profile->policy_lookups << '\n'
                << "missing_policy_lookups\t"
                << profile->missing_policy_lookups << '\n';
    }
    const uint64_t response_iterations =
        absl::GetFlag(FLAGS_best_response_iterations);
    if (response_iterations > 0) {
      const auto exploitability = poker::EstimateExploitability(
          *solver, *policy,
          {response_iterations,
           static_cast<uint64_t>(absl::GetFlag(FLAGS_eval_samples)),
           absl::GetFlag(FLAGS_evaluation_seed)});
      if (exploitability.ok()) {
        std::cout << "nash_conv\t" << exploitability->nash_conv << '\n'
                  << "exploitability\t" << exploitability->exploitability
                  << '\n'
                  << "missing_response_lookups\t"
                  << exploitability->player_a_response
                             .missing_opponent_lookups +
                         exploitability->player_b_response
                             .missing_opponent_lookups
                  << '\n';
      }
    }
  }
  return 0;
}
