#include "src/solver.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"

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
  const poker::SolverConfig config = poker::SolverConfig::Default();

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
    solver->run(absl::GetFlag(FLAGS_iterations));
    return solver->get_expected_value(poker::Player::kA);
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
  return 0;
}
