#include "src/solver.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "absl/log/initialize.h"
#include "src/hand_range.h"

namespace {

#ifndef POKER_BENCHMARK_PROD_DEFAULTS
#define POKER_BENCHMARK_PROD_DEFAULTS 0
#endif

constexpr bool kProdBenchmarkDefaults =
    POKER_BENCHMARK_PROD_DEFAULTS != 0;

constexpr int kDefaultIterations = kProdBenchmarkDefaults ? 1 : 100;
constexpr int kDefaultEvalSamples = kProdBenchmarkDefaults ? 1 : 100;
constexpr const char* kDefaultRange = "premium";

struct Options {
  int iterations = kDefaultIterations;
  int eval_samples = kDefaultEvalSamples;
  std::string range = kDefaultRange;
};

poker::ComboRange BenchmarkRange(std::string_view text) {
  if (text == "premium") {
    return poker::ParseRange("AA,KK,QQ,JJ,AKs,AQs,AKo");
  }
  return text == "all" ? poker::UniformRange() : poker::ParseRange(text);
}

double Rate(double count, double seconds) {
  return seconds > 0.0 ? count / seconds : 0.0;
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program << " [options]\n"
            << "  --iterations=N                 default "
            << kDefaultIterations << "\n"
            << "  --eval-samples=N               default "
            << kDefaultEvalSamples << "\n"
            << "  --range=premium|all|RANGE      default "
            << kDefaultRange << "\n"
            << poker::kSolverOptionUsage;
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
  absl::InitializeLog();
  poker::SolverConfig config;
  poker::SolverOptionState option_state;
  Options options;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string_view argument = argv[i];
      if (argument == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      if (argument.starts_with("--iterations=")) {
        options.iterations = poker::ParseIntOption(
            argument.substr(sizeof("--iterations=") - 1), "--iterations");
      } else if (argument.starts_with("--eval-samples=")) {
        options.eval_samples = poker::ParseIntOption(
            argument.substr(sizeof("--eval-samples=") - 1),
            "--eval-samples");
      } else if (argument.starts_with("--range=")) {
        options.range = argument.substr(sizeof("--range=") - 1);
      } else if (!poker::ApplySolverOption(argument, config, option_state)) {
        throw std::invalid_argument("Unknown option: " +
                                    std::string(argument));
      }
    }

    const poker::ComboRange a_range = BenchmarkRange(options.range);
    const poker::ComboRange b_range = BenchmarkRange(options.range);

    std::cout << "case\tseconds\tresult\n";
    Measure("range_expand", [&] {
      return a_range.count();
    });

    std::unique_ptr<poker::CFRSolver> solver;
    Measure("build_history", [&] {
      solver = std::make_unique<poker::CFRSolver>(config);
      return solver->get_history_count();
    });
    const double training_seconds = Measure("train_range", [&] {
      solver->run(options.iterations, a_range, b_range);
      return solver->get_expected_value(0);
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
      return solver->evaluate_strategy(
          options.eval_samples, a_range, b_range,
          config.accumulate_average_strategy
              ? poker::StrategySource::kAverage
              : poker::StrategySource::kCurrent);
    });
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
  return 0;
}
