#include "src/cfr_solver.h"
#include "src/hand_range.h"
#include "src/poker.pb.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int iterations = 100;
  int max_depth = 2;
  int chance_samples = 1;
  int eval_samples = 100;
  int exploitability_samples = 10;
  bool skip_exploitability = false;
};

bool ConsumePrefix(const std::string& arg,
                   const std::string& prefix,
                   std::string* value) {
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  *value = arg.substr(prefix.size());
  return true;
}

int ParseInt(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    throw std::invalid_argument("Invalid integer for " + flag + ": " + value);
  }
  return static_cast<int>(parsed);
}

poker::PokerConfig BenchmarkConfig(const Options& options) {
  poker::PokerConfig config;
  config.add_bet_sizes(0.5);
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(20);
  config.set_small_blind(1);
  config.set_big_blind(2);
  config.set_max_depth(options.max_depth);
  config.set_chance_samples(options.chance_samples);
  config.set_max_raises_per_street(2);
  return config;
}

poker::HandRange BenchmarkRange() {
  poker::HandRange range;
  range.set_from_string("AA,KK,QQ,JJ,AKs,AQs,AKo");
  return range;
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --iterations=N                 default 100\n"
      << "  --max-depth=N                   default 2\n"
      << "  --chance-samples=N              default 1\n"
      << "  --eval-samples=N                default 100\n"
      << "  --exploitability-samples=N      default 10\n"
      << "  --skip-exploitability\n";
}

void RunBenchmark(const std::string& name,
                  const std::function<double()>& benchmark) {
  auto start = std::chrono::steady_clock::now();
  double result = benchmark();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << name << "\t" << elapsed.count() << "\t" << result << "\n";
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--skip-exploitability") {
      options.skip_exploitability = true;
    } else if (ConsumePrefix(arg, "--iterations=", &value)) {
      options.iterations = ParseInt(value, "--iterations");
    } else if (ConsumePrefix(arg, "--max-depth=", &value)) {
      options.max_depth = ParseInt(value, "--max-depth");
    } else if (ConsumePrefix(arg, "--chance-samples=", &value)) {
      options.chance_samples = ParseInt(value, "--chance-samples");
    } else if (ConsumePrefix(arg, "--eval-samples=", &value)) {
      options.eval_samples = ParseInt(value, "--eval-samples");
    } else if (ConsumePrefix(arg, "--exploitability-samples=", &value)) {
      options.exploitability_samples =
          ParseInt(value, "--exploitability-samples");
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = ParseOptions(argc, argv);
    poker::PokerConfig config = BenchmarkConfig(options);
    poker::HandRange player_a_range = BenchmarkRange();
    poker::HandRange player_b_range = BenchmarkRange();

    std::cout << "case\tseconds\tresult\n";

    RunBenchmark("range_expand", [&] {
      double combos = 0.0;
      for (int i = 0; i < options.eval_samples; ++i) {
        combos += player_a_range.get_all_weighted_combos().size();
      }
      return combos;
    });

    RunBenchmark("train_deck", [&] {
      poker::CFRSolver solver(config);
      solver.run(options.iterations);
      return solver.get_equilibrium_strategy().get_info_sets().size();
    });

    RunBenchmark("train_range", [&] {
      poker::CFRSolver solver(config);
      solver.run(options.iterations, player_a_range, player_b_range);
      return solver.get_equilibrium_strategy().get_info_sets().size();
    });

    RunBenchmark("evaluate_range", [&] {
      poker::CFRSolver solver(config);
      solver.run(options.iterations, player_a_range, player_b_range);
      return solver.evaluate_strategy(options.eval_samples, player_a_range,
                                      player_b_range);
    });

    if (!options.skip_exploitability) {
      RunBenchmark("exploitability_range", [&] {
        poker::CFRSolver solver(config);
        solver.run(options.iterations, player_a_range, player_b_range);
        return solver.calculate_exploitability(options.exploitability_samples,
                                               player_a_range, player_b_range);
      });
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
