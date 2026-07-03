#include "src/cfr_solver.h"
#include "src/hand_range.h"
#include "src/poker.pb.h"

#include <chrono>
#include <cstdint>
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

struct BenchmarkResult {
  double result = 0.0;
  int64_t hands = 0;
  int64_t cfr_node_updates = 0;
  poker::CFRSolver::TraversalStats traversal_stats;
  poker::CFRSolver::UtilityCacheStats utility_cache_stats;
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

double RatePerSecond(int64_t count, double seconds) {
  if (count <= 0 || seconds <= 0.0) {
    return 0.0;
  }
  return count / seconds;
}

poker::CFRSolver::UtilityCacheStats CacheDelta(
    const poker::CFRSolver::UtilityCacheStats& after,
    const poker::CFRSolver::UtilityCacheStats& before) {
  return {after.hits - before.hits, after.misses - before.misses,
          after.entries};
}

void RunBenchmark(const std::string& name,
                  const std::function<BenchmarkResult()>& benchmark) {
  auto start = std::chrono::steady_clock::now();
  BenchmarkResult result = benchmark();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << name << "\t" << elapsed.count() << "\t" << result.result << "\t"
            << result.hands << "\t"
            << RatePerSecond(result.hands, elapsed.count()) << "\t"
            << result.cfr_node_updates << "\t"
            << RatePerSecond(result.cfr_node_updates, elapsed.count()) << "\t"
            << result.traversal_stats.preflop_updates << "\t"
            << result.traversal_stats.flop_updates << "\t"
            << result.traversal_stats.turn_updates << "\t"
            << result.traversal_stats.river_updates << "\t"
            << result.traversal_stats.max_decision_depth << "\t"
            << result.traversal_stats.canonical_state_visits << "\t"
            << result.traversal_stats.unique_canonical_states << "\t"
            << result.traversal_stats.duplicate_canonical_state_visits << "\t"
            << result.utility_cache_stats.hits << "\t"
            << result.utility_cache_stats.misses << "\t"
            << result.utility_cache_stats.entries << "\n";
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

    std::cout << "case\tseconds\tresult\thands\thands_per_second"
              << "\tcfr_node_updates\tcfr_node_updates_per_second"
              << "\tpreflop_updates\tflop_updates\tturn_updates"
              << "\triver_updates\tmax_decision_depth"
              << "\tcanonical_state_visits\tunique_canonical_states"
              << "\tduplicate_canonical_state_visits"
              << "\tutility_cache_hits\tutility_cache_misses"
              << "\tutility_cache_entries\n";

    RunBenchmark("range_expand", [&] {
      int64_t combos = 0;
      for (int i = 0; i < options.eval_samples; ++i) {
        combos += static_cast<int64_t>(
            player_a_range.get_all_weighted_combos().size());
      }
      return BenchmarkResult{static_cast<double>(combos), combos, 0};
    });

    RunBenchmark("train_deck", [&] {
      poker::CFRSolver solver(config);
      int64_t start_updates = solver.get_cfr_update_count();
      solver.run(options.iterations);
      int64_t updates = solver.get_cfr_update_count() - start_updates;
      return BenchmarkResult{
          static_cast<double>(
              solver.get_equilibrium_strategy().get_info_sets().size()),
          static_cast<int64_t>(options.iterations) * 2, updates,
          solver.get_traversal_stats(), solver.get_utility_cache_stats()};
    });

    RunBenchmark("train_range", [&] {
      poker::CFRSolver solver(config);
      int64_t start_updates = solver.get_cfr_update_count();
      solver.run(options.iterations, player_a_range, player_b_range);
      int64_t updates = solver.get_cfr_update_count() - start_updates;
      return BenchmarkResult{
          static_cast<double>(
              solver.get_equilibrium_strategy().get_info_sets().size()),
          options.iterations, updates, solver.get_traversal_stats(),
          solver.get_utility_cache_stats()};
    });

    poker::CFRSolver evaluate_solver(config);
    evaluate_solver.run(options.iterations, player_a_range, player_b_range);
    RunBenchmark("evaluate_range", [&] {
      poker::CFRSolver::UtilityCacheStats before =
          evaluate_solver.get_utility_cache_stats();
      double value = evaluate_solver.evaluate_strategy(
          options.eval_samples, player_a_range, player_b_range);
      return BenchmarkResult{
          value, options.eval_samples, 0, {},
          CacheDelta(evaluate_solver.get_utility_cache_stats(), before)};
    });

    if (!options.skip_exploitability) {
      poker::CFRSolver exploitability_solver(config);
      exploitability_solver.run(options.iterations, player_a_range,
                                player_b_range);
      RunBenchmark("best_response_player_a", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        double value = exploitability_solver.calculate_player_a_best_response_value(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples, 0, {},
            CacheDelta(exploitability_solver.get_utility_cache_stats(), before)};
      });
      RunBenchmark("best_response_player_b", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        double value = exploitability_solver.calculate_player_b_best_response_value(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples, 0, {},
            CacheDelta(exploitability_solver.get_utility_cache_stats(), before)};
      });
      RunBenchmark("exploitability_total", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        double value = exploitability_solver.calculate_exploitability(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples * 3, 0, {},
            CacheDelta(exploitability_solver.get_utility_cache_stats(), before)};
      });
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
