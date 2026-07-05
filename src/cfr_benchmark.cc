#include "src/cfr_solver.h"
#include "src/cfr_solver_proto_adapter.h"
#include "absl/log/initialize.h"
#include "src/hand_range.h"
#include "src/poker_config.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef POKER_BENCHMARK_PROD_DEFAULTS
#define POKER_BENCHMARK_PROD_DEFAULTS 0
#endif

constexpr bool kProdBenchmarkDefaults = POKER_BENCHMARK_PROD_DEFAULTS != 0;
constexpr int kDefaultIterations = kProdBenchmarkDefaults ? 5000 : 100;
constexpr int kDefaultEvalSamples = kProdBenchmarkDefaults ? 1 : 100;
constexpr int kDefaultExploitabilitySamples = 10;
constexpr int kDefaultMaxInfoSets = kProdBenchmarkDefaults ? 500000 : 0;
constexpr int kDefaultMaxTreeNodes = kProdBenchmarkDefaults ? 200000 : 0;
constexpr bool kDefaultSkipExploitability = kProdBenchmarkDefaults;
constexpr const char* kDefaultRange =
    kProdBenchmarkDefaults ? "all" : "premium";

struct Options {
  int iterations = kDefaultIterations;
  int eval_samples = kDefaultEvalSamples;
  int exploitability_samples = kDefaultExploitabilitySamples;
  std::string range = kDefaultRange;
  bool skip_exploitability = kDefaultSkipExploitability;
};

struct ParsedOptions {
  Options benchmark;
  poker::PokerConfig config;
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

double ParseDouble(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::invalid_argument("Invalid number for " + flag + ": " + value);
  }
  return parsed;
}

poker::HandRange BenchmarkRange(const Options& options) {
  poker::HandRange range;
  if (options.range == "premium") {
    range.set_from_string("AA,KK,QQ,JJ,AKs,AQs,AKo");
  } else if (options.range == "all") {
    range.set_uniform_range();
  } else {
    range.set_from_string(options.range);
  }
  return range;
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --config=PATH                  binary PokerConfig protobuf\n"
      << "  --iterations=N                 default " << kDefaultIterations
      << "\n"
      << "  --eval-samples=N                default " << kDefaultEvalSamples
      << "\n"
      << "  --exploitability-samples=N      default "
      << kDefaultExploitabilitySamples << "\n"
      << "  --range=premium|all|RANGE       default " << kDefaultRange << "\n"
      << "  --skip-exploitability           default "
      << (kDefaultSkipExploitability ? "true" : "false") << "\n"
      << "  --starting-stack=N              solver config override\n"
      << "  --small-blind=N                 solver config override\n"
      << "  --big-blind=N                   solver config override\n"
      << "  --max-depth=N                   solver config override\n"
      << "  --chance-samples=N              solver config override\n"
      << "  --max-info-sets=N               solver config override"
      << " (default " << kDefaultMaxInfoSets << ")\n"
      << "  --max-tree-nodes=N              solver config override"
      << " (default " << kDefaultMaxTreeNodes << ")\n"
      << "  --threads=N                     solver config override\n"
      << "  --warmup-iterations=N           solver config override\n"
      << "  --regret-only                   solver config override\n"
      << "  --bet-size=X                    replaces default global sizes on first use\n"
      << "  --preflop-bet-size=X            solver config override\n"
      << "  --flop-bet-size=X               solver config override\n"
      << "  --turn-bet-size=X               solver config override\n"
      << "  --river-bet-size=X              solver config override\n";
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

poker::CFRSolver::TraversalStats TraversalDelta(
    const poker::CFRSolver::TraversalStats& after,
    const poker::CFRSolver::TraversalStats& before) {
  poker::CFRSolver::TraversalStats delta;
  delta.cfr_updates = after.cfr_updates - before.cfr_updates;
  delta.preflop_updates = after.preflop_updates - before.preflop_updates;
  delta.flop_updates = after.flop_updates - before.flop_updates;
  delta.turn_updates = after.turn_updates - before.turn_updates;
  delta.river_updates = after.river_updates - before.river_updates;
  delta.child_nodes_created =
      after.child_nodes_created - before.child_nodes_created;
  delta.chance_samples = after.chance_samples - before.chance_samples;
  delta.terminal_utility_calls =
      after.terminal_utility_calls - before.terminal_utility_calls;
  delta.fold_utility_calls =
      after.fold_utility_calls - before.fold_utility_calls;
  delta.showdown_utility_calls =
      after.showdown_utility_calls - before.showdown_utility_calls;
  delta.action_entry_touches =
      after.action_entry_touches - before.action_entry_touches;
  return delta;
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
            << result.traversal_stats.action_entry_touches << "\t"
            << RatePerSecond(result.traversal_stats.action_entry_touches,
                             elapsed.count()) << "\t"
            << result.traversal_stats.preflop_updates << "\t"
            << result.traversal_stats.flop_updates << "\t"
            << result.traversal_stats.turn_updates << "\t"
            << result.traversal_stats.river_updates << "\t"
            << result.traversal_stats.max_decision_depth << "\t"
            << result.traversal_stats.child_nodes_created << "\t"
            << result.traversal_stats.chance_samples << "\t"
            << result.traversal_stats.terminal_utility_calls << "\t"
            << result.traversal_stats.fold_utility_calls << "\t"
            << result.traversal_stats.showdown_utility_calls << "\t"
            << result.utility_cache_stats.hits << "\t"
            << result.utility_cache_stats.misses << "\t"
            << result.utility_cache_stats.entries << "\n";
}

ParsedOptions ParseOptions(int argc, char** argv) {
  ParsedOptions parsed;
  parsed.config = poker::DefaultPokerConfig();
  bool saw_global_bet_size = false;
  if (kProdBenchmarkDefaults) {
    parsed.config.set_max_depth(0);
    parsed.config.set_regret_only_training(true);
    parsed.config.set_max_info_sets(kDefaultMaxInfoSets);
    parsed.config.set_max_tree_nodes(kDefaultMaxTreeNodes);
  }
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--skip-exploitability") {
      parsed.benchmark.skip_exploitability = true;
    } else if (arg == "--regret-only") {
      parsed.config.set_regret_only_training(true);
    } else if (ConsumePrefix(arg, "--config=", &value)) {
      poker::LoadPokerConfig(value, &parsed.config);
    } else if (ConsumePrefix(arg, "--iterations=", &value)) {
      parsed.benchmark.iterations = ParseInt(value, "--iterations");
    } else if (ConsumePrefix(arg, "--max-depth=", &value)) {
      parsed.config.set_max_depth(ParseInt(value, "--max-depth"));
    } else if (ConsumePrefix(arg, "--chance-samples=", &value)) {
      parsed.config.set_chance_samples(ParseInt(value, "--chance-samples"));
    } else if (ConsumePrefix(arg, "--eval-samples=", &value)) {
      parsed.benchmark.eval_samples = ParseInt(value, "--eval-samples");
    } else if (ConsumePrefix(arg, "--exploitability-samples=", &value)) {
      parsed.benchmark.exploitability_samples =
          ParseInt(value, "--exploitability-samples");
    } else if (ConsumePrefix(arg, "--range=", &value)) {
      parsed.benchmark.range = value;
    } else if (ConsumePrefix(arg, "--starting-stack=", &value)) {
      parsed.config.set_starting_stack_size(ParseInt(value, "--starting-stack"));
    } else if (ConsumePrefix(arg, "--small-blind=", &value)) {
      parsed.config.set_small_blind(ParseInt(value, "--small-blind"));
    } else if (ConsumePrefix(arg, "--big-blind=", &value)) {
      parsed.config.set_big_blind(ParseInt(value, "--big-blind"));
    } else if (ConsumePrefix(arg, "--max-info-sets=", &value)) {
      parsed.config.set_max_info_sets(ParseInt(value, "--max-info-sets"));
    } else if (ConsumePrefix(arg, "--max-tree-nodes=", &value)) {
      parsed.config.set_max_tree_nodes(ParseInt(value, "--max-tree-nodes"));
    } else if (ConsumePrefix(arg, "--threads=", &value)) {
      parsed.config.set_num_training_threads(ParseInt(value, "--threads"));
    } else if (ConsumePrefix(arg, "--warmup-iterations=", &value)) {
      parsed.config.set_warmup_iterations(
          ParseInt(value, "--warmup-iterations"));
    } else if (ConsumePrefix(arg, "--bet-size=", &value)) {
      if (!saw_global_bet_size) {
        parsed.config.clear_bet_sizes();
        saw_global_bet_size = true;
      }
      parsed.config.add_bet_sizes(ParseDouble(value, "--bet-size"));
    } else if (ConsumePrefix(arg, "--preflop-bet-size=", &value)) {
      poker::AddBetSize(&parsed.config, poker::Street::PREFLOP,
                        ParseDouble(value, "--preflop-bet-size"));
    } else if (ConsumePrefix(arg, "--flop-bet-size=", &value)) {
      poker::AddBetSize(&parsed.config, poker::Street::FLOP,
                        ParseDouble(value, "--flop-bet-size"));
    } else if (ConsumePrefix(arg, "--turn-bet-size=", &value)) {
      poker::AddBetSize(&parsed.config, poker::Street::TURN,
                        ParseDouble(value, "--turn-bet-size"));
    } else if (ConsumePrefix(arg, "--river-bet-size=", &value)) {
      poker::AddBetSize(&parsed.config, poker::Street::RIVER,
                        ParseDouble(value, "--river-bet-size"));
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }
  return parsed;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();

  try {
    ParsedOptions parsed = ParseOptions(argc, argv);
    Options options = parsed.benchmark;
    poker::SolverConfig config = poker::SolverConfigFromProto(parsed.config);
    poker::HandRange player_a_range = BenchmarkRange(options);
    poker::HandRange player_b_range = BenchmarkRange(options);

    std::cout << "case\tseconds\tresult\thands\thands_per_second"
              << "\tcfr_node_updates\tcfr_node_updates_per_second"
              << "\taction_entry_touches\taction_entry_touches_per_second"
              << "\tpreflop_updates\tflop_updates\tturn_updates"
              << "\triver_updates\tmax_decision_depth"
              << "\tchild_nodes_created\tchance_samples"
              << "\tterminal_utility_calls\tfold_utility_calls"
              << "\tshowdown_utility_calls"
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

    RunBenchmark("train_range", [&] {
      poker::CFRSolver solver(config);
      int64_t start_updates = solver.get_cfr_update_count();
      solver.run(options.iterations, player_a_range, player_b_range);
      int64_t updates = solver.get_cfr_update_count() - start_updates;
      return BenchmarkResult{
          static_cast<double>(solver.get_info_set_count()),
          options.iterations, updates, solver.get_traversal_stats(),
          solver.get_utility_cache_stats()};
    });

    poker::CFRSolver evaluate_solver(config);
    evaluate_solver.run(options.iterations, player_a_range, player_b_range);
    RunBenchmark("evaluate_range", [&] {
      poker::CFRSolver::UtilityCacheStats before =
          evaluate_solver.get_utility_cache_stats();
      poker::CFRSolver::TraversalStats traversal_before =
          evaluate_solver.get_traversal_stats();
      double value = evaluate_solver.evaluate_strategy(
          options.eval_samples, player_a_range, player_b_range);
      return BenchmarkResult{
          value, options.eval_samples, 0,
          TraversalDelta(evaluate_solver.get_traversal_stats(),
                         traversal_before),
          CacheDelta(evaluate_solver.get_utility_cache_stats(), before)};
    });

    if (!options.skip_exploitability) {
      poker::CFRSolver exploitability_solver(config);
      exploitability_solver.run(options.iterations, player_a_range,
                                player_b_range);
      RunBenchmark("best_response_player_a", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        poker::CFRSolver::TraversalStats traversal_before =
            exploitability_solver.get_traversal_stats();
        double value = exploitability_solver.calculate_player_a_best_response_value(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples, 0,
            TraversalDelta(exploitability_solver.get_traversal_stats(),
                           traversal_before),
            CacheDelta(exploitability_solver.get_utility_cache_stats(), before)};
      });
      RunBenchmark("best_response_player_b", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        poker::CFRSolver::TraversalStats traversal_before =
            exploitability_solver.get_traversal_stats();
        double value = exploitability_solver.calculate_player_b_best_response_value(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples, 0,
            TraversalDelta(exploitability_solver.get_traversal_stats(),
                           traversal_before),
            CacheDelta(exploitability_solver.get_utility_cache_stats(), before)};
      });
      RunBenchmark("exploitability_total", [&] {
        poker::CFRSolver::UtilityCacheStats before =
            exploitability_solver.get_utility_cache_stats();
        poker::CFRSolver::TraversalStats traversal_before =
            exploitability_solver.get_traversal_stats();
        double value = exploitability_solver.calculate_exploitability(
            options.exploitability_samples, player_a_range, player_b_range);
        return BenchmarkResult{
            value, options.exploitability_samples * 3, 0,
            TraversalDelta(exploitability_solver.get_traversal_stats(),
                           traversal_before),
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
