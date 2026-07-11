#include "src/cfr_solver.h"
#include "absl/log/initialize.h"
#include "src/build_flags.h"
#include "src/hand_range.h"
#include "src/poker_config.h"
#include "src/training_range.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using poker::kCoarsePublicBuckets;
using poker::kProdBenchmarkDefaults;

constexpr int kDefaultIterations = kProdBenchmarkDefaults ? 5000 : 100;
constexpr int kDefaultEvalSamples = kProdBenchmarkDefaults ? 1 : 100;
constexpr int kDefaultMaxDepth = 0;
constexpr int kDefaultMaxInfoSets = kProdBenchmarkDefaults ? 500000 : 0;
constexpr int kDefaultMaxPublicStates = kProdBenchmarkDefaults ? 200000 : 0;
constexpr const char* kDefaultRange =
    kProdBenchmarkDefaults ? "all" : "premium";

struct Options {
  int iterations = kDefaultIterations;
  int eval_samples = kDefaultEvalSamples;
  std::string range = kDefaultRange;
};

struct ParsedOptions {
  Options benchmark;
  poker::PokerConfig config;
};

struct BenchmarkResult {
  double result = 0.0;
  int64_t hands = 0;
  int64_t cfr_node_updates = 0;
  poker::CFRSolver::TrainingRunStats training_stats;
  poker::CFRSolver::TraversalStats traversal_stats;
  int64_t info_sets = 0;
  int64_t public_states = 0;
  int max_info_sets = 0;
  int max_public_states = 0;
  bool info_set_cap_hit = false;
  bool public_state_cap_hit = false;
};

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
      << "  --iterations=N                 default " << kDefaultIterations
      << "\n"
      << "  --eval-samples=N                default " << kDefaultEvalSamples
      << "\n"
      << "  --range=premium|all|RANGE       default " << kDefaultRange << "\n"
      << poker::kCommonSolverOptionUsage;
}

void PrintBenchmarkHeader() {
  std::cout << "case\tseconds\tresult"
            << "\tprimary_cfr_phase"
            << "\ttraversal_stats_enabled"
            << "\tprimary_cfr_node_updates"
            << "\tprimary_cfr_node_updates_per_second"
            << "\tprimary_cfr_node_updates_per_hand"
            << "\tcfr_node_updates\tcfr_node_updates_per_second"
            << "\tcfr_node_updates_per_hand"
            << "\tprebuild_seconds\tpublic_state_prebuild_complete"
            << "\tprebuild_public_states"
            << "\taction_transition_prebuild_complete"
            << "\tprebuild_action_transitions"
            << "\tmissing_action_transitions"
            << "\tchance_transition_prebuild_complete"
            << "\tprebuild_chance_transitions"
            << "\tmissing_chance_transitions"
            << "\tinfo_set_prebuild_seconds"
            << "\tinfo_set_prebuild_complete"
            << "\tprebuild_info_sets\tprebuild_action_entries"
            << "\tfrozen_info_set_lookup_prebuild_complete"
            << "\tprebuild_frozen_info_set_lookup_rows"
            << "\twarmup_seconds\twarmup_iterations"
            << "\twarmup_cfr_node_updates"
            << "\twarmup_cfr_node_updates_per_second"
            << "\twarmup_cfr_node_updates_per_hand"
            << "\tfrozen_seconds\tfrozen_iterations"
            << "\tfrozen_cfr_node_updates"
            << "\tfrozen_cfr_node_updates_per_second"
            << "\tfrozen_cfr_node_updates_per_hand"
            << "\thands\thands_per_second"
            << "\tinfo_sets\tpublic_states"
            << "\tmax_info_sets\tmax_public_states"
            << "\tinfo_set_cap_hit\tpublic_state_cap_hit"
            << "\taction_entry_touches\taction_entry_touches_per_second"
            << "\tatomic_regret_update_retries"
            << "\tpreflop_updates\tflop_updates\tturn_updates"
            << "\triver_updates\tmax_decision_depth"
            << "\tchild_nodes_created\tchance_samples"
            << "\tterminal_utility_calls\tfold_utility_calls"
            << "\tshowdown_utility_calls\n";
}

double RatePerSecond(int64_t count, double seconds) {
  if (count <= 0 || seconds <= 0.0) {
    return 0.0;
  }
  return count / seconds;
}

double RatePerHand(int64_t count, int64_t hands) {
  if (count <= 0 || hands <= 0) {
    return 0.0;
  }
  return static_cast<double>(count) / hands;
}

bool CapHit(int64_t count, int cap) {
  return cap > 0 && count >= cap;
}

bool RequireCompleteCoarseProdPrebuild(const poker::SolverConfig& config) {
  return kProdBenchmarkDefaults && kCoarsePublicBuckets &&
         config.num_training_threads > 1;
}

void AppendFailure(std::string* failure, const std::string& message) {
  if (!failure->empty()) {
    failure->append("; ");
  }
  failure->append(message);
}

std::string FrozenPrebuildFailure(
    const poker::CFRSolver::TrainingRunStats& stats) {
  std::string failure;
  if (!stats.public_state_prebuild_complete) {
    AppendFailure(&failure, "public-state prebuild incomplete");
  }
  if (!stats.chance_transition_prebuild_complete) {
    AppendFailure(&failure, "chance transition prebuild incomplete");
  }
  if (stats.missing_chance_transitions != 0) {
    AppendFailure(&failure, "missing chance transitions");
  }
  if (!stats.action_transition_prebuild_complete) {
    AppendFailure(&failure, "action transition prebuild incomplete");
  }
  if (stats.missing_action_transitions != 0) {
    AppendFailure(&failure, "missing action transitions");
  }
  if (!stats.info_set_prebuild_complete) {
    AppendFailure(&failure, "infoset prebuild incomplete");
  }
  if (!stats.frozen_info_set_lookup_prebuild_complete) {
    AppendFailure(&failure, "frozen infoset lookup prebuild incomplete");
  }
  return failure;
}

void RequireCompleteFrozenPrebuild(
    const poker::SolverConfig& config,
    const poker::CFRSolver::TrainingRunStats& stats) {
  if (!RequireCompleteCoarseProdPrebuild(config)) {
    return;
  }
  const std::string failure = FrozenPrebuildFailure(stats);
  if (!failure.empty()) {
    throw std::runtime_error("coarse prod benchmark requires complete frozen "
                             "prebuild: " +
                             failure);
  }
}

BenchmarkResult MakeBenchmarkResult(
    double result,
    int64_t hands,
    int64_t cfr_node_updates,
    poker::CFRSolver::TraversalStats traversal_stats = {}) {
  BenchmarkResult benchmark_result;
  benchmark_result.result = result;
  benchmark_result.hands = hands;
  benchmark_result.cfr_node_updates = cfr_node_updates;
  benchmark_result.traversal_stats = traversal_stats;
  return benchmark_result;
}

BenchmarkResult WithSolverState(BenchmarkResult result,
                                const poker::SolverConfig& config,
                                const poker::CFRSolver& solver) {
  result.info_sets = static_cast<int64_t>(solver.get_info_set_count());
  result.public_states = static_cast<int64_t>(solver.get_public_state_count());
  result.max_info_sets = config.max_info_sets;
  result.max_public_states = config.max_public_states;
  result.info_set_cap_hit = CapHit(result.info_sets, result.max_info_sets);
  result.public_state_cap_hit =
      CapHit(result.public_states, result.max_public_states);
  return result;
}

bool UseFrozenPrimaryMetric(
    const poker::CFRSolver::TrainingRunStats& training) {
  return training.frozen_iterations > 0;
}

const char* PrimaryCfrPhase(
    const poker::CFRSolver::TrainingRunStats& training) {
  return UseFrozenPrimaryMetric(training) ? "frozen" : "total";
}

int64_t PrimaryCfrNodeUpdates(const BenchmarkResult& result) {
  const auto& training = result.training_stats;
  return UseFrozenPrimaryMetric(training) ? training.frozen_cfr_updates
                                          : result.cfr_node_updates;
}

double PrimarySeconds(const BenchmarkResult& result, double elapsed_seconds) {
  const auto& training = result.training_stats;
  return UseFrozenPrimaryMetric(training) ? training.frozen_seconds
                                          : elapsed_seconds;
}

int64_t PrimaryHands(const BenchmarkResult& result) {
  const auto& training = result.training_stats;
  return UseFrozenPrimaryMetric(training) ? training.frozen_iterations
                                          : result.hands;
}

template <typename BenchmarkFn>
void RunBenchmark(const std::string& name, BenchmarkFn&& benchmark) {
  auto start = std::chrono::steady_clock::now();
  BenchmarkResult result = benchmark();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  const auto& training = result.training_stats;
  const int64_t primary_updates = PrimaryCfrNodeUpdates(result);
  const double primary_seconds = PrimarySeconds(result, elapsed.count());
  const int64_t primary_hands = PrimaryHands(result);
  std::cout << name << "\t" << elapsed.count() << "\t" << result.result << "\t"
            << PrimaryCfrPhase(training)
            << "\t" << poker::CFRSolver::traversal_stats_enabled() << "\t"
            << primary_updates << "\t"
            << RatePerSecond(primary_updates, primary_seconds) << "\t"
            << RatePerHand(primary_updates, primary_hands) << "\t"
            << result.cfr_node_updates << "\t"
            << RatePerSecond(result.cfr_node_updates, elapsed.count()) << "\t"
            << RatePerHand(result.cfr_node_updates, result.hands) << "\t"
            << training.prebuild_seconds << "\t"
            << training.public_state_prebuild_complete << "\t"
            << training.prebuild_public_states << "\t"
            << training.action_transition_prebuild_complete << "\t"
            << training.prebuild_action_transitions << "\t"
            << training.missing_action_transitions << "\t"
            << training.chance_transition_prebuild_complete << "\t"
            << training.prebuild_chance_transitions << "\t"
            << training.missing_chance_transitions << "\t"
            << training.info_set_prebuild_seconds << "\t"
            << training.info_set_prebuild_complete << "\t"
            << training.prebuild_info_sets << "\t"
            << training.prebuild_action_entries << "\t"
            << training.frozen_info_set_lookup_prebuild_complete << "\t"
            << training.prebuild_frozen_info_set_lookup_rows << "\t"
            << training.warmup_seconds << "\t"
            << training.warmup_iterations << "\t"
            << training.warmup_cfr_updates << "\t"
            << RatePerSecond(training.warmup_cfr_updates,
                             training.warmup_seconds) << "\t"
            << RatePerHand(training.warmup_cfr_updates,
                           training.warmup_iterations) << "\t"
            << training.frozen_seconds << "\t"
            << training.frozen_iterations << "\t"
            << training.frozen_cfr_updates << "\t"
            << RatePerSecond(training.frozen_cfr_updates,
                             training.frozen_seconds) << "\t"
            << RatePerHand(training.frozen_cfr_updates,
                           training.frozen_iterations) << "\t"
            << result.hands << "\t"
            << RatePerSecond(result.hands, elapsed.count()) << "\t"
            << result.info_sets << "\t"
            << result.public_states << "\t"
            << result.max_info_sets << "\t"
            << result.max_public_states << "\t"
            << result.info_set_cap_hit << "\t"
            << result.public_state_cap_hit << "\t"
            << result.traversal_stats.action_entry_touches << "\t"
            << RatePerSecond(result.traversal_stats.action_entry_touches,
                             elapsed.count()) << "\t"
            << result.traversal_stats.atomic_regret_update_retries << "\t"
            << result.traversal_stats.preflop_updates << "\t"
            << result.traversal_stats.flop_updates << "\t"
            << result.traversal_stats.turn_updates << "\t"
            << result.traversal_stats.river_updates << "\t"
            << result.traversal_stats.max_decision_depth << "\t"
            << result.traversal_stats.child_nodes_created << "\t"
            << result.traversal_stats.chance_samples << "\t"
            << result.traversal_stats.terminal_utility_calls << "\t"
            << result.traversal_stats.fold_utility_calls << "\t"
            << result.traversal_stats.showdown_utility_calls << "\n";
}

ParsedOptions ParseOptions(int argc, char** argv) {
  ParsedOptions parsed;
  parsed.config = poker::DefaultPokerConfig();
  poker::CommonOptionState option_state;
  if (kProdBenchmarkDefaults) {
    parsed.config.set_max_depth(kDefaultMaxDepth);
    parsed.config.set_regret_only_training(true);
    parsed.config.set_max_info_sets(kDefaultMaxInfoSets);
    parsed.config.set_max_public_states(kDefaultMaxPublicStates);
  }
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg.starts_with("--iterations=")) {
      parsed.benchmark.iterations = poker::ParseIntOption(
          arg.substr(sizeof("--iterations=") - 1), "--iterations");
    } else if (arg.starts_with("--eval-samples=")) {
      parsed.benchmark.eval_samples = poker::ParseIntOption(
          arg.substr(sizeof("--eval-samples=") - 1), "--eval-samples");
    } else if (arg.starts_with("--range=")) {
      parsed.benchmark.range =
          std::string(arg.substr(sizeof("--range=") - 1));
    } else if (!poker::ApplySolverOption(arg, parsed.config, option_state)) {
      throw std::invalid_argument("Unknown option: " + std::string(arg));
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
    poker::HandRange a_range = BenchmarkRange(options);
    poker::HandRange b_range = BenchmarkRange(options);

    std::cout << "recall_policy\t"
              << poker::RecallPolicyName(config.recall_policy) << "\n";
    PrintBenchmarkHeader();

    RunBenchmark("range_expand", [&] {
      int64_t combos = 0;
      for (int i = 0; i < options.eval_samples; ++i) {
        combos += static_cast<int64_t>(
            poker::BuildTrainingRange(a_range).active_count);
      }
      return MakeBenchmarkResult(static_cast<double>(combos), combos, 0);
    });

    RunBenchmark("train_range", [&] {
      poker::CFRSolver solver(config);
      int64_t start_updates = solver.get_cfr_update_count();
      solver.run(options.iterations, a_range, b_range);
      int64_t updates = solver.get_cfr_update_count() - start_updates;
      BenchmarkResult result = WithSolverState(MakeBenchmarkResult(
          static_cast<double>(solver.get_info_set_count()),
          options.iterations, updates, solver.get_traversal_stats()),
          config, solver);
      result.training_stats = solver.get_last_training_run_stats();
      RequireCompleteFrozenPrebuild(config, result.training_stats);
      return result;
    });

    poker::CFRSolver evaluate_solver(config);
    evaluate_solver.run(options.iterations, a_range, b_range);
    evaluate_solver.reset_traversal_stats();
    RunBenchmark("evaluate_range", [&] {
      double value = evaluate_solver.evaluate_strategy(
          options.eval_samples, a_range, b_range);
      return WithSolverState(MakeBenchmarkResult(
          value, options.eval_samples, 0,
          evaluate_solver.get_traversal_stats()),
          config, evaluate_solver);
    });
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
