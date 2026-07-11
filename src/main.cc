#include "src/solver.h"

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>

ABSL_FLAG(int, iterations, 100, "CFR iterations");
ABSL_FLAG(int, starting_stack, 100, "starting stack in chips");
ABSL_FLAG(int, small_blind, 1, "small blind in chips");
ABSL_FLAG(int, big_blind, 2, "big blind in chips");
ABSL_FLAG(int, chance_samples, 1, "chance samples per chance node");
ABSL_FLAG(int, max_info_sets, 500000, "maximum infosets");
ABSL_FLAG(int64_t, max_memory_mb, 4096,
          "hard memory limit in MB; 0 is unlimited");
ABSL_FLAG(bool, accumulate_average_strategy, true,
          "store the average strategy");
ABSL_FLAG(bool, log, false, "show INFO logs and VLOG(1) progress");
ABSL_FLAG(std::vector<std::string>, bet_sizes,
          std::vector<std::string>({"0.25", "0.5", "1.0"}),
          "comma-separated bet sizes for every street");
ABSL_FLAG(std::vector<std::string>, preflop_bet_sizes, {},
          "comma-separated preflop bet sizes");
ABSL_FLAG(std::vector<std::string>, flop_bet_sizes, {},
          "comma-separated flop bet sizes");
ABSL_FLAG(std::vector<std::string>, turn_bet_sizes, {},
          "comma-separated turn bet sizes");
ABSL_FLAG(std::vector<std::string>, river_bet_sizes, {},
          "comma-separated river bet sizes");

namespace {

void SetMemoryLimit(int64_t megabytes) {
  if (megabytes <= 0) {
    return;
  }
  const rlim_t bytes = static_cast<rlim_t>(megabytes) * 1024ULL * 1024ULL;
  struct rlimit limit;
  limit.rlim_cur = bytes;
  limit.rlim_max = bytes;
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    std::cerr << "Warning: failed to set memory limit to " << megabytes
              << " MB: " << std::strerror(errno) << "\n";
  }
}

absl::StatusOr<std::vector<double>> ParseBetSizes(
    const std::vector<std::string>& values) {
  std::vector<double> sizes;
  sizes.reserve(values.size());
  for (const std::string& value : values) {
    double size = 0.0;
    if (!absl::SimpleAtod(value, &size)) {
      return absl::InvalidArgumentError("invalid bet size: " + value);
    }
    sizes.push_back(size);
  }
  return sizes;
}

absl::Status OverrideBetSizes(poker::SolverConfigOptions& config,
                              poker::StreetKind street,
                              const std::vector<std::string>& values) {
  if (!values.empty()) {
    const auto sizes = ParseBetSizes(values);
    if (!sizes.ok()) {
      return sizes.status();
    }
    config.bet_sizes[static_cast<size_t>(street)] = *sizes;
  }
  return absl::OkStatus();
}

absl::StatusOr<poker::SolverConfig> ConfigFromFlags() {
  poker::SolverConfigOptions config;
  config.starting_stack = absl::GetFlag(FLAGS_starting_stack);
  config.small_blind = absl::GetFlag(FLAGS_small_blind);
  config.big_blind = absl::GetFlag(FLAGS_big_blind);
  config.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  config.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  config.accumulate_average_strategy =
      absl::GetFlag(FLAGS_accumulate_average_strategy);

  const auto sizes = ParseBetSizes(absl::GetFlag(FLAGS_bet_sizes));
  if (!sizes.ok()) {
    return sizes.status();
  }
  config.bet_sizes.fill(*sizes);
  for (const auto& [street, values] : {
           std::pair{poker::StreetKind::kPreflop,
                     absl::GetFlag(FLAGS_preflop_bet_sizes)},
           std::pair{poker::StreetKind::kFlop,
                     absl::GetFlag(FLAGS_flop_bet_sizes)},
           std::pair{poker::StreetKind::kTurn,
                     absl::GetFlag(FLAGS_turn_bet_sizes)},
           std::pair{poker::StreetKind::kRiver,
                     absl::GetFlag(FLAGS_river_bet_sizes)},
       }) {
    const absl::Status status = OverrideBetSizes(config, street, values);
    if (!status.ok()) {
      return status;
    }
  }
  return poker::SolverConfig::Create(std::move(config));
}

bool CapHit(size_t count, int cap) {
  return count >= static_cast<size_t>(cap);
}

void PrintRunSummary(const poker::CFRSolver& solver,
                     const poker::SolverConfig& config,
                     double seconds) {
  const size_t info_sets = solver.get_info_set_count();
  const size_t history_nodes = solver.get_history_count();
  const uint64_t visits = solver.get_stats().decision_visits;

  std::cout << "iterations=" << solver.get_iterations_run() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets() << "\n";
  std::cout << "info_set_cap_hit=" << CapHit(info_sets, config.max_info_sets())
            << "\n";
  std::cout << "player_a_ev=" << solver.get_expected_value(poker::Player::kA)
            << "\n";
  std::cout << "seconds=" << seconds << "\n";
  std::cout << "history_nodes=" << history_nodes << "\n";
  std::cout << "decision_visits=" << visits << "\n";
  if (seconds > 0.0) {
    std::cout << "decision_visits_per_second="
              << static_cast<double>(visits) / seconds << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the heads-up poker CFR solver.");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (absl::GetFlag(FLAGS_log)) {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    absl::SetGlobalVLogLevel(1);
  }

  const int64_t memory_limit_mb = absl::GetFlag(FLAGS_max_memory_mb);
  if (memory_limit_mb < 0) {
    std::cerr << "Error: --max_memory_mb must be non-negative\n";
    return 1;
  }
  const auto config = ConfigFromFlags();
  if (!config.ok()) {
    std::cerr << "Error: " << config.status() << "\n";
    return 1;
  }
  const poker::ComboRange a_range = poker::UniformRange();
  const poker::ComboRange b_range = poker::UniformRange();
  const auto deals = poker::DealDistribution::Create(a_range, b_range);
  if (!deals.ok()) {
    std::cerr << "Error: " << deals.status() << "\n";
    return 1;
  }

  SetMemoryLimit(memory_limit_mb);
  poker::CFRSolver solver(*config);
  const auto start = std::chrono::steady_clock::now();
  const poker::TrainingResult result =
      solver.run(absl::GetFlag(FLAGS_iterations), *deals);
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  PrintRunSummary(solver, *config, elapsed.count());
  std::cout << "stop_reason="
            << (result.stop_reason ==
                        poker::TrainingStopReason::kIterationsCompleted
                    ? "iterations_completed"
                    : "info_set_limit")
            << "\n";
  return 0;
}
