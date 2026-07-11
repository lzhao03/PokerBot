#include "src/cfr_solver.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "src/hand_range.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>

namespace {

constexpr int64_t kDefaultMemoryLimitMb = 4096;

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

constexpr int64_t kDefaultMaxInfoSets = 500000;

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --iterations=N                 CFR iterations, default 100\n"
      << poker::kSolverOptionUsage
      << "  --max-memory-mb=N                hard memory limit in MB (default "
      << kDefaultMemoryLimitMb << ", 0 = unlimited)\n"
      << "  --log                           show INFO logs and VLOG(1) progress\n";
}

bool CapHit(size_t count, int cap) {
  return cap > 0 && count >= static_cast<size_t>(cap);
}

void PrintRunSummary(const poker::CFRSolver& solver,
                     const poker::SolverConfig& config,
                     double seconds) {
  const size_t info_sets = solver.get_info_set_count();
  const size_t history_nodes = solver.get_history_count();
  const uint64_t visits = solver.get_stats().decision_visits;

  std::cout << "iterations=" << solver.get_iterations_run() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets << "\n";
  std::cout << "info_set_cap_hit=" << CapHit(info_sets, config.max_info_sets)
            << "\n";
  std::cout << "player_a_ev=" << solver.get_expected_value(0) << "\n";
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
  absl::InitializeLog();

  poker::SolverConfig config;
  int iterations = 100;
  poker::SolverOptionState option_state;
  int64_t memory_limit_mb = kDefaultMemoryLimitMb;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--help") {
        PrintUsage(argv[0]);
        return 0;
      } else if (arg == "--log") {
        absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
        absl::SetGlobalVLogLevel(1);
      } else if (arg.starts_with("--iterations=")) {
        iterations = poker::ParseIntOption(
            arg.substr(sizeof("--iterations=") - 1), "--iterations");
      } else if (arg.starts_with("--max-memory-mb=")) {
        memory_limit_mb = poker::ParseInt64Option(
            arg.substr(sizeof("--max-memory-mb=") - 1), "--max-memory-mb");
        if (memory_limit_mb < 0) {
          throw std::invalid_argument("--max-memory-mb must be non-negative");
        }
      } else if (!poker::ApplySolverOption(arg, config, option_state)) {
        throw std::invalid_argument("Unknown option: " + std::string(arg));
      }
    }

    // Apply sensible memory caps when the user did not explicitly set them.
    if (!option_state.saw_max_info_sets) {
      config.max_info_sets = kDefaultMaxInfoSets;
    }
    const poker::ComboRange a_range = poker::UniformRange();
    const poker::ComboRange b_range = poker::UniformRange();

    SetMemoryLimit(memory_limit_mb);
    poker::CFRSolver solver(config);
    auto start = std::chrono::steady_clock::now();
    solver.run(iterations, a_range, b_range);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    PrintRunSummary(solver, config, elapsed.count());
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
