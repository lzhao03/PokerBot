#include "src/cfr_solver.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "src/hand_range.h"
#include "src/poker_config.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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

int64_t ParseInt64(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  long long parsed = std::strtoll(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    throw std::invalid_argument("Invalid integer for " + flag + ": " + value);
  }
  return static_cast<int64_t>(parsed);
}

double ParseDouble(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::invalid_argument("Invalid number for " + flag + ": " + value);
  }
  return parsed;
}

constexpr int64_t kDefaultMaxInfoSets = 500000;
constexpr int64_t kDefaultMaxPublicStates = 200000;

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --config=PATH                  binary PokerConfig protobuf\n"
      << "  --iterations=N                 CFR iterations, default 100\n"
      << "  --starting-stack=N\n"
      << "  --small-blind=N\n"
      << "  --big-blind=N\n"
      << "  --max-depth=N                   0 disables depth cutoff\n"
      << "  --chance-samples=N\n"
      << "  --bet-size=X                    replaces default global sizes on first use\n"
      << "  --preflop-bet-size=X\n"
      << "  --flop-bet-size=X\n"
      << "  --turn-bet-size=X\n"
      << "  --river-bet-size=X\n"
      << "  --max-info-sets=N               cap info set allocations (default "
      << kDefaultMaxInfoSets << ", 0 = unlimited)\n"
      << "  --max-public-states=N           cap graph nodes (default "
      << kDefaultMaxPublicStates << ", 0 = unlimited)\n"
      << "  --threads=N                     parallel training threads (0 or 1 = single-threaded)\n"
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
  const size_t public_states = solver.get_public_state_count();
  const int64_t touches = solver.get_traversal_stats().action_entry_touches;

  std::cout << "iterations=" << solver.get_iterations_run() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets << "\n";
  std::cout << "info_set_cap_hit=" << CapHit(info_sets, config.max_info_sets)
            << "\n";
  std::cout << "player_a_ev=" << solver.get_expected_value(0) << "\n";
  std::cout << "seconds=" << seconds << "\n";
  std::cout << "public_states=" << public_states << "\n";
  std::cout << "max_public_states=" << config.max_public_states << "\n";
  std::cout << "public_state_cap_hit="
            << CapHit(public_states, config.max_public_states) << "\n";
  std::cout << "action_entry_touches=" << touches << "\n";
  if (seconds > 0.0) {
    std::cout << "action_entry_touches_per_second="
              << static_cast<double>(touches) / seconds << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();

  poker::PokerConfig config = poker::DefaultPokerConfig();
  int iterations = 100;
  bool saw_global_bet_size = false;
  bool saw_max_info_sets = false;
  bool saw_max_public_states = false;
  int64_t memory_limit_mb = kDefaultMemoryLimitMb;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      std::string value;
      if (arg == "--help") {
        PrintUsage(argv[0]);
        return 0;
      } else if (arg == "--log") {
        absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
        absl::SetGlobalVLogLevel(1);
      } else if (ConsumePrefix(arg, "--config=", &value)) {
        poker::LoadPokerConfig(value, &config);
      } else if (ConsumePrefix(arg, "--iterations=", &value)) {
        iterations = ParseInt(value, "--iterations");
      } else if (ConsumePrefix(arg, "--starting-stack=", &value)) {
        config.set_starting_stack_size(ParseInt(value, "--starting-stack"));
      } else if (ConsumePrefix(arg, "--small-blind=", &value)) {
        config.set_small_blind(ParseInt(value, "--small-blind"));
      } else if (ConsumePrefix(arg, "--big-blind=", &value)) {
        config.set_big_blind(ParseInt(value, "--big-blind"));
      } else if (ConsumePrefix(arg, "--max-depth=", &value)) {
        config.set_max_depth(ParseInt(value, "--max-depth"));
      } else if (ConsumePrefix(arg, "--chance-samples=", &value)) {
        config.set_chance_samples(ParseInt(value, "--chance-samples"));
      } else if (ConsumePrefix(arg, "--bet-size=", &value)) {
        if (!saw_global_bet_size) {
          config.clear_bet_sizes();
          saw_global_bet_size = true;
        }
        config.add_bet_sizes(ParseDouble(value, "--bet-size"));
      } else if (ConsumePrefix(arg, "--preflop-bet-size=", &value)) {
        poker::AddBetSize(&config, poker::Street::PREFLOP,
                          ParseDouble(value, "--preflop-bet-size"));
      } else if (ConsumePrefix(arg, "--flop-bet-size=", &value)) {
        poker::AddBetSize(&config, poker::Street::FLOP,
                          ParseDouble(value, "--flop-bet-size"));
      } else if (ConsumePrefix(arg, "--turn-bet-size=", &value)) {
        poker::AddBetSize(&config, poker::Street::TURN,
                          ParseDouble(value, "--turn-bet-size"));
      } else if (ConsumePrefix(arg, "--river-bet-size=", &value)) {
        poker::AddBetSize(&config, poker::Street::RIVER,
                          ParseDouble(value, "--river-bet-size"));
      } else if (ConsumePrefix(arg, "--max-info-sets=", &value)) {
        config.set_max_info_sets(ParseInt(value, "--max-info-sets"));
        saw_max_info_sets = true;
      } else if (ConsumePrefix(arg, "--max-public-states=", &value)) {
        config.set_max_public_states(ParseInt(value, "--max-public-states"));
        saw_max_public_states = true;
      } else if (ConsumePrefix(arg, "--max-memory-mb=", &value)) {
        memory_limit_mb = ParseInt64(value, "--max-memory-mb");
        if (memory_limit_mb < 0) {
          throw std::invalid_argument("--max-memory-mb must be non-negative");
        }
      } else if (ConsumePrefix(arg, "--threads=", &value)) {
        config.set_num_training_threads(ParseInt(value, "--threads"));
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    // Apply sensible memory caps when the user did not explicitly set them.
    if (!saw_max_info_sets) {
      config.set_max_info_sets(kDefaultMaxInfoSets);
    }
    if (!saw_max_public_states) {
      config.set_max_public_states(kDefaultMaxPublicStates);
    }

    poker::HandRange a_range;
    poker::HandRange b_range;
    a_range.set_uniform_range();
    b_range.set_uniform_range();

    const poker::SolverConfig native_config =
        poker::SolverConfigFromProto(config);
    SetMemoryLimit(memory_limit_mb);
    poker::CFRSolver solver(native_config);
    auto start = std::chrono::steady_clock::now();
    solver.run(iterations, a_range, b_range);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    PrintRunSummary(solver, native_config, elapsed.count());
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
