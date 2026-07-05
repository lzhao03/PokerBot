#include "src/cfr_solver.h"
#include "src/cfr_solver_proto_adapter.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "src/hand_range.h"
#include "src/poker_config.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

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

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --config=PATH                  binary PokerConfig protobuf\n"
      << "  --iterations=N                 CFR iterations, default 100\n"
      << "  --exploitability-samples=N      estimate exploitability after training\n"
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
      << "  --max-info-sets=N               cap info set allocations (0 = unlimited)\n"
      << "  --log                           show INFO logs and VLOG(1) progress\n";
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();

  poker::PokerConfig config = poker::DefaultPokerConfig();
  int iterations = 100;
  int exploitability_samples = 0;
  bool saw_global_bet_size = false;

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
      } else if (ConsumePrefix(arg, "--exploitability-samples=", &value)) {
        exploitability_samples =
            ParseInt(value, "--exploitability-samples");
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
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    poker::HandRange player_a_range;
    poker::HandRange player_b_range;
    player_a_range.set_uniform_range();
    player_b_range.set_uniform_range();

    const poker::SolverConfig native_config =
        poker::SolverConfigFromProto(config);
    poker::CFRSolver solver(native_config);
    auto start = std::chrono::steady_clock::now();
    solver.run(iterations, player_a_range, player_b_range);
    auto end = std::chrono::steady_clock::now();

    poker::CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "iterations=" << solver.get_iterations_run() << "\n";
    std::cout << "info_sets=" << strategy.size() << "\n";
    std::cout << "player_a_ev=" << solver.get_expected_value(0) << "\n";
    std::cout << "seconds=" << elapsed.count() << "\n";

    if (exploitability_samples > 0) {
      std::cout << "exploitability="
                << solver.calculate_exploitability(
                       exploitability_samples, player_a_range, player_b_range)
                << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
