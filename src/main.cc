#include "src/cfr_solver.h"
#include "src/poker.pb.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

poker::PokerConfig DefaultConfig() {
  poker::PokerConfig config;
  config.add_bet_sizes(0.25);
  config.add_bet_sizes(0.5);
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(100);
  config.set_small_blind(1);
  config.set_big_blind(2);
  return config;
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

double ParseDouble(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::invalid_argument("Invalid number for " + flag + ": " + value);
  }
  return parsed;
}

void LoadConfig(const std::string& path, poker::PokerConfig* config) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open config: " + path);
  }
  config->Clear();
  if (!config->ParseFromIstream(&file)) {
    throw std::runtime_error("Could not parse config: " + path);
  }
}

void AddBetSize(poker::PokerConfig* config,
                poker::Street street,
                double size) {
  switch (street) {
    case poker::Street::PREFLOP:
      config->add_preflop_bet_sizes(size);
      break;
    case poker::Street::FLOP:
      config->add_flop_bet_sizes(size);
      break;
    case poker::Street::TURN:
      config->add_turn_bet_sizes(size);
      break;
    case poker::Street::RIVER:
      config->add_river_bet_sizes(size);
      break;
    default:
      config->add_bet_sizes(size);
      break;
  }
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --config=PATH                  binary PokerConfig protobuf\n"
      << "  --iterations=N                 CFR iterations, default 100\n"
      << "  --output=PATH                   strategy snapshot, default strategy.pb\n"
      << "  --exploitability-samples=N      estimate exploitability after training\n"
      << "  --starting-stack=N\n"
      << "  --small-blind=N\n"
      << "  --big-blind=N\n"
      << "  --max-depth=N                   0 disables depth cutoff\n"
      << "  --chance-samples=N\n"
      << "  --max-raises-per-street=N\n"
      << "  --bet-size=X                    replaces default global sizes on first use\n"
      << "  --preflop-bet-size=X\n"
      << "  --flop-bet-size=X\n"
      << "  --turn-bet-size=X\n"
      << "  --river-bet-size=X\n"
      << "  --log\n";
}

}  // namespace

int main(int argc, char** argv) {
  poker::PokerConfig config = DefaultConfig();
  int iterations = 100;
  int exploitability_samples = 0;
  std::string output_path = "strategy.pb";
  bool saw_global_bet_size = false;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      std::string value;
      if (arg == "--help") {
        PrintUsage(argv[0]);
        return 0;
      } else if (arg == "--log") {
        config.set_enable_logging(true);
      } else if (ConsumePrefix(arg, "--config=", &value)) {
        LoadConfig(value, &config);
      } else if (ConsumePrefix(arg, "--iterations=", &value)) {
        iterations = ParseInt(value, "--iterations");
      } else if (ConsumePrefix(arg, "--output=", &value)) {
        output_path = value;
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
      } else if (ConsumePrefix(arg, "--max-raises-per-street=", &value)) {
        config.set_max_raises_per_street(
            ParseInt(value, "--max-raises-per-street"));
      } else if (ConsumePrefix(arg, "--bet-size=", &value)) {
        if (!saw_global_bet_size) {
          config.clear_bet_sizes();
          saw_global_bet_size = true;
        }
        config.add_bet_sizes(ParseDouble(value, "--bet-size"));
      } else if (ConsumePrefix(arg, "--preflop-bet-size=", &value)) {
        AddBetSize(&config, poker::Street::PREFLOP,
                   ParseDouble(value, "--preflop-bet-size"));
      } else if (ConsumePrefix(arg, "--flop-bet-size=", &value)) {
        AddBetSize(&config, poker::Street::FLOP,
                   ParseDouble(value, "--flop-bet-size"));
      } else if (ConsumePrefix(arg, "--turn-bet-size=", &value)) {
        AddBetSize(&config, poker::Street::TURN,
                   ParseDouble(value, "--turn-bet-size"));
      } else if (ConsumePrefix(arg, "--river-bet-size=", &value)) {
        AddBetSize(&config, poker::Street::RIVER,
                   ParseDouble(value, "--river-bet-size"));
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    poker::CFRSolver solver(config);
    auto start = std::chrono::steady_clock::now();
    solver.run(iterations);
    auto end = std::chrono::steady_clock::now();

    poker::Strategy strategy = solver.get_equilibrium_strategy();
    solver.save_strategy(output_path);

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "iterations=" << solver.get_iterations_run() << "\n";
    std::cout << "info_sets=" << strategy.get_info_sets().size() << "\n";
    std::cout << "player_a_ev=" << solver.get_expected_value(0) << "\n";
    std::cout << "seconds=" << elapsed.count() << "\n";
    std::cout << "strategy_path=" << output_path << "\n";

    if (exploitability_samples > 0) {
      std::cout << "exploitability="
                << solver.calculate_exploitability(exploitability_samples)
                << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
