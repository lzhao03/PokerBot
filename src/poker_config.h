#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"

#include "src/poker.pb.h"
#include "src/poker_types.h"

namespace poker {

inline PokerConfig DefaultPokerConfig() {
  PokerConfig config;
  config.add_bet_sizes(0.25);
  config.add_bet_sizes(0.5);
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(100);
  config.set_small_blind(1);
  config.set_big_blind(2);
  return config;
}

inline SolverConfig SolverConfigFromProto(const PokerConfig& config) {
  SolverConfig native;
  native.bet_sizes.assign(config.bet_sizes().begin(), config.bet_sizes().end());
  native.starting_stack_size = config.starting_stack_size();
  native.enable_logging = config.enable_logging();
  native.small_blind = config.small_blind();
  native.big_blind = config.big_blind();
  native.chance_samples = config.chance_samples();
  native.preflop_bet_sizes.assign(config.preflop_bet_sizes().begin(),
                                  config.preflop_bet_sizes().end());
  native.flop_bet_sizes.assign(config.flop_bet_sizes().begin(),
                               config.flop_bet_sizes().end());
  native.turn_bet_sizes.assign(config.turn_bet_sizes().begin(),
                               config.turn_bet_sizes().end());
  native.river_bet_sizes.assign(config.river_bet_sizes().begin(),
                                config.river_bet_sizes().end());
  native.regret_only_training = config.regret_only_training();
  native.max_info_sets = static_cast<int>(config.max_info_sets());
  native.num_training_threads = config.num_training_threads();
  return native;
}

inline void LoadPokerConfig(const std::string& path, PokerConfig* config) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open config: " + path);
  }
  config->Clear();
  if (!config->ParseFromIstream(&file)) {
    throw std::runtime_error("Could not parse config: " + path);
  }
}

inline void AddBetSize(PokerConfig* config, Street street, double size) {
  switch (street) {
    case Street::PREFLOP:
      config->add_preflop_bet_sizes(size);
      break;
    case Street::FLOP:
      config->add_flop_bet_sizes(size);
      break;
    case Street::TURN:
      config->add_turn_bet_sizes(size);
      break;
    case Street::RIVER:
      config->add_river_bet_sizes(size);
      break;
    default:
      config->add_bet_sizes(size);
      break;
  }
}

struct CommonOptionState {
  bool saw_global_bet_size = false;
  bool saw_max_info_sets = false;
};

namespace cli_internal {

template <typename Integer>
Integer ParseIntegerOption(std::string_view value, std::string_view option) {
  Integer parsed = 0;
  const absl::string_view text(value.data(), value.size());
  if (!absl::SimpleAtoi(text, &parsed)) {
    throw std::invalid_argument("Invalid integer for " +
                                std::string(option) + ": " +
                                std::string(value));
  }
  return parsed;
}

}  // namespace cli_internal

inline int ParseIntOption(std::string_view value, std::string_view option) {
  return cli_internal::ParseIntegerOption<int>(value, option);
}

inline int64_t ParseInt64Option(std::string_view value,
                                std::string_view option) {
  return cli_internal::ParseIntegerOption<int64_t>(value, option);
}

inline double ParseDoubleOption(std::string_view value,
                                std::string_view option) {
  double parsed = 0.0;
  const absl::string_view text(value.data(), value.size());
  if (!absl::SimpleAtod(text, &parsed)) {
    throw std::invalid_argument("Invalid number for " +
                                std::string(option) + ": " +
                                std::string(value));
  }
  return parsed;
}

inline bool ApplySolverOption(std::string_view argument,
                              PokerConfig& config,
                              CommonOptionState& state) {
  if (argument == "--regret-only") {
    config.set_regret_only_training(true);
    return true;
  }
  if (argument.starts_with("--config=")) {
    LoadPokerConfig(std::string(argument.substr(sizeof("--config=") - 1)),
                    &config);
    return true;
  }

  struct IntOption {
    std::string_view prefix;
    void (PokerConfig::*setter)(int32_t);
  };
  static constexpr IntOption kIntOptions[] = {
      {"--starting-stack=", &PokerConfig::set_starting_stack_size},
      {"--small-blind=", &PokerConfig::set_small_blind},
      {"--big-blind=", &PokerConfig::set_big_blind},
      {"--chance-samples=", &PokerConfig::set_chance_samples},
      {"--threads=", &PokerConfig::set_num_training_threads},
  };
  for (const IntOption& option : kIntOptions) {
    if (!argument.starts_with(option.prefix)) {
      continue;
    }
    const auto name = option.prefix.substr(0, option.prefix.size() - 1);
    const int value = ParseIntOption(argument.substr(option.prefix.size()),
                                     name);
    (config.*option.setter)(value);
    return true;
  }

  if (argument.starts_with("--max-info-sets=")) {
    config.set_max_info_sets(ParseIntOption(
        argument.substr(sizeof("--max-info-sets=") - 1),
        "--max-info-sets"));
    state.saw_max_info_sets = true;
    return true;
  }
  if (argument.starts_with("--bet-size=")) {
    if (!state.saw_global_bet_size) {
      config.clear_bet_sizes();
      state.saw_global_bet_size = true;
    }
    config.add_bet_sizes(ParseDoubleOption(
        argument.substr(sizeof("--bet-size=") - 1), "--bet-size"));
    return true;
  }

  struct StreetOption {
    std::string_view prefix;
    Street street;
  };
  static constexpr StreetOption kStreetOptions[] = {
      {"--preflop-bet-size=", Street::PREFLOP},
      {"--flop-bet-size=", Street::FLOP},
      {"--turn-bet-size=", Street::TURN},
      {"--river-bet-size=", Street::RIVER},
  };
  for (const StreetOption& option : kStreetOptions) {
    if (!argument.starts_with(option.prefix)) {
      continue;
    }
    const auto name = option.prefix.substr(0, option.prefix.size() - 1);
    const double value = ParseDoubleOption(
        argument.substr(option.prefix.size()), name);
    AddBetSize(&config, option.street, value);
    return true;
  }
  return false;
}

inline constexpr std::string_view kCommonSolverOptionUsage =
    "  --config=PATH                  binary PokerConfig protobuf\n"
    "  --starting-stack=N             solver config override\n"
    "  --small-blind=N                solver config override\n"
    "  --big-blind=N                  solver config override\n"
    "  --chance-samples=N             solver config override\n"
    "  --max-info-sets=N              solver config override\n"
    "  --threads=N                    solver config override\n"
    "  --regret-only                  solver config override\n"
    "  --bet-size=X                   replace/append global bet sizes\n"
    "  --preflop-bet-size=X           solver config override\n"
    "  --flop-bet-size=X              solver config override\n"
    "  --turn-bet-size=X              solver config override\n"
    "  --river-bet-size=X             solver config override\n";

}  // namespace poker
