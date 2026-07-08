#include "src/poker_config.h"

#include <fstream>
#include <stdexcept>

namespace poker {

PokerConfig DefaultPokerConfig() {
  PokerConfig config;
  config.add_bet_sizes(0.25);
  config.add_bet_sizes(0.5);
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(100);
  config.set_small_blind(1);
  config.set_big_blind(2);
  return config;
}

SolverConfig SolverConfigFromProto(const PokerConfig& config) {
  SolverConfig native;
  native.bet_sizes.assign(config.bet_sizes().begin(), config.bet_sizes().end());
  native.starting_stack_size = config.starting_stack_size();
  native.max_depth = config.max_depth();
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
  native.max_public_states = static_cast<int>(config.max_public_states());
  native.num_training_threads = config.num_training_threads();
  native.warmup_iterations = config.warmup_iterations();
  return native;
}

void LoadPokerConfig(const std::string& path, PokerConfig* config) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open config: " + path);
  }
  config->Clear();
  if (!config->ParseFromIstream(&file)) {
    throw std::runtime_error("Could not parse config: " + path);
  }
}

void AddBetSize(PokerConfig* config, Street street, double size) {
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

}  // namespace poker
