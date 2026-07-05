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
