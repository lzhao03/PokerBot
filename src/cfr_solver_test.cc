#include "src/cfr_solver.h"

#include <stdexcept>

int main() {
  poker::PokerConfig config;
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(10);
  config.set_max_depth(4);

  poker::CFRSolver solver(config);
  solver.run(1);

  if (solver.get_equilibrium_strategy().get_info_sets().empty()) {
    throw std::runtime_error("CFR did not visit any information sets");
  }

  return 0;
}
