#include "src/subgame_value.h"

#include "src/cfr_solver.h"
#include "src/game_tree.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace poker {
namespace {

std::string CacheKey(const ContinuationContext& context) {
  std::ostringstream key;
  key << static_cast<int>(context.state.street) << ':' << context.state.pot
      << ':' << context.state.stack[0] << ':' << context.state.stack[1]
      << ':' << context.state.player_contribution[0] << ':'
      << context.state.player_contribution[1] << ':'
      << context.state.folded_player << ':' << context.state.player_to_act
      << ':' << context.state.all_in << ':' << context.player_a_hand << ':'
      << context.player_b_hand << ':';
  for (CardId card : context.state.board_cards) {
    key << static_cast<int>(card) << ',';
  }
  key << ':';
  for (const GameAction& action : context.state.history) {
    key << action.player << ',' << static_cast<int>(action.kind) << ','
        << action.amount << ';';
  }
  return key.str();
}

}  // namespace

ExactHandNestedCFRContinuationValueProvider::
    ExactHandNestedCFRContinuationValueProvider(
    SolverConfig config,
    int iterations)
    : config_(std::move(config)), iterations_(iterations) {
  if (iterations_ <= 0) {
    throw std::invalid_argument("Subgame CFR iterations must be positive");
  }
}

double ExactHandNestedCFRContinuationValueProvider::value(
    GameTree& game_tree,
    const ContinuationContext& context) const {
  std::string key = CacheKey(context);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = values_.find(key);
    if (it != values_.end()) {
      ++hits_;
      return it->second;
    }
  }

  double computed = compute_value(game_tree, context);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto inserted = values_.emplace(std::move(key), computed);
    if (inserted.second) {
      ++misses_;
      return computed;
    }
    ++hits_;
    return inserted.first->second;
  }
}

ExactHandNestedCFRContinuationValueProvider::Stats
ExactHandNestedCFRContinuationValueProvider::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {hits_, misses_, static_cast<int64_t>(values_.size())};
}

double ExactHandNestedCFRContinuationValueProvider::compute_value(
    GameTree& game_tree,
    const ContinuationContext& context) const {
  if (game_tree.is_terminal(context.state)) {
    return game_tree.get_utility(
        context.state, context.player_a_hand, context.player_b_hand);
  }

  SolverConfig subgame_config = config_;
  subgame_config.max_depth = 0;
  CFRSolver subgame_solver(subgame_config, context.state);
  subgame_solver.run(
      iterations_, context.player_a_hand, context.player_b_hand);
  return subgame_solver.evaluate_strategy(
      context.player_a_hand, context.player_b_hand);
}

}  // namespace poker
