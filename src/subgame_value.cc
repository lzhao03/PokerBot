#include "src/subgame_value.h"

#include "src/cfr_solver.h"
#include "src/game_tree.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace poker {
namespace {

template <typename Message>
void AppendSerialized(std::string& key, const Message& message) {
  std::string serialized;
  message.SerializeToString(&serialized);
  key.append(std::to_string(serialized.size()));
  key.append(":");
  key.append(serialized);
}

std::string CacheKey(const ContinuationContext& context) {
  std::string key;
  AppendSerialized(key, context.state);
  AppendSerialized(key, context.player_a_hand);
  AppendSerialized(key, context.player_b_hand);
  return key;
}

}  // namespace

ExactHandNestedCFRContinuationValueProvider::
    ExactHandNestedCFRContinuationValueProvider(
    PokerConfig config,
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

  PokerConfig subgame_config = config_;
  subgame_config.set_max_depth(0);
  CFRSolver subgame_solver(subgame_config, context.state);
  subgame_solver.run(
      iterations_, context.player_a_hand, context.player_b_hand);
  return subgame_solver.evaluate_strategy(
      context.player_a_hand, context.player_b_hand);
}

}  // namespace poker
