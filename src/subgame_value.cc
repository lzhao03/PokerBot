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
void AppendSerialized(std::string* key, const Message& message) {
  std::string serialized;
  message.SerializeToString(&serialized);
  key->append(std::to_string(serialized.size()));
  key->append(":");
  key->append(serialized);
}

std::string CacheKey(const BoardState& state,
                     const Hand& player_a_hand,
                     const Hand& player_b_hand) {
  std::string key;
  AppendSerialized(&key, state);
  AppendSerialized(&key, player_a_hand);
  AppendSerialized(&key, player_b_hand);
  return key;
}

}  // namespace

NestedCFRContinuationValueProvider::NestedCFRContinuationValueProvider(
    PokerConfig config,
    int iterations)
    : config_(std::move(config)), iterations_(iterations) {
  if (iterations_ <= 0) {
    throw std::invalid_argument("Subgame CFR iterations must be positive");
  }
}

double NestedCFRContinuationValueProvider::value(
    GameTree* game_tree,
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand) const {
  if (game_tree == nullptr) {
    throw std::invalid_argument("Game tree cannot be null");
  }

  std::string key = CacheKey(state, player_a_hand, player_b_hand);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = values_.find(key);
    if (it != values_.end()) {
      ++hits_;
      return it->second;
    }
  }

  double computed =
      compute_value(game_tree, state, player_a_hand, player_b_hand);

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

NestedCFRContinuationValueProvider::Stats
NestedCFRContinuationValueProvider::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {hits_, misses_, static_cast<int64_t>(values_.size())};
}

double NestedCFRContinuationValueProvider::compute_value(
    GameTree* game_tree,
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand) const {
  if (game_tree->is_terminal(state)) {
    return game_tree->get_utility(state, player_a_hand, player_b_hand);
  }

  PokerConfig subgame_config = config_;
  subgame_config.set_max_depth(0);
  CFRSolver subgame_solver(subgame_config, state);
  subgame_solver.run(iterations_, player_a_hand, player_b_hand);
  return subgame_solver.evaluate_strategy(player_a_hand, player_b_hand);
}

}  // namespace poker
