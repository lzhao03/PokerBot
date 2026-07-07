#include "src/terminal_utility_cache.h"

namespace poker {
namespace {

void HashCombine(size_t& seed, int value) {
  seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

void HashCombine(size_t& seed, uint64_t value) {
  seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
          (seed << 6) + (seed >> 2);
}

}  // namespace

TerminalUtilityCache::Stats TerminalUtilityCache::stats() const {
  Stats stats;
  for (const Shard& shard : shards_) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    stats.hits += shard.hits;
    stats.misses += shard.misses;
    stats.entries += static_cast<int64_t>(shard.values.size());
  }
  return stats;
}

double TerminalUtilityCache::get_or_compute(
    const CompactPublicState& state,
    ComboId player_a_hand,
    ComboId player_b_hand,
    const std::function<double()>& compute) {
  return get_or_compute_key(
      key_for(state, player_a_hand, player_b_hand), compute);
}

double TerminalUtilityCache::get_or_compute_key(
    Key key,
    const std::function<double()>& compute) {
  Shard& shard = shards_[key.hash & (kShardCount - 1)];

  {
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto it = shard.values.find(key);
    if (it != shard.values.end()) {
      ++shard.hits;
      return it->second;
    }
  }

  const double value = compute();

  {
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto inserted = shard.values.emplace(key, value);
    if (inserted.second) {
      ++shard.misses;
      return value;
    }
    ++shard.hits;
    return inserted.first->second;
  }
}

bool TerminalUtilityCache::Key::operator==(const Key& other) const {
  return street == other.street && pot == other.pot &&
         player_a_contribution == other.player_a_contribution &&
         player_b_contribution == other.player_b_contribution &&
         board_size == other.board_size &&
         player_a_hand == other.player_a_hand &&
         player_b_hand == other.player_b_hand &&
         board_mask == other.board_mask;
}

size_t TerminalUtilityCache::KeyHash::operator()(const Key& key) const {
  return key.hash;
}

size_t TerminalUtilityCache::compute_hash(const Key& key) {
  size_t seed = 0;
  HashCombine(seed, key.street);
  HashCombine(seed, key.pot);
  HashCombine(seed, key.player_a_contribution);
  HashCombine(seed, key.player_b_contribution);
  HashCombine(seed, key.board_size);
  HashCombine(seed, static_cast<int>(key.player_a_hand));
  HashCombine(seed, static_cast<int>(key.player_b_hand));
  HashCombine(seed, static_cast<uint64_t>(key.board_mask));
  return seed;
}

TerminalUtilityCache::Key TerminalUtilityCache::key_for(
    const CompactPublicState& state,
    ComboId player_a_hand,
    ComboId player_b_hand) {
  Key key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.player_a_contribution = state.player_contribution[0];
  key.player_b_contribution = state.player_contribution[1];
  key.board_size = state.board_count;
  key.player_a_hand = player_a_hand;
  key.player_b_hand = player_b_hand;
  key.board_mask = state.board_mask;
  key.hash = compute_hash(key);
  return key;
}

}  // namespace poker
