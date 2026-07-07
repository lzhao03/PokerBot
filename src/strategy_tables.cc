#include "src/strategy_tables.h"

#include <algorithm>
#include <functional>

namespace poker {
namespace {

void HashCombine(size_t& seed, int value) {
  seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

void HashCombine(size_t& seed, uint64_t value) {
  seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
}

template <size_t N>
void HashArray(size_t& seed, const std::array<int, N>& values) {
  for (int value : values) {
    HashCombine(seed, value);
  }
}

}  // namespace

bool FrozenStrategyTables::BettingHistoryKey::operator==(
    const BettingHistoryKey& other) const {
  const int inline_history_size =
      std::min(history_size, kInlineHistoryValues);
  return street == other.street && pot == other.pot &&
         stack_a == other.stack_a && stack_b == other.stack_b &&
         all_in == other.all_in && folded_player == other.folded_player &&
         player_to_act == other.player_to_act &&
         player_contribution_size == other.player_contribution_size &&
         player_contributions == other.player_contributions &&
         history_size == other.history_size &&
         std::equal(history_values.begin(),
                    history_values.begin() + inline_history_size,
                    other.history_values.begin()) &&
         history_overflow == other.history_overflow;
}

size_t FrozenStrategyTables::BettingHistoryKeyHash::operator()(
    const BettingHistoryKey& key) const {
  size_t seed = 0;
  HashCombine(seed, key.street);
  HashCombine(seed, key.pot);
  HashCombine(seed, key.stack_a);
  HashCombine(seed, key.stack_b);
  HashCombine(seed, key.all_in);
  HashCombine(seed, key.folded_player);
  HashCombine(seed, key.player_to_act);
  HashCombine(seed, key.player_contribution_size);
  HashArray(seed, key.player_contributions);
  HashCombine(seed, key.history_size);
  const int inline_history_size =
      std::min(key.history_size, BettingHistoryKey::kInlineHistoryValues);
  for (int i = 0; i < inline_history_size; ++i) {
    HashCombine(seed, key.history_values[i]);
  }
  for (int value : key.history_overflow) {
    HashCombine(seed, value);
  }
  return seed;
}

bool FrozenStrategyTables::PublicStateKey::operator==(
    const PublicStateKey& other) const {
  return betting_history_id == other.betting_history_id &&
         public_bucket == other.public_bucket;
}

size_t FrozenStrategyTables::PublicStateKeyHash::operator()(
    const PublicStateKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.betting_history_id));
  HashCombine(seed, key.public_bucket);
  return seed;
}

bool FrozenStrategyTables::ChanceTransitionKey::operator==(
    const ChanceTransitionKey& other) const {
  return parent_public_state_id == other.parent_public_state_id &&
         outcome_id == other.outcome_id;
}

size_t FrozenStrategyTables::ChanceTransitionKeyHash::operator()(
    const ChanceTransitionKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.parent_public_state_id));
  HashCombine(seed, key.outcome_id);
  return seed;
}

}  // namespace poker
