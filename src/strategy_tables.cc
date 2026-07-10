#include "src/strategy_tables.h"

#include <functional>

namespace poker {
namespace {

void HashCombine(size_t& seed, uint64_t value) {
  seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
}

}  // namespace

bool StrategyTables::PublicStateKey::operator==(
    const PublicStateKey& other) const {
  return betting_node_id == other.betting_node_id &&
         public_bucket == other.public_bucket;
}

size_t StrategyTables::PublicStateKeyHash::operator()(
    const PublicStateKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.betting_node_id));
  HashCombine(seed, key.public_bucket);
  return seed;
}

bool StrategyTables::ChanceTransitionKey::operator==(
    const ChanceTransitionKey& other) const {
  return parent_public_state_id == other.parent_public_state_id &&
         outcome_id == other.outcome_id;
}

size_t StrategyTables::ChanceTransitionKeyHash::operator()(
    const ChanceTransitionKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.parent_public_state_id));
  HashCombine(seed, key.outcome_id);
  return seed;
}

}  // namespace poker
