#include "src/strategy_tables.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>

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

std::optional<uint32_t> StrategyTables::chance_child(
    uint32_t parent_public_state_id,
    PublicBucketId outcome_id) const {
  if (parent_public_state_id >= public_state_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_state_rows[parent_public_state_id];
  const size_t begin = row.chance_child_offset;
  const size_t end = begin + row.chance_child_count;
  if (begin > chance_child_entries.size() || end > chance_child_entries.size()) {
    return std::nullopt;
  }

  const auto first =
      chance_child_entries.begin() + static_cast<std::ptrdiff_t>(begin);
  const auto last =
      chance_child_entries.begin() + static_cast<std::ptrdiff_t>(end);
  const auto iter = std::lower_bound(
      first, last, outcome_id,
      [](const ChanceChildEntry& entry, PublicBucketId target) {
        return entry.outcome_id < target;
      });
  if (iter == last || iter->outcome_id != outcome_id) {
    return std::nullopt;
  }
  return iter->public_state_id;
}

std::optional<uint32_t> StrategyTables::action_child(
    uint32_t parent_public_state_id,
    int action_index) const {
  if (parent_public_state_id >= public_state_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_state_rows[parent_public_state_id];
  if (row.betting_node_id >= betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node = betting_nodes[row.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    return std::nullopt;
  }

  const size_t child_index = static_cast<size_t>(row.action_child_offset) +
                             static_cast<size_t>(action_index);
  if (child_index >= action_child_ids.size()) {
    return std::nullopt;
  }
  return action_child_ids[child_index];
}

}  // namespace poker
