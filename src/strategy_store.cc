#include "src/strategy_store.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/build_flags.h"

namespace poker {
namespace {

void AtomicFloatAdd(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float new_val;
    std::memcpy(&new_val, &old_bits, sizeof(float));
    new_val += delta;
    std::memcpy(&new_bits, &new_val, sizeof(float));
  } while (!__atomic_compare_exchange_n(
      reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
      /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

int64_t AtomicCFRPlusRegretUpdate(float* target, float delta) {
  static_assert(sizeof(float) == sizeof(int32_t), "float must be 32-bit");
  int32_t old_bits, new_bits;
  bool exchanged = false;
  int64_t retries = 0;
  do {
    old_bits = __atomic_load_n(reinterpret_cast<int32_t*>(target),
                               __ATOMIC_RELAXED);
    float old_val;
    std::memcpy(&old_val, &old_bits, sizeof(float));
    const float new_val = std::max(0.0f, old_val + delta);
    std::memcpy(&new_bits, &new_val, sizeof(float));
    exchanged = __atomic_compare_exchange_n(
        reinterpret_cast<int32_t*>(target), &old_bits, new_bits,
        /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    if constexpr (kCasRetryStatsEnabled) {
      if (!exchanged) {
        ++retries;
      }
    }
  } while (!exchanged);
  if constexpr (kCasRetryStatsEnabled) {
    return retries;
  }
  return 0;
}

float AtomicFloatLoad(const float* src) {
  const int32_t bits = __atomic_load_n(
      reinterpret_cast<const int32_t*>(src), __ATOMIC_RELAXED);
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

void FillUniform(absl::Span<double> out) {
  if (out.empty()) {
    return;
  }
  std::fill(out.begin(), out.end(), 1.0 / out.size());
}

}  // namespace

StrategyTables& SolverStorage::mutable_ref() {
  if (is_frozen()) {
    throw std::logic_error("Strategy tables are frozen");
  }
  return *mutable_tables;
}

const StrategyTables& SolverStorage::frozen_ref() const {
  return *frozen_tables;
}

CfrState& SolverStorage::cfr_state_ref() {
  return *cfr_state;
}

const CfrState& SolverStorage::cfr_state_ref() const {
  return *cfr_state;
}

void SolverStorage::freeze() {
  if (is_frozen()) {
    return;
  }
  mutable_tables->node_ids.clear();
  mutable_tables->node_ids.rehash(0);
  mutable_tables->public_chance_child_ids.clear();
  mutable_tables->public_chance_child_ids.rehash(0);
  mutable_tables->growing_info_sets.clear();
  frozen_tables = mutable_tables;
  mutable_tables.reset();
}

void SolverStorage::bind_frozen(
    std::shared_ptr<const StrategyTables> frozen_in,
    std::shared_ptr<CfrState> cfr_state_in) {
  mutable_tables.reset();
  frozen_tables = std::move(frozen_in);
  cfr_state = std::move(cfr_state_in);
}

void ActionBlock::regret_matching(RegretLoadMode mode,
                                  absl::Span<double> out) const {
  if (out.size() != action_count_) {
    throw std::logic_error("regret-matching action count mismatch");
  }

  const auto& regrets = store_->cfr_state().regret_sum;
  double positive_sum = 0.0;
  for (size_t action_index = 0; action_index < out.size(); ++action_index) {
    store_->stats_->record_action_entries();
    const size_t index = static_cast<size_t>(action_offset_) + action_index;
    const float raw = mode == RegretLoadMode::kAtomic
                          ? AtomicFloatLoad(&regrets[index])
                          : regrets[index];
    const double regret = std::max(0.0, static_cast<double>(raw));
    out[action_index] = regret;
    positive_sum += regret;
  }

  if (positive_sum <= 0.0) {
    FillUniform(out);
    return;
  }
  for (double& probability : out) {
    probability /= positive_sum;
  }
}

void ActionBlock::add_cfr_plus_regret(
    size_t action_index,
    float delta,
    RegretUpdateOptions options) const {
  if (action_index >= action_count_) {
    throw std::logic_error("regret update action index out of range");
  }

  const size_t index = static_cast<size_t>(action_offset_) + action_index;
  auto& regrets = store_->cfr_state().regret_sum;
  if (index >= regrets.size()) {
    throw std::logic_error("regret update table index out of range");
  }

  store_->stats_->record_action_entries(2);
  float* regret_entry = &regrets[index];
  if (options.mode == RegretUpdateMode::kAtomic) {
    const int64_t retries = AtomicCFRPlusRegretUpdate(regret_entry, delta);
    if (options.record_atomic_retry_stats) {
      store_->stats_->record_atomic_retries(retries);
    }
    return;
  }
  *regret_entry = std::max(0.0f, *regret_entry + delta);
}

void ActionBlock::add_average_strategy(absl::Span<const double> probs,
                                       double reach_weight,
                                       RegretUpdateMode mode) const {
  if (probs.size() != action_count_) {
    throw std::logic_error("average-strategy probability span size mismatch");
  }

  auto& strategies = store_->cfr_state().strategy_sum;
  for (size_t action_index = 0; action_index < probs.size(); ++action_index) {
    const size_t index = static_cast<size_t>(action_offset_) + action_index;
    if (index >= strategies.size()) {
      throw std::logic_error("average-strategy table index out of range");
    }

    store_->stats_->record_action_entries(2);
    const float delta = static_cast<float>(reach_weight * probs[action_index]);
    if (mode == RegretUpdateMode::kAtomic) {
      AtomicFloatAdd(&strategies[index], delta);
    } else {
      strategies[index] += delta;
    }
  }
}

void ActionBlock::average_strategy(bool regret_only_training,
                                   double fallback_probability,
                                   absl::Span<double> out) const {
  if (out.size() != action_count_) {
    throw std::logic_error("average-strategy action count mismatch");
  }

  std::fill(out.begin(), out.end(), 0.0);
  double probability_sum = 0.0;
  const auto& regret_sum = store_->cfr_state().regret_sum;
  const auto& strategy_sum = store_->cfr_state().strategy_sum;

  for (size_t i = 0; i < out.size(); ++i) {
    store_->stats_->record_action_entries();
    const size_t index = static_cast<size_t>(action_offset_) + i;
    out[i] =
        regret_only_training
            ? std::max(0.0,
                       static_cast<double>(AtomicFloatLoad(
                           &regret_sum[index])))
            : static_cast<double>(AtomicFloatLoad(
                  &strategy_sum[index]));
    probability_sum += out[i];
  }

  if (probability_sum <= 0.0) {
    std::fill(out.begin(), out.end(), fallback_probability);
    return;
  }
  for (double& probability : out) {
    probability /= probability_sum;
  }
}

StrategyStore::StrategyStore(
    const SolverConfig& config,
    SolverStorage& storage,
    TraversalStats* stats)
    : config_(config),
      storage_(storage),
      stats_(stats) {}

StrategyTables& StrategyStore::mutable_tables() {
  return tables_for_growth();
}

const StrategyTables& StrategyStore::frozen_tables() const {
  return storage_.frozen_ref();
}

StrategyTables& StrategyStore::tables_for_growth() {
  return storage_.mutable_ref();
}

CfrState& StrategyStore::cfr_state() {
  return storage_.cfr_state_ref();
}

const CfrState& StrategyStore::cfr_state() const {
  return storage_.cfr_state_ref();
}

ActionBlock StrategyStore::block_for_row(const InfoSetRow& row) {
  return ActionBlock(this, row.action_offset, row.action_count);
}

std::optional<ActionBlock> StrategyStore::block_for_row(
    const InfoSetRow* row,
    size_t expected_action_count) {
  if (row == nullptr ||
      static_cast<size_t>(row->action_count) != expected_action_count) {
    return std::nullopt;
  }
  return block_for_row(*row);
}

std::optional<ActionBlock> StrategyStore::find(
    InfoSetKey key,
    size_t expected_action_count) {
  if (!action_count_matches(key.node_id, expected_action_count)) {
    return std::nullopt;
  }
  if (storage_.is_frozen()) {
    return find_frozen(key, expected_action_count);
  }
  return block_for_row(find_growing_row(key), expected_action_count);
}

std::optional<ActionBlock> StrategyStore::get_or_create(
    InfoSetKey key,
    size_t action_count) {
  if (!action_count_matches(key.node_id, action_count)) {
    return std::nullopt;
  }
  if (storage_.is_frozen()) {
    return find_frozen(key, action_count);
  }
  return block_for_row(get_or_create_info_set_row(key, action_count),
                       action_count);
}

std::optional<ActionBlock> StrategyStore::find_frozen(
    InfoSetKey key,
    size_t expected_action_count) {
  if (!action_count_matches(key.node_id, expected_action_count)) {
    return std::nullopt;
  }
  const FrozenInfoSetEntry* entry = find_frozen_entry(key);
  if (entry == nullptr) {
    return std::nullopt;
  }
  return ActionBlock(this, entry->action_offset,
                     static_cast<uint16_t>(expected_action_count));
}

void StrategyStore::regret_matching_or_uniform(
    std::optional<ActionBlock> block,
    size_t legal_action_count,
    RegretLoadMode load_mode,
    absl::Span<double> out) {
  if (out.size() != legal_action_count) {
    throw std::logic_error("regret-matching probability span size mismatch");
  }
  if (legal_action_count == 0) {
    return;
  }
  if (!block.has_value() || block->action_count() != legal_action_count) {
    FillUniform(out);
    return;
  }
  block->regret_matching(load_mode, out);
}

void StrategyStore::average_strategy(InfoSetKey key,
                                     size_t action_count,
                                     bool regret_only_training,
                                     absl::Span<double> out) {
  if (out.size() != action_count) {
    throw std::logic_error("average-strategy probability span size mismatch");
  }
  if (out.empty()) {
    return;
  }

  const double uniform_probability = 1.0 / out.size();
  std::optional<ActionBlock> block = find(key, action_count);
  if (!block.has_value()) {
    std::fill(out.begin(), out.end(), uniform_probability);
    return;
  }
  block->average_strategy(regret_only_training, uniform_probability, out);
}

void StrategyStore::regret_matching_for_observation(
    InfoSetKey key,
    size_t action_count,
    absl::Span<double> out) {
  if (out.size() != action_count) {
    throw std::logic_error("conditioned strategy span size mismatch");
  }
  if (out.empty()) {
    return;
  }

  std::optional<ActionBlock> block = find(key, action_count);
  if (!block.has_value()) {
    FillUniform(out);
    return;
  }
  block->regret_matching(RegretLoadMode::kAtomic, out);
}

bool StrategyStore::build_frozen_info_set_index() {
  if (storage_.is_frozen()) {
    return true;
  }

  StrategyTables& tables = tables_for_growth();
  tables.frozen_info_set_entries.clear();
  tables.frozen_info_set_entries.reserve(tables.info_set_count);
  tables.frozen_info_set_ranges.assign(tables.nodes.size(), {});

  for (NodeId node_id = 0; node_id < tables.nodes.size(); ++node_id) {
    const GrowingPublicInfoSets* rows = growing_rows(node_id);
    if (rows == nullptr) {
      continue;
    }
    const Node& row = tables.nodes[node_id];
    if (row.betting_node_id >= tables.betting_nodes.size()) {
      return false;
    }
    const size_t action_count =
        tables.betting_nodes[row.betting_node_id].action_count;
    if (!action_count_matches(node_id, action_count)) {
      return false;
    }

    FrozenPublicInfoSetRange& range =
        tables.frozen_info_set_ranges[node_id];
    range.begin = static_cast<uint32_t>(tables.frozen_info_set_entries.size());
    for (const auto& [private_observation, row] : rows->rows) {
      if (row.action_count != action_count) {
        return false;
      }
      tables.frozen_info_set_entries.push_back(
          {private_observation, row.action_offset});
    }
    auto first = tables.frozen_info_set_entries.begin() + range.begin;
    std::sort(first, tables.frozen_info_set_entries.end(),
              [](const FrozenInfoSetEntry& a,
                 const FrozenInfoSetEntry& b) {
                return a.private_observation < b.private_observation;
              });
    range.count = static_cast<uint32_t>(rows->rows.size());
  }
  return true;
}

const StrategyStore::GrowingPublicInfoSets* StrategyStore::growing_rows(
    NodeId node_id) const {
  const auto& rows = frozen_tables().growing_info_sets;
  if (node_id >= rows.size()) {
    return nullptr;
  }
  return rows[node_id].get();
}

StrategyStore::GrowingPublicInfoSets&
StrategyStore::get_or_create_growing_rows(NodeId node_id) {
  StrategyTables& tables = tables_for_growth();
  if (tables.growing_info_sets.size() <= node_id) {
    tables.growing_info_sets.resize(static_cast<size_t>(node_id) + 1);
  }
  std::unique_ptr<GrowingPublicInfoSets>& rows =
      tables.growing_info_sets[node_id];
  if (rows == nullptr) {
    rows = std::make_unique<GrowingPublicInfoSets>();
  }
  return *rows;
}

const StrategyStore::InfoSetRow* StrategyStore::find_growing_row(
    InfoSetKey key) const {
  const GrowingPublicInfoSets* rows = growing_rows(key.node_id);
  if (rows == nullptr) {
    return nullptr;
  }
  const auto row = rows->rows.find(key.private_observation);
  return row == rows->rows.end() ? nullptr : &row->second;
}

const StrategyStore::FrozenInfoSetEntry* StrategyStore::find_frozen_entry(
    InfoSetKey key) const {
  const auto& tables = frozen_tables();
  if (key.node_id >= tables.frozen_info_set_ranges.size()) {
    return nullptr;
  }
  const FrozenPublicInfoSetRange& range =
      tables.frozen_info_set_ranges[key.node_id];
  const size_t end = static_cast<size_t>(range.begin) + range.count;
  if (end > tables.frozen_info_set_entries.size()) {
    return nullptr;
  }
  const auto first = tables.frozen_info_set_entries.begin() + range.begin;
  const auto last = tables.frozen_info_set_entries.begin() + end;
  const auto entry = std::lower_bound(
      first, last, key.private_observation,
      [](const FrozenInfoSetEntry& candidate,
         PrivateObservationId observation) {
        return candidate.private_observation < observation;
      });
  if (entry == last ||
      entry->private_observation != key.private_observation) {
    return nullptr;
  }
  return &*entry;
}

const StrategyStore::InfoSetRow* StrategyStore::get_or_create_info_set_row(
    InfoSetKey key,
    size_t action_count) {
  if (const InfoSetRow* row = find_growing_row(key)) {
    return row;
  }

  if (storage_.is_frozen() ||
      (config_.max_info_sets > 0 &&
       static_cast<int>(frozen_tables().info_set_count) >=
           config_.max_info_sets)) {
    return nullptr;
  }

  InfoSetRow row = append_info_set_actions(action_count);
  GrowingPublicInfoSets& rows = get_or_create_growing_rows(key.node_id);
  const auto [entry, inserted] =
      rows.rows.try_emplace(key.private_observation, row);
  if (!inserted) {
    return &entry->second;
  }
  ++tables_for_growth().info_set_count;
  return &entry->second;
}

StrategyStore::InfoSetRow StrategyStore::append_info_set_actions(
    size_t action_count) {
  const size_t padding =
      (kCumulativeActionBlockAlignment -
       cfr_state().regret_sum.size() %
           kCumulativeActionBlockAlignment) %
      kCumulativeActionBlockAlignment;
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(
      cfr_state().regret_sum.size() + padding);
  row.action_count = static_cast<uint16_t>(action_count);
  const size_t required_capacity =
      cfr_state().regret_sum.size() + padding + action_count;
  if (required_capacity > cfr_state().regret_sum.capacity()) {
    const size_t current_capacity = cfr_state().regret_sum.capacity();
    const size_t grown_capacity = current_capacity == 0 ? 4096
                                                        : current_capacity * 2;
    const size_t new_capacity = std::max(required_capacity, grown_capacity);
    cfr_state().regret_sum.reserve(new_capacity);
    cfr_state().strategy_sum.reserve(new_capacity);
  }
  for (size_t i = 0; i < padding; ++i) {
    cfr_state().regret_sum.push_back(0.0f);
    cfr_state().strategy_sum.push_back(0.0f);
  }
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    cfr_state().regret_sum.push_back(0.0f);
    cfr_state().strategy_sum.push_back(0.0f);
    stats_->record_action_entries();
  }
  return row;
}

bool StrategyStore::action_count_matches(NodeId node_id,
                                         size_t action_count) const {
  if (action_count > std::numeric_limits<uint16_t>::max()) {
    return false;
  }
  const auto& tables = frozen_tables();
  if (node_id >= tables.nodes.size()) {
    return false;
  }
  const auto& row = tables.nodes[node_id];
  if (row.betting_node_id >= tables.betting_nodes.size()) {
    return false;
  }
  const auto& node = tables.betting_nodes[row.betting_node_id];
  return node.kind == StrategyTables::NodeKind::kDecision &&
         IsPlayer(node.state.player_to_act) &&
         node.action_count == action_count;
}

}  // namespace poker
