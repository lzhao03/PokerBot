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
  if (frozen || mutable_tables == nullptr) {
    throw std::logic_error("Strategy tables are frozen");
  }
  return *mutable_tables;
}

const StrategyTables& SolverStorage::frozen_ref() const {
  return *frozen_tables;
}

MutableCumulativeArrays& SolverStorage::cumulative_ref() {
  return *cumulative;
}

const MutableCumulativeArrays& SolverStorage::cumulative_ref() const {
  return *cumulative;
}

void SolverStorage::freeze() {
  if (mutable_tables == nullptr) {
    frozen = true;
    return;
  }
  mutable_tables->public_state_ids.clear();
  mutable_tables->public_state_ids.rehash(0);
  mutable_tables->public_chance_child_ids.clear();
  mutable_tables->public_chance_child_ids.rehash(0);
  frozen_tables = mutable_tables;
  mutable_tables.reset();
  frozen = true;
}

void SolverStorage::bind_frozen(
    std::shared_ptr<const StrategyTables> frozen_in,
    std::shared_ptr<MutableCumulativeArrays> cumulative_in) {
  mutable_tables.reset();
  frozen_tables = std::move(frozen_in);
  cumulative = std::move(cumulative_in);
  frozen = true;
}

void ActionBlock::regret_matching(RegretLoadMode mode,
                                  absl::Span<double> out) const {
  if (!valid() || out.size() != action_count_) {
    throw std::logic_error("regret-matching action count mismatch");
  }

  const auto& regrets = store_->cumulative().cumulative_regrets;
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
  if (!valid() || action_index >= action_count_) {
    throw std::logic_error("regret update action index out of range");
  }

  const size_t index = static_cast<size_t>(action_offset_) + action_index;
  auto& regrets = store_->cumulative().cumulative_regrets;
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
  if (!valid() || probs.size() != action_count_) {
    throw std::logic_error("average-strategy probability span size mismatch");
  }

  auto& strategies = store_->cumulative().cumulative_strategies;
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
  if (!valid() || out.size() != action_count_) {
    throw std::logic_error("average-strategy action count mismatch");
  }

  std::fill(out.begin(), out.end(), 0.0);
  double probability_sum = 0.0;
  const auto& cumulative_regrets = store_->cumulative().cumulative_regrets;
  const auto& cumulative_strategies =
      store_->cumulative().cumulative_strategies;

  for (size_t i = 0; i < out.size(); ++i) {
    store_->stats_->record_action_entries();
    const size_t index = static_cast<size_t>(action_offset_) + i;
    out[i] =
        regret_only_training
            ? std::max(0.0,
                       static_cast<double>(AtomicFloatLoad(
                           &cumulative_regrets[index])))
            : static_cast<double>(AtomicFloatLoad(
                  &cumulative_strategies[index]));
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

MutableCumulativeArrays& StrategyStore::cumulative() {
  return storage_.cumulative_ref();
}

const MutableCumulativeArrays& StrategyStore::cumulative() const {
  return storage_.cumulative_ref();
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
    InfoSetAddress address,
    size_t expected_action_count) {
  return block_for_row(find_info_set_row(address), expected_action_count);
}

std::optional<ActionBlock> StrategyStore::get_or_create(
    InfoSetAddress address,
    size_t action_count) {
  return block_for_row(get_or_create_info_set_row(address, action_count),
                       action_count);
}

std::optional<ActionBlock> StrategyStore::find_frozen(
    uint32_t node_id,
    PrivateBucketId bucket,
    size_t expected_action_count) {
  if (node_id >= frozen_tables().frozen_info_set_action_offsets.size()) {
    return std::nullopt;
  }
  if (expected_action_count > std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("infoset action count exceeds uint16_t");
  }

  const uint32_t offset = frozen_info_set_action_offset(node_id, bucket);
  if (offset == StrategyTables::kInvalidActionOffset) {
    return std::nullopt;
  }
  return ActionBlock(this, offset, static_cast<uint16_t>(expected_action_count));
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

void StrategyStore::average_strategy(uint32_t node_id,
                                     PrivateBucketId bucket,
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
  std::optional<ActionBlock> block =
      find({node_id, bucket}, action_count);
  if (!block.has_value()) {
    std::fill(out.begin(), out.end(), uniform_probability);
    return;
  }
  block->average_strategy(regret_only_training, uniform_probability, out);
}

void StrategyStore::regret_matching_for_bucket(uint32_t node_id,
                                               PrivateBucketId bucket,
                                               size_t action_count,
                                               absl::Span<double> out) {
  if (out.size() != action_count) {
    throw std::logic_error("conditioned strategy span size mismatch");
  }
  if (out.empty()) {
    return;
  }

  std::optional<ActionBlock> block =
      find({node_id, bucket}, action_count);
  if (!block.has_value()) {
    FillUniform(out);
    return;
  }
  block->regret_matching(RegretLoadMode::kAtomic, out);
}

bool StrategyStore::prebuild_frozen_info_set_action_offsets() {
  if (storage_.frozen) {
    return true;
  }

  StrategyTables& tables = tables_for_growth();
  tables.frozen_info_set_action_offsets.resize(tables.public_state_rows.size());
  for (auto& offset_row : tables.frozen_info_set_action_offsets) {
    offset_row.fill(StrategyTables::kInvalidActionOffset);
  }

  for (size_t node_id = 0; node_id < tables.public_state_rows.size();
       ++node_id) {
    const PublicStateRow& row = tables.public_state_rows[node_id];
    if (row.betting_node_id >= tables.betting_nodes.size()) {
      return false;
    }
    const auto& node = tables.betting_nodes[row.betting_node_id];
    const PublicInfoSetSlab* slab =
        public_info_set_slab(static_cast<uint32_t>(node_id));
    if (slab == nullptr) {
      continue;
    }
    const int player = node.state.player_to_act;
    if (!IsPlayer(player)) {
      continue;
    }

    auto& offset_row = tables.frozen_info_set_action_offsets[node_id];
    const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
    for (uint32_t bucket = 0; bucket < StrategyTables::kPrivateBucketCount;
         ++bucket) {
      const auto private_bucket = static_cast<PrivateBucketId>(bucket);
      const InfoSetRow* info_row = find_info_set_row(player_slab,
                                                     private_bucket);
      if (info_row == nullptr) {
        continue;
      }
      if (info_row->action_count != node.action_count) {
        return false;
      }
      offset_row[static_cast<size_t>(bucket)] = info_row->action_offset;
    }
  }
  return true;
}

const StrategyStore::PublicInfoSetSlab* StrategyStore::public_info_set_slab(
    uint32_t node_id) const {
  const auto& slabs = frozen_tables().public_info_set_slabs;
  if (node_id >= slabs.size()) {
    return nullptr;
  }
  return slabs[node_id].get();
}

StrategyStore::PublicInfoSetSlab&
StrategyStore::get_or_create_public_info_set_slab(uint32_t node_id) {
  StrategyTables& tables = tables_for_growth();
  if (tables.public_info_set_slabs.size() <= node_id) {
    tables.public_info_set_slabs.resize(static_cast<size_t>(node_id) + 1);
  }
  std::unique_ptr<PublicInfoSetSlab>& slab =
      tables.public_info_set_slabs[node_id];
  if (slab == nullptr) {
    slab = std::make_unique<PublicInfoSetSlab>();
  }
  return *slab;
}

const StrategyStore::InfoSetRow* StrategyStore::find_info_set_row(
    InfoSetAddress address) const {
  if (address.private_bucket >= StrategyTables::kPrivateBucketCount) {
    return nullptr;
  }
  const int player = player_for_public_state(address.public_state_id);
  if (player < 0 || player >= kPlayerCount) {
    return nullptr;
  }
  const PublicInfoSetSlab* slab = public_info_set_slab(address.public_state_id);
  if (slab == nullptr) {
    return nullptr;
  }
  const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
  return find_info_set_row(player_slab, address.private_bucket);
}

const StrategyStore::InfoSetRow* StrategyStore::find_info_set_row(
    const PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  if (private_bucket >= StrategyTables::kPrivateBucketCount) {
    return nullptr;
  }
  const size_t chunk_index =
      private_bucket / StrategyTables::kPrivateBucketChunkSize;
  const size_t chunk_offset =
      private_bucket % StrategyTables::kPrivateBucketChunkSize;
  const std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    return nullptr;
  }
  const int32_t row_id = chunk->rows[chunk_offset];
  if (row_id < 0 ||
      static_cast<size_t>(row_id) >= player_slab.rows.size()) {
    return nullptr;
  }
  return &player_slab.rows[row_id];
}

int32_t& StrategyStore::get_or_create_private_row_slot(
    PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  const size_t chunk_index =
      private_bucket / StrategyTables::kPrivateBucketChunkSize;
  const size_t chunk_offset =
      private_bucket % StrategyTables::kPrivateBucketChunkSize;
  std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    chunk = std::make_unique<PrivateRowChunk>();
  }
  return chunk->rows[chunk_offset];
}

const StrategyStore::InfoSetRow* StrategyStore::get_or_create_info_set_row(
    InfoSetAddress address,
    size_t action_count) {
  if (address.private_bucket >= StrategyTables::kPrivateBucketCount) {
    return nullptr;
  }
  const int player = player_for_public_state(address.public_state_id);
  if (player < 0 || player >= kPlayerCount) {
    return nullptr;
  }

  if (const InfoSetRow* row = find_info_set_row(address)) {
    return row;
  }

  if (storage_.frozen ||
      (config_.max_info_sets > 0 &&
       static_cast<int>(frozen_tables().info_set_count) >=
           config_.max_info_sets)) {
    return nullptr;
  }

  InfoSetRow row = append_info_set_actions(action_count);
  PublicInfoSetSlab& slab =
      get_or_create_public_info_set_slab(address.public_state_id);
  PublicInfoSetSlabPlayer& player_slab = slab.players[player];
  int32_t& row_id =
      get_or_create_private_row_slot(player_slab, address.private_bucket);
  row_id = static_cast<int32_t>(player_slab.rows.size());
  player_slab.rows.push_back(row);
  ++tables_for_growth().info_set_count;
  return &player_slab.rows.back();
}

StrategyStore::InfoSetRow StrategyStore::append_info_set_actions(
    size_t action_count) {
  const size_t padding =
      (kCumulativeActionBlockAlignment -
       cumulative().cumulative_regrets.size() %
           kCumulativeActionBlockAlignment) %
      kCumulativeActionBlockAlignment;
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(
      cumulative().cumulative_regrets.size() + padding);
  row.action_count = static_cast<uint16_t>(action_count);
  const size_t required_capacity =
      cumulative().cumulative_regrets.size() + padding + action_count;
  if (required_capacity > cumulative().cumulative_regrets.capacity()) {
    const size_t current_capacity = cumulative().cumulative_regrets.capacity();
    const size_t grown_capacity = current_capacity == 0 ? 4096
                                                        : current_capacity * 2;
    const size_t new_capacity = std::max(required_capacity, grown_capacity);
    cumulative().cumulative_regrets.reserve(new_capacity);
    cumulative().cumulative_strategies.reserve(new_capacity);
  }
  for (size_t i = 0; i < padding; ++i) {
    cumulative().cumulative_regrets.push_back(0.0f);
    cumulative().cumulative_strategies.push_back(0.0f);
  }
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    cumulative().cumulative_regrets.push_back(0.0f);
    cumulative().cumulative_strategies.push_back(0.0f);
    stats_->record_action_entries();
  }
  return row;
}

uint32_t StrategyStore::frozen_info_set_action_offset(
    uint32_t public_state_id,
    PrivateBucketId private_bucket) const {
  return frozen_tables().frozen_info_set_action_offsets[public_state_id]
                                                       [private_bucket];
}

int StrategyStore::player_for_public_state(uint32_t public_state_id) const {
  const auto& tables = frozen_tables();
  if (public_state_id >= tables.public_state_rows.size()) {
    return -1;
  }
  const auto& row = tables.public_state_rows[public_state_id];
  if (row.betting_node_id >= tables.betting_nodes.size()) {
    return -1;
  }
  return tables.betting_nodes[row.betting_node_id].state.player_to_act;
}

}  // namespace poker
