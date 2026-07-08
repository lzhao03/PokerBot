#include "src/strategy_store.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace poker {

void ActionBlock::regret_matching(absl::Span<const int> legal_action_ids,
                                  absl::Span<double> out) const {
  if (!valid() || out.size() != legal_action_ids.size()) {
    throw std::logic_error("aligned regret-matching action count mismatch");
  }
  if (out.empty()) {
    return;
  }

  const double fallback_probability = 1.0 / out.size();
  const auto& ids = store_->frozen_tables().action_ids;
  const auto& regrets = store_->cumulative().cumulative_regrets;
  double positive_regret_sum = 0.0;
  std::fill(out.begin(), out.end(), 0.0);
  const size_t matched_action_count =
      std::min(out.size(), static_cast<size_t>(action_count_));
  for (size_t action_index = 0; action_index < matched_action_count;
       ++action_index) {
    const size_t table_index =
        static_cast<size_t>(action_offset_) + action_index;
    if (ids[table_index] != legal_action_ids[action_index]) {
      continue;
    }

    store_->record_action_entry_touches();
    const double positive_regret =
        std::max(
            0.0,
            static_cast<double>(
                strategy_store_internal::AtomicFloatLoad(
                    &regrets[table_index])));
    out[action_index] = positive_regret;
    positive_regret_sum += positive_regret;
  }

  if (positive_regret_sum <= 0.0) {
    std::fill(out.begin(), out.end(), fallback_probability);
    return;
  }
  for (double& probability : out) {
    probability /= positive_regret_sum;
  }
}

void ActionBlock::average_strategy(bool regret_only_training,
                                   absl::Span<const int> legal_action_ids,
                                   double fallback_probability,
                                   absl::Span<double> out) const {
  if (!valid() || out.size() != legal_action_ids.size()) {
    throw std::logic_error("average-strategy action count mismatch");
  }

  std::fill(out.begin(), out.end(), 0.0);
  double probability_sum = 0.0;
  const auto& ids = store_->frozen_tables().action_ids;
  const auto& cumulative_regrets = store_->cumulative().cumulative_regrets;
  const auto& cumulative_strategies =
      store_->cumulative().cumulative_strategies;

  const bool aligned_action_ids =
      legal_action_ids.size() == action_count_ &&
      [&]() {
        for (size_t i = 0; i < legal_action_ids.size(); ++i) {
          if (legal_action_ids[i] != ids[action_offset_ + i]) {
            return false;
          }
        }
        return true;
      }();

  if (aligned_action_ids) {
    for (size_t i = 0; i < legal_action_ids.size(); ++i) {
      store_->record_action_entry_touches();
      const size_t table_index = static_cast<size_t>(action_offset_) + i;
      out[i] =
          regret_only_training
              ? std::max(0.0,
                         static_cast<double>(
                             strategy_store_internal::AtomicFloatLoad(
                                 &cumulative_regrets[table_index])))
              : static_cast<double>(
                    strategy_store_internal::AtomicFloatLoad(
                        &cumulative_strategies[table_index]));
      probability_sum += out[i];
    }
  } else {
    for (size_t legal_action_index = 0;
         legal_action_index < legal_action_ids.size(); ++legal_action_index) {
      const int legal_action_id = legal_action_ids[legal_action_index];
      for (uint16_t action_index = 0; action_index < action_count_;
           ++action_index) {
        store_->record_action_entry_touches();
        const size_t table_index =
            static_cast<size_t>(action_offset_) + action_index;
        if (ids[table_index] == legal_action_id) {
          out[legal_action_index] =
              regret_only_training
                  ? std::max(
                        0.0,
                        static_cast<double>(
                            strategy_store_internal::AtomicFloatLoad(
                                &cumulative_regrets[table_index])))
                  : static_cast<double>(
                        strategy_store_internal::AtomicFloatLoad(
                            &cumulative_strategies[table_index]));
          probability_sum += out[legal_action_index];
          break;
        }
      }
    }
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
    const CardAbstraction& card_abstraction,
    std::shared_ptr<const FrozenStrategyTables>* frozen_tables,
    std::shared_ptr<FrozenStrategyTables>* mutable_tables,
    std::shared_ptr<MutableCumulativeArrays>* cumulative,
    bool* frozen,
    TraversalStats* stats)
    : config_(config),
      card_abstraction_(card_abstraction),
      frozen_tables_owner_(frozen_tables),
      mutable_tables_owner_(mutable_tables),
      cumulative_owner_(cumulative),
      frozen_(frozen),
      stats_(stats) {
  bind_tables();
}

void StrategyStore::bind_tables() {
  frozen_tables_ = frozen_tables_owner_->get();
  mutable_tables_ = mutable_tables_owner_->get();
  cumulative_ = cumulative_owner_->get();
}

FrozenStrategyTables& StrategyStore::mutable_tables() {
  return tables_for_growth();
}

FrozenStrategyTables& StrategyStore::tables_for_growth() {
  if (*frozen_ || mutable_tables_ == nullptr) {
    throw std::logic_error("Strategy tables are frozen");
  }
  return *mutable_tables_;
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
    absl::Span<const int> action_ids) {
  return block_for_row(get_or_create_info_set_row(address, action_ids),
                       action_ids.size());
}

std::optional<ActionBlock> StrategyStore::find_frozen(
    uint32_t public_state_id,
    int player,
    ComboId combo_id,
    size_t expected_action_count) {
  if (player < 0 || player >= kPlayerCount ||
      public_state_id >= frozen_tables().private_bucket_rows.size() ||
      public_state_id >=
          frozen_tables().frozen_info_set_action_offsets.size()) {
    return std::nullopt;
  }
  if (expected_action_count > std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("infoset action count exceeds uint16_t");
  }

  const PrivateBucketId private_bucket =
      private_bucket_for_frozen_row(public_state_id, combo_id);
  const uint32_t action_offset =
      frozen_info_set_action_offset(public_state_id, player, private_bucket);
  if (action_offset == FrozenStrategyTables::kInvalidActionOffset) {
    return std::nullopt;
  }
  return ActionBlock(this, action_offset,
                     static_cast<uint16_t>(expected_action_count));
}

void StrategyStore::average_strategy(uint32_t public_state_id,
                                     const PublicStateRow& row,
                                     int player,
                                     PrivateBucketId private_bucket,
                                     bool regret_only_training,
                                     absl::Span<double> out) {
  if (out.size() != row.action_count) {
    throw std::logic_error("average-strategy probability span size mismatch");
  }
  if (out.empty()) {
    return;
  }

  const double uniform_probability = 1.0 / out.size();
  std::optional<ActionBlock> block =
      find({public_state_id, player, private_bucket}, row.action_count);
  if (!block.has_value()) {
    std::fill(out.begin(), out.end(), uniform_probability);
    return;
  }
  block->average_strategy(
      regret_only_training,
      absl::Span<const int>(row.action_ids.data(), row.action_count),
      uniform_probability, out);
}

void StrategyStore::regret_matching_for_bucket(
    uint32_t public_state_id,
    int player,
    PrivateBucketId private_bucket,
    absl::Span<const int> legal_action_ids,
    absl::Span<double> out) {
  if (out.size() != legal_action_ids.size()) {
    throw std::logic_error("conditioned strategy span size mismatch");
  }
  if (out.empty()) {
    return;
  }

  std::optional<ActionBlock> block =
      find({public_state_id, player, private_bucket}, legal_action_ids.size());
  if (!block.has_value()) {
    strategy_store_internal::FillUniform(out);
    return;
  }
  block->regret_matching(legal_action_ids, out);
}

bool StrategyStore::prebuild_private_bucket_rows() {
  if (*frozen_) {
    return true;
  }

  FrozenStrategyTables& tables = tables_for_growth();
  tables.private_bucket_rows.resize(tables.public_state_rows.size());
  for (size_t public_state_id = 0;
       public_state_id < tables.public_state_rows.size(); ++public_state_id) {
    const PublicStateRow& row = tables.public_state_rows[public_state_id];
    const uint32_t bucket_count =
        card_abstraction_.private_bucket_count(row.state);
    if (bucket_count == 0 || bucket_count > kComboCount) {
      return false;
    }
    auto& bucket_row = tables.private_bucket_rows[public_state_id];
    for (int combo = 0; combo < kComboCount; ++combo) {
      const PrivateBucketId private_bucket =
          card_abstraction_.private_bucket(static_cast<ComboId>(combo),
                                           row.state);
      if (private_bucket >= bucket_count) {
        return false;
      }
      bucket_row[static_cast<size_t>(combo)] = private_bucket;
    }
  }
  return true;
}

bool StrategyStore::prebuild_frozen_info_set_action_offsets() {
  if (*frozen_) {
    return true;
  }

  FrozenStrategyTables& tables = tables_for_growth();
  tables.frozen_info_set_action_offsets.resize(tables.public_state_rows.size());
  for (auto& offset_row : tables.frozen_info_set_action_offsets) {
    for (auto& player_offsets : offset_row) {
      player_offsets.fill(FrozenStrategyTables::kInvalidActionOffset);
    }
  }

  for (size_t public_state_id = 0;
       public_state_id < tables.public_state_rows.size(); ++public_state_id) {
    const PublicStateRow& public_row = tables.public_state_rows[public_state_id];
    const PublicInfoSetSlab* slab =
        public_info_set_slab(static_cast<uint32_t>(public_state_id));
    if (slab == nullptr) {
      continue;
    }

    auto& offset_row =
        tables.frozen_info_set_action_offsets[public_state_id];
    for (int player = 0; player < kPlayerCount; ++player) {
      const PublicInfoSetSlabPlayer& player_slab = slab->players[player];
      auto& player_offsets = offset_row[player];
      for (int bucket = 0; bucket < kComboCount; ++bucket) {
        const InfoSetRow* info_set_row = find_info_set_row(
            player_slab, static_cast<PrivateBucketId>(bucket));
        if (info_set_row == nullptr) {
          continue;
        }
        if (info_set_row->action_count != public_row.action_count) {
          return false;
        }
        player_offsets[static_cast<size_t>(bucket)] =
            info_set_row->action_offset;
      }
    }
  }
  return true;
}

const StrategyStore::PublicInfoSetSlab* StrategyStore::public_info_set_slab(
    uint32_t public_state_id) const {
  const auto& slabs = frozen_tables().public_info_set_slabs;
  if (public_state_id >= slabs.size()) {
    return nullptr;
  }
  return slabs[public_state_id].get();
}

StrategyStore::PublicInfoSetSlab&
StrategyStore::get_or_create_public_info_set_slab(uint32_t public_state_id) {
  FrozenStrategyTables& tables = tables_for_growth();
  if (tables.public_info_set_slabs.size() <= public_state_id) {
    tables.public_info_set_slabs.resize(static_cast<size_t>(public_state_id) +
                                        1);
  }
  std::unique_ptr<PublicInfoSetSlab>& slab =
      tables.public_info_set_slabs[public_state_id];
  if (slab == nullptr) {
    slab = std::make_unique<PublicInfoSetSlab>();
  }
  return *slab;
}

const StrategyStore::InfoSetRow* StrategyStore::find_info_set_row(
    InfoSetAddress address) const {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return nullptr;
  }
  const PublicInfoSetSlab* slab = public_info_set_slab(address.public_state_id);
  if (slab == nullptr) {
    return nullptr;
  }
  const PublicInfoSetSlabPlayer& player_slab = slab->players[address.player];
  return find_info_set_row(player_slab, address.private_bucket);
}

const StrategyStore::InfoSetRow* StrategyStore::find_info_set_row(
    const PublicInfoSetSlabPlayer& player_slab,
    PrivateBucketId private_bucket) {
  if (private_bucket >= kComboCount) {
    return nullptr;
  }
  const size_t chunk_index =
      private_bucket / FrozenStrategyTables::kPrivateBucketChunkSize;
  const size_t chunk_offset =
      private_bucket % FrozenStrategyTables::kPrivateBucketChunkSize;
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
      private_bucket / FrozenStrategyTables::kPrivateBucketChunkSize;
  const size_t chunk_offset =
      private_bucket % FrozenStrategyTables::kPrivateBucketChunkSize;
  std::unique_ptr<PrivateRowChunk>& chunk =
      player_slab.private_row_chunks[chunk_index];
  if (chunk == nullptr) {
    chunk = std::make_unique<PrivateRowChunk>();
  }
  return chunk->rows[chunk_offset];
}

const StrategyStore::InfoSetRow* StrategyStore::get_or_create_info_set_row(
    InfoSetAddress address,
    absl::Span<const int> action_ids) {
  if (address.player < 0 || address.player >= kPlayerCount ||
      address.private_bucket >= kComboCount) {
    return nullptr;
  }

  if (const InfoSetRow* row = find_info_set_row(address)) {
    return row;
  }

  if (*frozen_ || (config_.max_info_sets > 0 &&
                   static_cast<int>(frozen_tables().info_set_count) >=
                       config_.max_info_sets)) {
    return nullptr;
  }

  InfoSetRow row = append_info_set_actions(action_ids);
  PublicInfoSetSlab& slab =
      get_or_create_public_info_set_slab(address.public_state_id);
  PublicInfoSetSlabPlayer& player_slab = slab.players[address.player];
  int32_t& row_id =
      get_or_create_private_row_slot(player_slab, address.private_bucket);
  row_id = static_cast<int32_t>(player_slab.rows.size());
  player_slab.rows.push_back(row);
  ++tables_for_growth().info_set_count;
  return &player_slab.rows.back();
}

StrategyStore::InfoSetRow StrategyStore::append_info_set_actions(
    absl::Span<const int> action_ids) {
  FrozenStrategyTables& tables = tables_for_growth();
  const size_t padding =
      (kCumulativeActionBlockAlignment -
       tables.action_ids.size() % kCumulativeActionBlockAlignment) %
      kCumulativeActionBlockAlignment;
  InfoSetRow row;
  row.action_offset = static_cast<uint32_t>(tables.action_ids.size() + padding);
  row.action_count = static_cast<uint16_t>(action_ids.size());
  const size_t required_action_capacity =
      tables.action_ids.size() + padding + action_ids.size();
  if (required_action_capacity > tables.action_ids.capacity()) {
    const size_t new_capacity =
        std::max(required_action_capacity,
                 tables.action_ids.empty() ? size_t{4096}
                                           : tables.action_ids.capacity() * 2);
    tables.action_ids.reserve(new_capacity);
    cumulative().cumulative_regrets.reserve(new_capacity);
    cumulative().cumulative_strategies.reserve(new_capacity);
  }
  for (size_t i = 0; i < padding; ++i) {
    tables.action_ids.push_back(0);
    cumulative().cumulative_regrets.push_back(0.0f);
    cumulative().cumulative_strategies.push_back(0.0f);
  }
  for (int action_id : action_ids) {
    tables.action_ids.push_back(action_id);
    cumulative().cumulative_regrets.push_back(0.0f);
    cumulative().cumulative_strategies.push_back(0.0f);
    record_action_entry_touches();
  }
  return row;
}

StrategyStore::PrivateBucketId StrategyStore::private_bucket_for_frozen_row(
    uint32_t public_state_id,
    ComboId combo_id) const {
  return frozen_tables().private_bucket_rows[public_state_id][combo_id];
}

uint32_t StrategyStore::frozen_info_set_action_offset(
    uint32_t public_state_id,
    int player,
    PrivateBucketId private_bucket) const {
  return frozen_tables().frozen_info_set_action_offsets[public_state_id][player]
                                                       [private_bucket];
}

}  // namespace poker
