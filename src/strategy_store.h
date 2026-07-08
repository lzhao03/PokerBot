#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>

#include "absl/types/span.h"
#include "src/build_flags.h"
#include "src/card_abstraction.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"

namespace poker {
namespace strategy_store_internal {

inline void AtomicFloatAdd(float* target, float delta) {
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

inline int64_t AtomicCFRPlusRegretUpdate(float* target, float delta) {
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

inline float AtomicFloatLoad(const float* src) {
  const int32_t bits = __atomic_load_n(
      reinterpret_cast<const int32_t*>(src), __ATOMIC_RELAXED);
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

inline void FillUniform(absl::Span<double> out) {
  if (out.empty()) {
    return;
  }
  std::fill(out.begin(), out.end(), 1.0 / out.size());
}

}  // namespace strategy_store_internal

enum class RegretLoadMode {
  kPlain,
  kAtomic,
};

enum class RegretUpdateMode {
  kPlain,
  kAtomic,
};

struct RegretUpdateOptions {
  RegretUpdateMode mode = RegretUpdateMode::kPlain;
  bool record_atomic_retry_stats = false;
};

struct ActionBlockOptions {
  RegretLoadMode regret_load_mode = RegretLoadMode::kAtomic;
  RegretUpdateMode update_mode = RegretUpdateMode::kAtomic;
  bool record_atomic_retry_stats = false;
};

struct TraversalStats {
  int64_t cfr_updates = 0;
  int64_t preflop_updates = 0;
  int64_t flop_updates = 0;
  int64_t turn_updates = 0;
  int64_t river_updates = 0;
  int max_decision_depth = 0;
  int64_t child_nodes_created = 0;
  int64_t chance_samples = 0;
  int64_t terminal_utility_calls = 0;
  int64_t fold_utility_calls = 0;
  int64_t showdown_utility_calls = 0;
  int64_t action_entry_touches = 0;
  int64_t atomic_regret_update_retries = 0;
  int64_t betting_history_transition_hits = 0;
  int64_t betting_history_transition_misses = 0;
};

class StrategyStore;

class ActionBlock {
 public:
  ActionBlock() = default;

  bool valid() const { return store_ != nullptr; }
  size_t action_count() const { return action_count_; }
  uint32_t action_offset() const { return action_offset_; }

  absl::Span<const int> action_ids() const;
  void regret_matching(RegretLoadMode mode, absl::Span<double> out) const;
  void regret_matching(absl::Span<const int> legal_action_ids,
                       absl::Span<double> out) const;
  void add_cfr_plus_regret(size_t action_index,
                           float delta,
                           RegretUpdateOptions options) const;
  void add_average_strategy(absl::Span<const double> probs,
                            double reach_weight,
                            RegretUpdateMode mode) const;
  void average_strategy(bool regret_only_training,
                        absl::Span<const int> legal_action_ids,
                        double fallback_probability,
                        absl::Span<double> out) const;

 private:
  friend class StrategyStore;

  ActionBlock(StrategyStore* store,
              uint32_t action_offset,
              uint16_t action_count)
      : store_(store),
        action_offset_(action_offset),
        action_count_(action_count) {}

  StrategyStore* store_ = nullptr;
  uint32_t action_offset_ = 0;
  uint16_t action_count_ = 0;
};

class StrategyStore {
 public:
  using PrivateBucketId = FrozenStrategyTables::PrivateBucketId;
  using InfoSetAddress = FrozenStrategyTables::InfoSetAddress;
  using InfoSetRow = FrozenStrategyTables::InfoSetRow;
  using PublicStateRow = FrozenStrategyTables::PublicStateRow;
  using PrivateRowChunk = FrozenStrategyTables::PrivateRowChunk;
  using PublicInfoSetSlabPlayer =
      FrozenStrategyTables::PublicInfoSetSlabPlayer;
  using PublicInfoSetSlab = FrozenStrategyTables::PublicInfoSetSlab;

  StrategyStore(
      const SolverConfig& config,
      const CardAbstraction& card_abstraction,
      std::shared_ptr<const FrozenStrategyTables>* frozen_tables,
      std::shared_ptr<FrozenStrategyTables>* mutable_tables,
      std::shared_ptr<MutableCumulativeArrays>* cumulative,
      bool* frozen,
      TraversalStats* stats);

  void bind_tables();
  FrozenStrategyTables& mutable_tables();

  std::optional<ActionBlock> find(InfoSetAddress address,
                                  size_t expected_action_count);
  std::optional<ActionBlock> get_or_create(InfoSetAddress address,
                                           absl::Span<const int> action_ids);
  std::optional<ActionBlock> find_frozen(uint32_t public_state_id,
                                         int player,
                                         ComboId combo_id,
                                         size_t expected_action_count);

  void regret_matching_or_uniform(std::optional<ActionBlock> block,
                                  size_t legal_action_count,
                                  RegretLoadMode load_mode,
                                  absl::Span<double> out);
  void average_strategy(uint32_t public_state_id,
                        const PublicStateRow& row,
                        int player,
                        PrivateBucketId private_bucket,
                        bool regret_only_training,
                        absl::Span<double> out);
  void regret_matching_for_bucket(uint32_t public_state_id,
                                  int player,
                                  PrivateBucketId private_bucket,
                                  absl::Span<const int> legal_action_ids,
                                  absl::Span<double> out);

  bool prebuild_private_bucket_rows();
  bool prebuild_frozen_info_set_action_offsets();

 private:
  friend class ActionBlock;

  const FrozenStrategyTables& frozen_tables() const;
  FrozenStrategyTables& tables_for_growth();
  MutableCumulativeArrays& cumulative();
  const MutableCumulativeArrays& cumulative() const;

  ActionBlock block_for_row(const InfoSetRow& row);
  std::optional<ActionBlock> block_for_row(const InfoSetRow* row,
                                           size_t expected_action_count);
  const PublicInfoSetSlab* public_info_set_slab(uint32_t public_state_id) const;
  PublicInfoSetSlab& get_or_create_public_info_set_slab(
      uint32_t public_state_id);
  const InfoSetRow* find_info_set_row(InfoSetAddress address) const;
  static const InfoSetRow* find_info_set_row(
      const PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  static int32_t& get_or_create_private_row_slot(
      PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  const InfoSetRow* get_or_create_info_set_row(
      InfoSetAddress address,
      absl::Span<const int> action_ids);
  InfoSetRow append_info_set_actions(absl::Span<const int> action_ids);
  PrivateBucketId private_bucket_for_frozen_row(uint32_t public_state_id,
                                                ComboId combo_id) const;
  uint32_t frozen_info_set_action_offset(
      uint32_t public_state_id,
      int player,
      PrivateBucketId private_bucket) const;

  void record_action_entry_touches(int64_t count = 1) const;
  void record_atomic_regret_update_retries(int64_t count) const;

  const SolverConfig& config_;
  const CardAbstraction& card_abstraction_;
  std::shared_ptr<const FrozenStrategyTables>* frozen_tables_owner_;
  std::shared_ptr<FrozenStrategyTables>* mutable_tables_owner_;
  std::shared_ptr<MutableCumulativeArrays>* cumulative_owner_;
  const FrozenStrategyTables* frozen_tables_ = nullptr;
  FrozenStrategyTables* mutable_tables_ = nullptr;
  MutableCumulativeArrays* cumulative_ = nullptr;
  bool* frozen_;
  TraversalStats* stats_;
};

inline const FrozenStrategyTables& StrategyStore::frozen_tables() const {
  return *frozen_tables_;
}

inline MutableCumulativeArrays& StrategyStore::cumulative() {
  return *cumulative_;
}

inline const MutableCumulativeArrays& StrategyStore::cumulative() const {
  return *cumulative_;
}

inline void StrategyStore::record_action_entry_touches(int64_t count) const {
  if constexpr (kTraversalStatsEnabled) {
    stats_->action_entry_touches += count;
  }
}

inline void StrategyStore::record_atomic_regret_update_retries(
    int64_t count) const {
  if constexpr (kCasRetryStatsEnabled) {
    stats_->atomic_regret_update_retries += count;
  }
}

inline absl::Span<const int> ActionBlock::action_ids() const {
  if (!valid()) {
    return {};
  }
  const std::vector<int>& ids = store_->frozen_tables().action_ids;
  return absl::Span<const int>(
      ids.data() + static_cast<size_t>(action_offset_), action_count_);
}

inline void ActionBlock::regret_matching(RegretLoadMode mode,
                                         absl::Span<double> out) const {
  if (!valid() || out.size() != action_count_) {
    throw std::logic_error("regret-matching action count mismatch");
  }

  const auto& regrets = store_->cumulative().cumulative_regrets;
  double sum_positive_regrets = 0.0;
  for (size_t action_index = 0; action_index < out.size(); ++action_index) {
    store_->record_action_entry_touches();
    const size_t table_index =
        static_cast<size_t>(action_offset_) + action_index;
    const float raw_regret =
        mode == RegretLoadMode::kAtomic
            ? strategy_store_internal::AtomicFloatLoad(&regrets[table_index])
            : regrets[table_index];
    const double positive_regret =
        std::max(0.0, static_cast<double>(raw_regret));
    out[action_index] = positive_regret;
    sum_positive_regrets += positive_regret;
  }

  if (sum_positive_regrets <= 0.0) {
    strategy_store_internal::FillUniform(out);
    return;
  }
  for (double& probability : out) {
    probability /= sum_positive_regrets;
  }
}

inline void ActionBlock::add_cfr_plus_regret(
    size_t action_index,
    float delta,
    RegretUpdateOptions options) const {
  if (!valid() || action_index >= action_count_) {
    throw std::logic_error("regret update action index out of range");
  }

  const size_t table_index =
      static_cast<size_t>(action_offset_) + action_index;
  auto& regrets = store_->cumulative().cumulative_regrets;
  if (table_index >= regrets.size()) {
    throw std::logic_error("regret update table index out of range");
  }

  store_->record_action_entry_touches(2);
  float* regret_entry = &regrets[table_index];
  if (options.mode == RegretUpdateMode::kAtomic) {
    const int64_t retries =
        strategy_store_internal::AtomicCFRPlusRegretUpdate(regret_entry,
                                                           delta);
    if (options.record_atomic_retry_stats) {
      store_->record_atomic_regret_update_retries(retries);
    }
    return;
  }
  *regret_entry = std::max(0.0f, *regret_entry + delta);
}

inline void ActionBlock::add_average_strategy(absl::Span<const double> probs,
                                              double reach_weight,
                                              RegretUpdateMode mode) const {
  if (!valid() || probs.size() != action_count_) {
    throw std::logic_error("average-strategy probability span size mismatch");
  }

  auto& strategies = store_->cumulative().cumulative_strategies;
  for (size_t action_index = 0; action_index < probs.size(); ++action_index) {
    const size_t table_index =
        static_cast<size_t>(action_offset_) + action_index;
    if (table_index >= strategies.size()) {
      throw std::logic_error("average-strategy table index out of range");
    }

    store_->record_action_entry_touches(2);
    const float delta =
        static_cast<float>(reach_weight * probs[action_index]);
    if (mode == RegretUpdateMode::kAtomic) {
      strategy_store_internal::AtomicFloatAdd(&strategies[table_index], delta);
    } else {
      strategies[table_index] += delta;
    }
  }
}

inline void StrategyStore::regret_matching_or_uniform(
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
    strategy_store_internal::FillUniform(out);
    return;
  }
  block->regret_matching(load_mode, out);
}

}  // namespace poker
