#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/types/span.h"
#include "src/card_abstraction.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"
#include "src/traversal_stats.h"

namespace poker {

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

struct SolverStorage {
  std::shared_ptr<StrategyTables> mutable_tables =
      std::make_shared<StrategyTables>();
  std::shared_ptr<const StrategyTables> frozen_tables = mutable_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative =
      std::make_shared<MutableCumulativeArrays>();
  bool frozen = false;

  StrategyTables& mutable_ref();
  const StrategyTables& frozen_ref() const;
  MutableCumulativeArrays& cumulative_ref();
  const MutableCumulativeArrays& cumulative_ref() const;
  void freeze();
  void bind_frozen(std::shared_ptr<const StrategyTables> frozen_in,
                   std::shared_ptr<MutableCumulativeArrays> cumulative_in);
};

class StrategyStore {
 public:
  using PrivateBucketId = StrategyTables::PrivateBucketId;
  using InfoSetAddress = StrategyTables::InfoSetAddress;
  using InfoSetRow = StrategyTables::InfoSetRow;
  using PublicStateRow = StrategyTables::PublicStateRow;
  using PrivateRowChunk = StrategyTables::PrivateRowChunk;
  using PublicInfoSetSlabPlayer =
      StrategyTables::PublicInfoSetSlabPlayer;
  using PublicInfoSetSlab = StrategyTables::PublicInfoSetSlab;

  StrategyStore(
      const SolverConfig& config,
      SolverStorage& storage,
      TraversalStats* stats);

  StrategyTables& mutable_tables();

  std::optional<ActionBlock> find(InfoSetAddress address,
                                  size_t expected_action_count);
  std::optional<ActionBlock> get_or_create(InfoSetAddress address,
                                           absl::Span<const int> action_ids);
  std::optional<ActionBlock> find_frozen(uint32_t public_state_id,
                                         int player,
                                         PrivateBucketId private_bucket,
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

  bool prebuild_frozen_info_set_action_offsets();

 private:
  friend class ActionBlock;

  const StrategyTables& frozen_tables() const;
  StrategyTables& tables_for_growth();
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
  uint32_t frozen_info_set_action_offset(
      uint32_t public_state_id,
      int player,
      PrivateBucketId private_bucket) const;

  const SolverConfig& config_;
  SolverStorage& storage_;
  TraversalStats* stats_;
};

}  // namespace poker
