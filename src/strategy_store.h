#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/types/span.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"
#include "src/solver_stats.h"

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
  size_t action_count() const { return action_count_; }

  void regret_matching(RegretLoadMode mode, absl::Span<double> out) const;
  void add_cfr_plus_regret(size_t action_index,
                           float delta,
                           RegretUpdateOptions options) const;
  void add_average_strategy(absl::Span<const double> probs,
                            double reach_weight,
                            RegretUpdateMode mode) const;
  void average_strategy(bool regret_only_training,
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

  StrategyStore* store_;
  uint32_t action_offset_;
  uint16_t action_count_;
};

struct SolverStorage {
  std::shared_ptr<StrategyTables> mutable_tables =
      std::make_shared<StrategyTables>();
  std::shared_ptr<const StrategyTables> frozen_tables = mutable_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative =
      std::make_shared<MutableCumulativeArrays>();

  bool is_frozen() const noexcept { return mutable_tables == nullptr; }
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
  using InfoSetKey = StrategyTables::InfoSetKey;
  using InfoSetRow = StrategyTables::InfoSetRow;
  using Node = StrategyTables::Node;
  using GrowingPublicInfoSets = StrategyTables::GrowingPublicInfoSets;
  using FrozenInfoSetEntry = StrategyTables::FrozenInfoSetEntry;
  using FrozenPublicInfoSetRange = StrategyTables::FrozenPublicInfoSetRange;

  StrategyStore(
      const SolverConfig& config,
      SolverStorage& storage,
      TraversalStats* stats);

  StrategyTables& mutable_tables();

  std::optional<ActionBlock> find(InfoSetKey key,
                                  size_t expected_action_count);
  std::optional<ActionBlock> get_or_create(InfoSetKey key,
                                           size_t action_count);
  std::optional<ActionBlock> find_frozen(InfoSetKey key,
                                         size_t expected_action_count);

  void regret_matching_or_uniform(std::optional<ActionBlock> block,
                                  size_t legal_action_count,
                                  RegretLoadMode load_mode,
                                  absl::Span<double> out);
  void average_strategy(InfoSetKey key,
                        size_t action_count,
                        bool regret_only_training,
                        absl::Span<double> out);
  void regret_matching_for_observation(InfoSetKey key,
                                       size_t action_count,
                                       absl::Span<double> out);

  bool build_frozen_info_set_index();

 private:
  friend class ActionBlock;

  const StrategyTables& frozen_tables() const;
  StrategyTables& tables_for_growth();
  MutableCumulativeArrays& cumulative();
  const MutableCumulativeArrays& cumulative() const;

  ActionBlock block_for_row(const InfoSetRow& row);
  std::optional<ActionBlock> block_for_row(const InfoSetRow* row,
                                           size_t expected_action_count);
  const GrowingPublicInfoSets* growing_rows(NodeId node_id) const;
  GrowingPublicInfoSets& get_or_create_growing_rows(NodeId node_id);
  const InfoSetRow* find_growing_row(InfoSetKey key) const;
  const FrozenInfoSetEntry* find_frozen_entry(InfoSetKey key) const;
  const InfoSetRow* get_or_create_info_set_row(
      InfoSetKey key,
      size_t action_count);
  InfoSetRow append_info_set_actions(size_t action_count);
  bool action_count_matches(NodeId node_id, size_t action_count) const;

  const SolverConfig& config_;
  SolverStorage& storage_;
  TraversalStats* stats_;
};

}  // namespace poker
