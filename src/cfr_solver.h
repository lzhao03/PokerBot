#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/public_state_graph.h"
#include "src/strategy_store.h"
#include "src/strategy_tables.h"
#include "src/training_range.h"

namespace poker {

struct CFRSolverTestAccess;

class CFRSolver {
 public:
  using TraversalStats = poker::TraversalStats;
  using TrainingRunStats = poker::TrainingRunStats;

  CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config,
            const CompactPublicState& initial_state);
  CFRSolver(const SolverConfig& config,
            const ExactGameState& initial_state);

  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);

  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);

  double get_expected_value(int player_id) const;
  int get_iterations_run() const { return iterations_run_; }
  int64_t get_cfr_update_count() const { return cfr_update_count_; }
  size_t get_info_set_count() const {
    return tables().info_set_count;
  }
  size_t get_public_state_count() const {
    return rows().size();
  }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  TrainingRunStats get_last_training_run_stats() const {
    return last_training_run_stats_;
  }
  static bool traversal_stats_enabled();

 private:
  friend struct CFRSolverTestAccess;

  const StrategyTables& tables() const {
    return storage_.frozen_ref();
  }
  StrategyTables& mtables() {
    return strategy_store_.mutable_tables();
  }
  MutableCumulativeArrays& arrays() {
    return storage_.cumulative_ref();
  }
  const std::vector<StrategyTables::PublicStateRow>& rows() const {
    return tables().public_state_rows;
  }

  using PrivateBucketId = StrategyTables::PrivateBucketId;
  using InfoSetAddress = StrategyTables::InfoSetAddress;
  using PublicStateRow = StrategyTables::PublicStateRow;

  struct Deal {
    std::array<ComboId, kPlayerCount> hands = {};
    CardMask blocked_mask = 0;

    ComboId hand(int player) const {
      return hands[static_cast<size_t>(player)];
    }

    CardMask known_private_cards() const {
      return blocked_mask;
    }
  };

  struct TraversalOptions {
    int update_player = 0;
    int iteration = 0;
    int max_depth = 0;
    RegretLoadMode regret_load_mode = RegretLoadMode::kAtomic;
    RegretUpdateMode regret_update_mode = RegretUpdateMode::kAtomic;
    bool write_average_strategy = true;
    bool use_fixed_infoset_lookup = false;
    bool record_atomic_retry_stats = false;
  };

  struct NodeRef {
    uint32_t public_state_id = kInvalidPublicStateId;
    Board exact_board;
    BoardFeatures board_features;
  };

  class MutableTraversalGraph {
   public:
    explicit MutableTraversalGraph(CFRSolver& solver);

    std::optional<NodeRef> action_child(
        NodeRef parent,
        int action_index);
    std::optional<NodeRef> sample_chance_child(
        NodeRef parent,
        CardMask known_private_cards);

   private:
    CFRSolver& solver_;
  };

  class FrozenTraversalGraph {
   public:
    explicit FrozenTraversalGraph(CFRSolver& solver);

    NodeRef action_child(NodeRef parent, int action_index) const;
    NodeRef sample_chance_child(NodeRef parent,
                                CardMask known_private_cards);

   private:
    uint32_t required_action_child_id(uint32_t parent_public_state_id,
                                      int action_index) const;
    uint32_t required_chance_child_id(
        uint32_t parent_public_state_id,
        const ExactGameState& child_state) const;

    CFRSolver& solver_;
  };

  struct RangeScratchFrame {
    std::array<TrainingRangeView, kPlayerCount> filtered_ranges;
    std::array<TrainingRangeView, kMaxActionsPerNode> conditioned_ranges;
  };

  struct TraversalScratch {
    explicit TraversalScratch(size_t depth_count);
    RangeScratchFrame& frame(size_t depth);

    std::vector<RangeScratchFrame> frames;
  };

  struct TraversalRun {
    Deal deal;
    TraversalOptions options;
    TraversalScratch* scratch = nullptr;
  };

  struct TraversalFrame {
    std::array<double, kPlayerCount> reach = {1.0, 1.0};
    std::array<const TrainingRangeView*, kPlayerCount> ranges = {
        nullptr, nullptr};
    uint16_t decision_depth = 0;
    uint16_t scratch_depth = 0;
  };

  template <typename Graph>
  class CfrTraversal {
   public:
    CfrTraversal(CFRSolver& solver,
                 TraversalRun& run,
                 Graph& graph)
        : solver_(solver), run_(run), graph_(graph) {}

    double value(NodeRef node, const TraversalFrame& frame);

   private:
    double terminal(NodeRef node, const PublicStateRow& row);
    double chance(NodeRef node, const TraversalFrame& frame);
    double depth_limit_value(NodeRef node, const PublicStateRow& row);
    double decision(NodeRef node,
                    const PublicStateRow& row,
                    const TraversalFrame& frame);

    CFRSolver& solver_;
    TraversalRun& run_;
    Graph& graph_;
  };

  bool prepare_prebuilt_training(
      uint32_t root_id,
      int max_depth,
      const TrainingRangeView& a_view,
      const TrainingRangeView& b_view);
  void run_growing_iterations(
      int iterations,
      uint32_t root_id,
      RangeSampler& sampler,
      const TrainingRangeView& a_view,
      const TrainingRangeView& b_view,
      int max_depth);
  void run_fixed_storage_iterations(
      int iterations,
      int num_threads,
      uint32_t root_id,
      const Board& root_board,
      const RangeSampler& sampler,
      const TrainingRange& a_range,
      const TrainingRange& b_range);
  Deal traversal_deal(RangeDeal deal) const;
  TraversalOptions traversal_options(int iteration, int max_depth) const;
  void log_training_summary() const;
  template <typename WorkerFn, typename AccumulateFn>
  void run_sharded(int work_count,
                   int worker_count,
                   int first_index,
                   WorkerFn&& worker_fn,
                   AccumulateFn&& accumulate_fn);
  template <typename Graph>
  double cfr(NodeRef node,
             TraversalRun& run,
             const TraversalFrame& frame,
             Graph& graph);
  template <typename Graph, typename EvalChild>
  double sample_chance_children(int samples,
                                NodeRef node,
                                CardMask known_private_cards,
                                Graph& graph,
                                EvalChild&& eval_child);

  bool prebuild_info_set_rows(const TrainingRangeView& a_view,
                              const TrainingRangeView& b_view,
                              absl::Span<const std::optional<Board>> row_boards);
  absl::Span<TrainingRangeView> condition_ranges_for_actions(
      const TrainingRangeView& range,
      StreetKind street,
      const Board& board,
      const BoardFeatures& features,
      uint32_t node_id,
      int player,
      size_t action_count,
      RangeScratchFrame& scratch_frame);
  double terminal_utility(const PublicStateRow& row,
                          const Board& board,
                          ComboId player_a_hand,
                          ComboId player_b_hand);
  double evaluate_strategy_node(NodeRef node,
                                const Deal& deal,
                                MutableTraversalGraph& graph);
  double evaluate_strategy_node(NodeRef node,
                                const Deal& deal,
                                FrozenTraversalGraph& graph);
  template <typename Graph>
  double evaluate_strategy_node_impl(NodeRef node,
                                     const Deal& deal,
                                     Graph& graph);
  double evaluate_strategy_samples(
      int samples,
      uint32_t root_id,
      const Board& root_board,
      const RangeSampler& sampler,
      bool allow_parallel);
  SolverConfig config_;
  ExactGameState initial_state_;
  std::mt19937 rng_;
  double cumulative_root_utility_ = 0.0;
  int iterations_run_ = 0;
  int64_t cfr_update_count_ = 0;
  TraversalStats traversal_stats_;
  TrainingRunStats last_training_run_stats_;
  BettingAbstraction betting_abstraction_;
  SolverStorage storage_;
  StrategyStore strategy_store_;
  GraphBuilder graph_builder_;
};

}  // namespace poker
