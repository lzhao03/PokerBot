#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
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

  struct PrivateCards {
    static PrivateCards FromCombo(ComboId combo_id);
    CardMask mask() const;

    ComboId combo = 0;
  };

  struct TraversalDeal {
    std::array<PrivateCards, kPlayerCount> cards;

    const PrivateCards& player_cards(int player) const {
      return cards[static_cast<size_t>(player)];
    }

    CardMask known_private_cards() const {
      return cards[0].mask() | cards[1].mask();
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

  using OptionalTrainingRange =
      std::optional<std::reference_wrapper<const TrainingRangeView>>;

  enum class NodeGraphMode {
    kGrow,
    kSkipMissing,
    kRequirePresent,
  };

  struct NodeRef {
    uint32_t public_state_id = kInvalidPublicStateId;
    Board exact_board;
  };

  enum class ChildStatus {
    kOk,
    kMissing,
    kCapped,
    kInvalid,
  };

  struct ChildResult {
    ChildStatus status = ChildStatus::kInvalid;
    NodeRef node;
  };

  class NodeGraph {
   public:
    NodeGraph(CFRSolver& solver, NodeGraphMode mode);

    ChildResult action_child(
        NodeRef parent,
        int action_index);
    ChildResult sample_chance_child(
        NodeRef parent,
        CardMask known_private_cards);

   private:
    ChildResult make_child_result(
        std::optional<uint32_t> child_id,
        Board exact_board,
        const char* missing_message) const;

    CFRSolver& solver_;
    NodeGraphMode mode_;
  };

  // TODO: Hide action-conditioned range borrowing behind TraversalContext too.
  struct RangeScratchFrame {
    std::vector<TrainingRangeView> conditioned_ranges;
    std::array<TrainingRangeView, kPlayerCount> public_player_ranges;
  };

  struct TraversalScratch {
    explicit TraversalScratch(size_t depth_count);
    RangeScratchFrame& frame(size_t depth);

    std::vector<RangeScratchFrame> frames;
  };

  struct DecisionFrame {
    uint32_t public_state_id = kInvalidPublicStateId;
    Player player = Player::kA;
    StreetKind street = StreetKind::kPreflop;
    uint8_t action_count = 0;
    std::array<int, kMaxActionsPerNode> action_ids = {};

    absl::Span<const int> action_ids_span() const {
      return absl::Span<const int>(
          action_ids.data(), static_cast<size_t>(action_count));
    }
  };

  class TraversalContext {
   public:
    TraversalContext(TraversalDeal deal,
                     TraversalOptions options,
                     TraversalScratch& scratch,
                     OptionalTrainingRange player_a_range = {},
                     OptionalTrainingRange player_b_range = {});

    const TraversalDeal& deal() const { return deal_; }
    const TraversalOptions& options() const { return options_; }

    const PrivateCards& cards(int player) const {
      return deal_.player_cards(player);
    }

    CardMask known_private_cards() const {
      return deal_.known_private_cards();
    }

    double reach(int player) const {
      return reach_[static_cast<size_t>(player)];
    }

    double opponent_reach(int player) const {
      return reach_[static_cast<size_t>(1 - player)];
    }

    bool is_update_player(int player) const {
      return player == options_.update_player;
    }

    int iteration() const { return options_.iteration; }
    int depth() const { return depth_; }
    int max_depth() const { return options_.max_depth; }

    bool depth_limited() const {
      return options_.max_depth > 0 && depth_ >= options_.max_depth;
    }

    RangeScratchFrame& scratch_frame() {
      return scratch_->frame(static_cast<size_t>(depth_));
    }

    OptionalTrainingRange range(int player) const {
      return ranges_[static_cast<size_t>(player)];
    }

    OptionalTrainingRange range_without_mask(int player, CardMask blocked_mask);

    double average_strategy_weight(int player) const {
      return reach(player) * static_cast<double>(options_.iteration + 1);
    }

    RegretUpdateOptions regret_update_options() const;

    class ChildTraversalScope {
     public:
      ChildTraversalScope(TraversalContext& ctx,
                          int acting_player,
                          double action_probability,
                          OptionalTrainingRange player_a_range = {},
                          OptionalTrainingRange player_b_range = {},
                          bool override_ranges = false);

      ChildTraversalScope(TraversalContext& ctx,
                          OptionalTrainingRange player_a_range,
                          OptionalTrainingRange player_b_range);

      ChildTraversalScope(const ChildTraversalScope&) = delete;
      ChildTraversalScope& operator=(const ChildTraversalScope&) = delete;
      ~ChildTraversalScope();

     private:
      TraversalContext& ctx_;
      int player_ = 0;
      double previous_reach_ = 1.0;
      std::array<OptionalTrainingRange, kPlayerCount> previous_ranges_;
      bool restore_reach_ = false;
      bool restore_depth_ = false;
      bool restore_ranges_ = false;
    };

   private:
    TraversalDeal deal_;
    TraversalOptions options_;
    TraversalScratch* scratch_ = nullptr;
    std::array<double, kPlayerCount> reach_ = {1.0, 1.0};
    std::array<OptionalTrainingRange, kPlayerCount> ranges_;
    int depth_ = 0;
  };

  class ActionRangeConditioning {
   public:
    ActionRangeConditioning(CFRSolver& solver,
                            TraversalContext& ctx,
                            StreetKind street,
                            const Board& board,
                            uint32_t node_id,
                            int player,
                            absl::Span<const int> action_ids);

    bool enabled() const {
      return condition_player_a_ || condition_player_b_;
    }

    OptionalTrainingRange player_a_range_for(size_t action_index) const;
    OptionalTrainingRange player_b_range_for(size_t action_index) const;

   private:
    OptionalTrainingRange original_player_a_range_;
    OptionalTrainingRange original_player_b_range_;
    absl::Span<TrainingRangeView> conditioned_ranges_;
    bool condition_player_a_ = false;
    bool condition_player_b_ = false;
  };

  class CfrTraversal {
   public:
    CfrTraversal(CFRSolver& solver,
                 TraversalContext& ctx,
                 NodeGraph& graph)
        : solver_(solver), ctx_(ctx), graph_(graph) {}

    double value(NodeRef node);

   private:
    double terminal(NodeRef node, const PublicStateRow& row);
    double chance(NodeRef node);
    double depth_limit_value(NodeRef node, const PublicStateRow& row);
    double decision(NodeRef node, const PublicStateRow& row);

    CFRSolver& solver_;
    TraversalContext& ctx_;
    NodeGraph& graph_;
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
  TraversalDeal traversal_deal(RangeDeal deal) const;
  TraversalOptions traversal_options(int iteration, int max_depth) const;
  void log_training_summary() const;
  template <typename WorkerFn, typename AccumulateFn>
  void run_sharded(int work_count,
                   int worker_count,
                   int first_index,
                   WorkerFn&& worker_fn,
                   AccumulateFn&& accumulate_fn);
  std::optional<NodeRef> root_node_ref(uint32_t root_id) const;
  static std::optional<DecisionFrame> make_decision_frame(
      uint32_t node_id,
      const PublicStateRow& row);
  double cfr(NodeRef node, TraversalContext& ctx, NodeGraph& graph);
  template <typename EvalChild>
  double sample_chance_children(int samples,
                                NodeRef node,
                                CardMask known_private_cards,
                                NodeGraph& graph,
                                EvalChild&& eval_child);

  bool prebuild_info_set_rows(const TrainingRangeView& a_view,
                              const TrainingRangeView& b_view,
                              absl::Span<const std::optional<Board>> row_boards);
  absl::Span<TrainingRangeView> condition_ranges_for_actions(
      const TrainingRangeView& range,
      StreetKind street,
      const Board& board,
      uint32_t node_id,
      int player,
      absl::Span<const int> action_ids,
      RangeScratchFrame& scratch_frame);
  double terminal_utility(const PublicStateRow& row,
                          const Board& board,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(NodeRef node,
                                const TraversalDeal& deal,
                                NodeGraph& graph);
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
  CardAbstraction card_abstraction_;
  BettingAbstraction betting_abstraction_;
  bool require_frozen_children_ = false;
  SolverStorage storage_;
  StrategyStore strategy_store_;
  PublicStateGraph public_graph_;
};

}  // namespace poker
