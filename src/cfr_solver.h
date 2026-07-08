#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"
#include "src/training_range.h"

namespace poker {

class TerminalUtilityCache;

class CFRSolver {
 public:
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

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  struct TrainingRunStats {
    bool public_state_prebuild_complete = false;
    bool betting_history_transition_prebuild_complete = false;
    bool action_transition_prebuild_complete = false;
    bool chance_transition_prebuild_complete = false;
    bool info_set_prebuild_complete = false;
    bool private_bucket_prebuild_complete = false;
    bool frozen_info_set_lookup_prebuild_complete = false;
    int64_t prebuild_public_states = 0;
    int64_t prebuild_betting_histories = 0;
    int64_t prebuild_betting_history_transitions = 0;
    int64_t missing_betting_history_transitions = 0;
    int64_t prebuild_action_transitions = 0;
    int64_t missing_action_transitions = 0;
    int64_t prebuild_chance_transitions = 0;
    int64_t missing_chance_transitions = 0;
    int64_t prebuild_info_sets = 0;
    int64_t prebuild_action_entries = 0;
    int64_t prebuild_private_bucket_rows = 0;
    int64_t prebuild_frozen_info_set_lookup_rows = 0;
    double prebuild_seconds = 0.0;
    double info_set_prebuild_seconds = 0.0;
    int warmup_iterations = 0;
    int frozen_iterations = 0;
    double warmup_seconds = 0.0;
    double frozen_seconds = 0.0;
    int64_t warmup_cfr_updates = 0;
    int64_t frozen_cfr_updates = 0;
  };

  CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config, const GameState& initial_state);

  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);

  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);

  double get_expected_value(int player_id) const;
  int get_iterations_run() const { return iterations_run_; }
  int64_t get_cfr_update_count() const { return cfr_update_count_; }
  size_t get_info_set_count() const { return frozen_tables_->info_set_count; }
  size_t get_public_state_count() const {
    return frozen_tables_->public_state_rows.size();
  }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  TrainingRunStats get_last_training_run_stats() const {
    return last_training_run_stats_;
  }
  void add_traversal_stats(const TraversalStats& stats);
  UtilityCacheStats get_utility_cache_stats() const;
  static bool traversal_stats_enabled();

 private:
  using PrivateBucketId = FrozenStrategyTables::PrivateBucketId;
  using BettingHistoryKey = FrozenStrategyTables::BettingHistoryKey;
  using PublicBucketId = FrozenStrategyTables::PublicBucketId;
  using PublicStateKey = FrozenStrategyTables::PublicStateKey;
  using ChanceTransitionKey = FrozenStrategyTables::ChanceTransitionKey;
  using BettingHistoryRow = FrozenStrategyTables::BettingHistoryRow;
  using InfoSetRow = FrozenStrategyTables::InfoSetRow;
  using InfoSetAddress = FrozenStrategyTables::InfoSetAddress;
  using PublicStateRow = FrozenStrategyTables::PublicStateRow;
  using PrivateRowChunk = FrozenStrategyTables::PrivateRowChunk;
  using PublicInfoSetSlabPlayer = FrozenStrategyTables::PublicInfoSetSlabPlayer;
  using PublicInfoSetSlab = FrozenStrategyTables::PublicInfoSetSlab;
  static constexpr int kPrivateBucketChunkSize =
      FrozenStrategyTables::kPrivateBucketChunkSize;

  struct InfoSetHandle {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

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

  struct PrivateCards {
    static PrivateCards FromCombo(ComboId combo_id);
    CardMask mask() const;

    ComboId combo = 0;
  };

  struct ExactBoardState {
    std::array<CardId, kMaxBoardCards> cards = {};
    uint8_t count = 0;
    CardMask mask = 0;
  };

  using OptionalTrainingRange =
      std::optional<std::reference_wrapper<const TrainingRangeView>>;
  using StrategyProbabilities = absl::InlinedVector<double, 8>;

  static constexpr uint32_t kCappedPublicStateId =
      GameTree::kInvalidPublicStateId - 1;

  enum class ChildResolverMode {
    kGrow,
    kSkipMissing,
    kRequirePresent,
  };

  struct NodeRef {
    uint32_t public_state_id = GameTree::kInvalidPublicStateId;
    ExactBoardState exact_board;
  };

  struct NodeView {
    NodeRef ref;
    PublicStateRow row;

    CompactPublicState exact_state() const;
    CardMask board_mask() const { return ref.exact_board.mask; }
  };

  class ChildResolver {
   public:
    ChildResolver(CFRSolver& solver, ChildResolverMode mode);

    std::optional<NodeRef> action_child(
        NodeRef parent,
        int action_index);
    std::optional<NodeRef> sample_chance_child(
        NodeRef parent,
        CardMask known_private_cards);

   private:
    CFRSolver& solver_;
    ChildResolverMode mode_;
  };

  // TODO: Move buffer borrowing behind TraversalScratch methods so traversal
  // code does not manually pass scratch vectors/ranges around.
  struct RangeScratchFrame {
    std::vector<TrainingRangeView> conditioned_ranges;
    TrainingRangeView public_player_a_range;
    TrainingRangeView public_player_b_range;
  };

  struct TraversalScratch {
    explicit TraversalScratch(size_t depth_count) {
      frames.reserve(depth_count);
    }

    RangeScratchFrame& frame(size_t depth) {
      if (depth >= frames.capacity()) {
        throw std::logic_error("TraversalScratch depth reserve exhausted");
      }
      while (frames.size() <= depth) {
        frames.emplace_back();
      }
      return frames[depth];
    }

    std::vector<RangeScratchFrame> frames;
  };

  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache);
  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            GameState initial_state);

  FrozenStrategyTables& mutable_tables();
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
  bool prepare_frozen_training(
      uint32_t root_public_state_id,
      int num_threads,
      int max_depth,
      bool can_use_frozen_regret_only,
      const TrainingRangeView& player_a_hands_view,
      const TrainingRangeView& player_b_hands_view);
  int run_warmup_phase(int iterations,
                       uint32_t root_public_state_id,
                       const CompactPublicState& root_state,
                       RangeSampler& range_sampler,
                       const TrainingRangeView& player_a_hands_view,
                       const TrainingRangeView& player_b_hands_view,
                       int max_depth,
                       bool should_run_frozen_phase,
                       bool can_use_frozen_regret_only);
  void maybe_run_frozen_phase(
      int iterations,
      int completed_warmup,
      int num_threads,
      uint32_t root_public_state_id,
      const RangeSampler& range_sampler,
      const TrainingRange& player_a_training_range,
      const TrainingRange& player_b_training_range);
  void run_frozen_iterations(int iterations,
                             int num_threads,
                             uint32_t root_public_state_id,
                             const RangeSampler& range_sampler,
                             const TrainingRange& player_a_training_range,
                             const TrainingRange& player_b_training_range);
  static ExactBoardState ExactBoardFromState(
      const CompactPublicState& state);
  static void ApplyExactBoard(CompactPublicState& state,
                              const ExactBoardState& board);
  std::optional<NodeView> view(NodeRef node) const;
  std::optional<NodeRef> root_node_ref(uint32_t root_public_state_id) const;
  ChildResolverMode default_child_resolver_mode() const;
  double cfr_with_ranges(
      NodeRef node,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range,
      ChildResolver& children);
  double cfr_frozen_regret_only(
      NodeRef node,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int depth,
      bool use_atomic_updates);
  double chance_sampling_cfr(
      NodeRef node,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range,
      ChildResolver& children);
  double chance_sampling_frozen_regret_only(
      NodeRef node,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int depth,
      bool use_atomic_updates);

  BettingHistoryKey make_betting_history_key(
      const CompactPublicState& state) const;
  BettingHistoryRow make_betting_history_row(
      const CompactPublicState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const CompactPublicState& state) const;
  PublicStateRow make_public_state_row(uint32_t betting_history_id,
                                       CompactPublicState state);
  uint32_t get_or_create_betting_history_id(BettingHistoryKey key,
                                            BettingHistoryRow row);
  uint32_t get_or_create_betting_history_id(
      const CompactPublicState& state);
  uint32_t get_or_create_action_child_betting_history_id(
      uint32_t parent_betting_history_id,
      int action_index,
      const CompactPublicState& child_state);
  uint32_t get_or_create_chance_child_betting_history_id(
      uint32_t parent_betting_history_id,
      const CompactPublicState& child_state);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const PublicStateRow& row);
  std::optional<uint32_t> get_or_create_public_state_row(
      uint32_t betting_history_id,
      CompactPublicState state);
  std::optional<uint32_t> get_or_create_public_state_row(
      const CompactPublicState& state);
  std::optional<uint32_t> action_child_public_state(
      uint32_t public_state_id,
      int action_index) const;
  std::optional<uint32_t> chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state) const;
  std::optional<uint32_t> chance_child_public_state(
      uint32_t public_state_id,
      absl::Span<const CardId> cards) const;
  bool for_each_required_chance_transition(
      const PublicStateRow& row,
      const std::function<bool(const CompactPublicState&,
                               absl::Span<const CardId>)>& callback) const;
  PublicBucketId chance_outcome_id(
      const CompactPublicState& child_state) const;
  std::optional<uint32_t> get_or_create_action_child_public_state(
      uint32_t public_state_id,
      int action_index);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      absl::Span<const CardId> cards);
  bool prebuild_public_state_rows(uint32_t root_public_state_id,
                                  int max_depth);
  void rebuild_chance_child_entries();
  bool validate_prebuilt_transitions(
      uint32_t root_public_state_id,
      int max_depth,
      TrainingRunStats& stats) const;
  bool prebuild_info_set_rows(const TrainingRangeView& player_a_range,
                              const TrainingRangeView& player_b_range);
  bool prebuild_private_bucket_rows();
  bool prebuild_frozen_info_set_action_offsets();
  PrivateBucketId private_bucket_for_frozen_row(uint32_t public_state_id,
                                                ComboId combo_id) const;
  uint32_t frozen_info_set_action_offset(uint32_t public_state_id,
                                         int player,
                                         PrivateBucketId private_bucket) const;
  const InfoSetRow* get_or_create_info_set_row(
      InfoSetAddress address,
      absl::Span<const int> action_ids);
  std::optional<InfoSetHandle> make_info_set_handle(
      const InfoSetRow* row,
      size_t expected_action_count) const;
  std::optional<InfoSetHandle> find_info_set_handle(
      InfoSetAddress address,
      size_t expected_action_count) const;
  std::optional<InfoSetHandle> get_or_create_info_set_handle(
      InfoSetAddress address,
      absl::Span<const int> action_ids);
  std::optional<InfoSetHandle> find_frozen_info_set_handle(
      uint32_t public_state_id,
      int player,
      ComboId combo_id,
      size_t expected_action_count) const;
  InfoSetRow append_info_set_actions(absl::Span<const int> action_ids);
  PublicInfoSetSlab& get_or_create_public_info_set_slab(
      uint32_t public_state_id);
  const PublicInfoSetSlab* public_info_set_slab(
      uint32_t public_state_id) const;
  const InfoSetRow* find_info_set_row(InfoSetAddress address) const;
  static const InfoSetRow* find_info_set_row(
      const PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  static int32_t& get_or_create_private_row_slot(
      PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  void average_strategy_probabilities(
      uint32_t public_state_id,
      const PublicStateRow& row,
      int player,
      const PrivateCards& private_cards,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetRow& info_set_row,
      const PublicStateRow& public_state_row,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void condition_ranges_for_actions(
      const TrainingRangeView& range,
      const CompactPublicState& state,
      uint32_t public_state_id,
      int player,
      absl::Span<const int> action_ids,
      std::vector<TrainingRangeView>& conditioned_ranges);
  void fill_regret_matched_strategy_for_row(
      const InfoSetRow& row,
      absl::Span<const int> action_ids,
      double* action_probabilities);
  double utility(const CompactPublicState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double frozen_utility(const PublicStateRow& row,
                        const ExactBoardState& exact_board,
                        const PrivateCards& player_a_cards,
                        const PrivateCards& player_b_cards);
  double uncached_utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(NodeRef node,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards,
                                ChildResolver& children);
  double evaluate_strategy_samples(
      int samples,
      uint32_t root_public_state_id,
      RangeSampler range_sampler);
  void fill_regret_matching(
      std::optional<InfoSetHandle> info_set,
      size_t legal_action_count,
      RegretLoadMode load_mode,
      absl::Span<double> action_probabilities);
  void add_cfr_plus_regret(InfoSetHandle info_set,
                           size_t action_index,
                           float delta,
                           RegretUpdateOptions options);
  void add_average_strategy(InfoSetHandle info_set,
                            absl::Span<const double> action_probabilities,
                            double reach_weight,
                            RegretUpdateMode update_mode);
  void update_strategy(size_t action_offset,
                       const double* action_probabilities,
                       size_t action_count,
                       double reach_prob);
  void fill_regret_matched_strategy(size_t action_offset,
                                    size_t action_count,
                                    bool has_info_set_row,
                                    bool use_atomic_loads,
                                    double* action_probabilities);
  NodeRef sample_frozen_chance_transition(
      NodeRef parent,
      const PublicStateRow& row,
      CardMask known_private_cards);
  void record_action_entry_touches(int64_t count = 1);
  void record_cfr_update(StreetKind street, int depth);
  void record_chance_samples(int64_t count);
  void record_terminal_utility(bool showdown);
  void record_child_node_created();
  void record_betting_history_transition_hit();
  void record_betting_history_transition_miss();
  void record_atomic_regret_update_retries(int64_t count);

  SolverConfig config_;
  CompactPublicState initial_state_;
  std::shared_ptr<GameTree> game_tree_;
  std::mt19937 rng_;
  double cumulative_root_utility_ = 0.0;
  int iterations_run_ = 0;
  int64_t cfr_update_count_ = 0;
  TraversalStats traversal_stats_;
  TrainingRunStats last_training_run_stats_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  CardAbstraction card_abstraction_;
  BettingAbstraction betting_abstraction_;
  bool frozen_ = false;
  bool require_frozen_children_ = false;
  std::shared_ptr<FrozenStrategyTables> mutable_tables_;
  std::shared_ptr<const FrozenStrategyTables> frozen_tables_;
  std::shared_ptr<MutableCumulativeArrays> cumulative_;
};

}  // namespace poker
