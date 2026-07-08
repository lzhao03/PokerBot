#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "src/build_flags.h"
#include "src/card_abstraction.h"
#include "src/card_utils.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/strategy_tables.h"
#include "src/terminal_utility_cache.h"
#include "src/thread_pool.h"
#include "src/training_range.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;
constexpr int kAutoWarmupNoGrowthLimit = 100;

std::optional<double> UtilityBeforeShowdown(const CompactPublicState& state,
                                            uint8_t board_count) {
  const double player_a_contribution = state.player_contribution[0];
  if (state.folded_player == 0) {
    return -player_a_contribution;
  }
  if (state.folded_player == 1) {
    return state.pot - player_a_contribution;
  }
  if (board_count + 2 < 5) {
    return 0.0;
  }
  return std::nullopt;
}

double ShowdownUtilityFromComparison(const CompactPublicState& state,
                                     int comparison) {
  const double player_a_contribution = state.player_contribution[0];
  if (comparison > 0) {
    return state.pot - player_a_contribution;
  }
  if (comparison < 0) {
    return -player_a_contribution;
  }
  return (state.pot / 2.0) - player_a_contribution;
}

size_t ScratchDepthReserve(const SolverConfig& config, int max_depth) {
  if (max_depth > 0) {
    return static_cast<size_t>(max_depth) + 2;
  }
  const int stack_size = std::max(0, config.starting_stack_size);
  return std::max<size_t>(32, static_cast<size_t>(stack_size) + 12);
}

int ChanceSamples(const SolverConfig& config) {
  return std::max(1, config.chance_samples);
}

int WorkerCountForSamples(int samples) {
  int worker_count = static_cast<int>(std::thread::hardware_concurrency());
  if (worker_count <= 0) {
    worker_count = 1;
  }
  return std::min(worker_count, samples);
}

GameState DefaultInitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const int starting_stack = config.starting_stack_size;

  GameState initial_state;
  initial_state.stack[0] = std::max(0, starting_stack - small_blind);
  initial_state.stack[1] = std::max(0, starting_stack - big_blind);
  initial_state.pot = small_blind + big_blind;
  initial_state.folded_player = -1;
  initial_state.street = StreetKind::kPreflop;
  initial_state.all_in = false;
  initial_state.player_to_act = 0;
  initial_state.player_contribution = {small_blind, big_blind};
  return initial_state;
}

}  // namespace

CompactPublicState CompactPublicStateFromGameState(const GameState& state);

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>()) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const GameState& initial_state)
    : CFRSolver(config, std::make_shared<TerminalUtilityCache>(), initial_state) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     std::shared_ptr<TerminalUtilityCache> utility_cache)
    : CFRSolver(config, std::move(utility_cache), DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(
    const SolverConfig& config,
    std::shared_ptr<TerminalUtilityCache> utility_cache,
    GameState initial_state)
  : config_(config),
    initial_state_(CompactPublicStateFromGameState(initial_state)),
    game_tree_(std::make_shared<GameTree>(config)),
    rng_(12345),
    cumulative_root_utility_(0.0),
    utility_cache_(std::move(utility_cache)),
    storage_(),
    strategy_store_(config_, card_abstraction_, storage_, &traversal_stats_),
    public_graph_(config_,
                  storage_,
                  *game_tree_,
                  card_abstraction_,
                  betting_abstraction_,
                  &traversal_stats_) {
  // Pre-allocate strategy table storage when limits are known upfront.
  // This gives fully deterministic peak memory: no reallocation after init.
  if (config_.max_info_sets > 0) {
    // avg ~4 actions per info set (between 2 and kMaxActionsPerNode).
    constexpr int kAvgActionsPerInfoSet = 4;
    const size_t info_set_cap = static_cast<size_t>(config_.max_info_sets);
    const size_t action_cap = info_set_cap * kAvgActionsPerInfoSet;
    storage_.mutable_ref().action_ids.reserve(action_cap);
    arrays().cumulative_regrets.reserve(action_cap);
    arrays().cumulative_strategies.reserve(action_cap);
  }
  if (config_.max_public_states > 0) {
    const size_t public_state_cap =
        static_cast<size_t>(config_.max_public_states);
    storage_.mutable_ref().public_state_ids.reserve(public_state_cap);
    storage_.mutable_ref().public_state_rows.reserve(public_state_cap);
    storage_.mutable_ref().public_chance_child_ids.reserve(public_state_cap);
    storage_.mutable_ref().chance_child_entries.reserve(public_state_cap);
    storage_.mutable_ref().private_bucket_rows.reserve(public_state_cap);
    storage_.mutable_ref().frozen_info_set_action_offsets.reserve(
        public_state_cap);
    storage_.mutable_ref().public_info_set_slabs.reserve(public_state_cap);
    storage_.mutable_ref().betting_history_ids.reserve(public_state_cap);
    storage_.mutable_ref().betting_history_rows.reserve(public_state_cap);
  }
}

CFRSolver::PrivateCards CFRSolver::PrivateCards::FromCombo(
    ComboId combo_id) {
  PrivateCards private_cards;
  private_cards.combo = combo_id;
  return private_cards;
}

CardMask CFRSolver::PrivateCards::mask() const {
  return ComboMask(combo);
}

CompactPublicState
CompactPublicStateFromGameState(const GameState& state) {
  CompactPublicState compact;
  compact.stack = {state.stack[0], state.stack[1]};
  compact.pot = state.pot;
  if (state.board_cards.size() > compact.board_cards.size()) {
    throw std::logic_error("GameState has too many board cards");
  }
  compact.board_count = static_cast<uint8_t>(state.board_cards.size());
  for (size_t i = 0; i < state.board_cards.size(); ++i) {
    compact.board_cards[i] = state.board_cards[i];
  }
  compact.board_mask = state.board_mask;
  if (state.history.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::logic_error("GameState history exceeds compact row capacity");
  }
  if (state.history.size() > CompactPublicState::kMaxHistoryActions) {
    throw std::logic_error("GameState history exceeds compact inline capacity");
  }
  for (const GameAction& action : state.history) {
    AppendHistoryAction(compact, action);
  }
  compact.street = state.street;
  compact.all_in = state.all_in;
  compact.folded_player = state.folded_player;
  compact.player_to_act = state.player_to_act;
  compact.player_contribution = state.player_contribution;
  compact.player_contribution_count = state.player_contribution_count;
  return compact;
}

CFRSolver::ExactBoardState CFRSolver::ExactBoardFromState(
    const CompactPublicState& state) {
  return ExactBoardState{
      state.board_cards,
      state.board_count,
      state.board_mask,
  };
}

void CFRSolver::ApplyExactBoard(CompactPublicState& state,
                                const ExactBoardState& board) {
  state.board_cards = board.cards;
  state.board_count = board.count;
  state.board_mask = board.mask;
}

const CompactPublicState& CFRSolver::NodeCursor::exact_state() const {
  if (!exact_state_.has_value()) {
    CompactPublicState state = row_.state;
    CFRSolver::ApplyExactBoard(state, ref_.exact_board);
    exact_state_.emplace(std::move(state));
  }
  return *exact_state_;
}

std::optional<CFRSolver::NodeCursor> CFRSolver::cursor(NodeRef node) const {
  const auto& public_rows = rows();
  if (node.public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  return NodeCursor{node, public_rows[node.public_state_id]};
}

std::optional<CFRSolver::NodeRef> CFRSolver::root_node_ref(
    uint32_t root_public_state_id) const {
  const auto& public_rows = rows();
  if (root_public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  return NodeRef{root_public_state_id,
                 ExactBoardFromState(public_rows[root_public_state_id].state)};
}

CFRSolver::DecisionFrame CFRSolver::make_decision_frame(
    NodeRef node,
    const PublicStateRow& row) {
  DecisionFrame frame;
  frame.public_state_id = node.public_state_id;
  frame.player = row.player_to_act;
  frame.street = row.state.street;
  frame.action_count = static_cast<uint8_t>(row.action_count);
  std::copy_n(row.action_ids.begin(), row.action_count,
              frame.action_ids.begin());
  return frame;
}

CFRSolver::NodeGraphMode CFRSolver::default_node_graph_mode() const {
  if (!storage_.frozen) {
    return NodeGraphMode::kGrow;
  }
  return require_frozen_children_ ? NodeGraphMode::kRequirePresent
                                  : NodeGraphMode::kSkipMissing;
}

CFRSolver::NodeGraph::NodeGraph(CFRSolver& solver,
                                NodeGraphMode mode)
    : solver_(solver), mode_(mode) {}

CFRSolver::ChildResult
CFRSolver::NodeGraph::make_child_result(
    std::optional<uint32_t> child_id,
    ExactBoardState exact_board,
    const char* missing_message) const {
  const auto& public_rows = solver_.rows();
  ChildStatus status = ChildStatus::kOk;
  if (!child_id.has_value() ||
      *child_id == GameTree::kInvalidPublicStateId) {
    status = ChildStatus::kMissing;
  } else if (*child_id == kCappedPublicStateId) {
    status = ChildStatus::kCapped;
  } else if (*child_id >= public_rows.size()) {
    status = ChildStatus::kInvalid;
  }

  if (status != ChildStatus::kOk) {
    if (mode_ == NodeGraphMode::kRequirePresent) {
      throw std::logic_error(missing_message);
    }
    return ChildResult{status, {}};
  }

  return ChildResult{ChildStatus::kOk, NodeRef{*child_id, exact_board}};
}

CFRSolver::ChildResult
CFRSolver::NodeGraph::action_child(NodeRef parent,
                                   int action_index) {
  std::optional<uint32_t> child_id;
  switch (mode_) {
    case NodeGraphMode::kGrow:
      child_id = solver_.public_graph_.get_or_create_action_child_public_state(
          parent.public_state_id, action_index);
      break;
    case NodeGraphMode::kSkipMissing:
    case NodeGraphMode::kRequirePresent:
      child_id = solver_.public_graph_.action_child_public_state(
          parent.public_state_id, action_index);
      break;
  }

  return make_child_result(child_id, parent.exact_board,
                           "required action child public state is missing");
}

CFRSolver::ChildResult
CFRSolver::NodeGraph::sample_chance_child(
    NodeRef parent,
    CardMask known_private_cards) {
  std::optional<NodeCursor> parent_cursor = solver_.cursor(parent);
  if (!parent_cursor.has_value()) {
    return ChildResult{ChildStatus::kInvalid, {}};
  }
  const CompactPublicState& exact_parent_state = parent_cursor->exact_state();
  const auto cards =
      SampleStreetCards(exact_parent_state, known_private_cards, solver_.rng_);
  CompactPublicState exact_child_state =
      solver_.game_tree_->apply_chance(exact_parent_state, cards);
  const ExactBoardState child_board =
      CFRSolver::ExactBoardFromState(exact_child_state);
  std::optional<uint32_t> child_id;
  switch (mode_) {
    case NodeGraphMode::kGrow:
      child_id = solver_.public_graph_.get_or_create_chance_child_public_state(
          parent.public_state_id, exact_child_state);
      break;
    case NodeGraphMode::kSkipMissing:
    case NodeGraphMode::kRequirePresent:
      child_id = solver_.public_graph_.chance_child_public_state(
          parent.public_state_id, exact_child_state);
      break;
  }

  return make_child_result(child_id, child_board,
                           "required chance child public state is missing");
}

bool CFRSolver::prebuild_info_set_rows(
    const TrainingRangeView& player_a_range,
    const TrainingRangeView& player_b_range) {
  if (storage_.frozen) {
    return true;
  }

  std::vector<uint32_t> seen_buckets;
  uint32_t seen_generation = 1;

  for (uint32_t public_state_id = 0;
       public_state_id < rows().size();
       ++public_state_id) {
    const PublicStateRow& row =
        rows()[public_state_id];
    const int player = row.player_to_act;
    if (row.is_terminal || row.is_chance_node || row.action_count == 0 ||
        !IsPlayer(player)) {
      continue;
    }

    const TrainingRangeView& range =
        player == 0 ? player_a_range : player_b_range;
    if (range.empty()) {
      continue;
    }

    const uint32_t bucket_count =
        card_abstraction_.private_bucket_count(row.state);
    if (bucket_count == 0 || bucket_count > kComboCount) {
      return false;
    }
    if (seen_buckets.size() < bucket_count) {
      seen_buckets.resize(bucket_count, 0);
    }
    if (seen_generation == 0) {
      std::fill(seen_buckets.begin(), seen_buckets.end(), 0);
      seen_generation = 1;
    }
    const uint32_t generation = seen_generation++;
    const absl::Span<const int> action_ids(row.action_ids.data(),
                                           row.action_count);
    const CardMask board_mask = row.state.board_mask;
    for (size_t i = 0; i < range.size(); ++i) {
      if (range.weight(i) <= 0.0f) {
        continue;
      }
      const ComboId combo_id = range.combo(i);
      if ((ComboMask(combo_id) & board_mask) != 0) {
        continue;
      }
      const PrivateBucketId private_bucket =
          card_abstraction_.private_bucket(combo_id, row.state);
      if (private_bucket >= bucket_count) {
        return false;
      }
      if (seen_buckets[private_bucket] == generation) {
        continue;
      }
      seen_buckets[private_bucket] = generation;
      if (!strategy_store_
               .get_or_create({public_state_id, player, private_bucket},
                              action_ids)
               .has_value()) {
        return false;
      }
    }
  }

  return true;
}

void CFRSolver::run(int iterations,
                    const HandRange& player_a_range,
                    const HandRange& player_b_range) {
  last_training_run_stats_ = {};
  if (iterations <= 0) {
    return;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  TrainingRangeView player_a_hands_view(player_a_training_range);
  TrainingRangeView player_b_hands_view(player_b_training_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);

  VLOG(1) << "Preparing compact public-state rows...";
  const std::optional<uint32_t> root_public_state_id =
      public_graph_.get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return;
  }
  VLOG(1) << "Compact root row has "
          << static_cast<int>(
                 rows()[*root_public_state_id].action_count)
          << " legal actions";

  const int num_threads =
      config_.num_training_threads <= 1 ? 1 : config_.num_training_threads;
  const int max_depth = config_.max_depth;
  const bool can_use_frozen_regret_only =
      config_.regret_only_training && max_depth == 0;

  const bool should_run_frozen_phase = prepare_frozen_training(
      *root_public_state_id, num_threads, max_depth, can_use_frozen_regret_only,
      player_a_hands_view, player_b_hands_view);
  const CompactPublicState root_state =
      rows()[*root_public_state_id].state;
  const int completed_warmup = run_warmup_phase(
      iterations, *root_public_state_id, root_state, range_sampler,
      player_a_hands_view, player_b_hands_view, max_depth,
      should_run_frozen_phase, can_use_frozen_regret_only);
  maybe_run_frozen_phase(iterations, completed_warmup, num_threads,
                         *root_public_state_id, range_sampler,
                         player_a_training_range, player_b_training_range);

  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_;
  LOG(INFO) << "Information sets: " << get_info_set_count();
  LOG(INFO) << "Public states: " << get_public_state_count();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

bool CFRSolver::prepare_frozen_training(
    uint32_t root_public_state_id,
    int num_threads,
    int max_depth,
    bool can_use_frozen_regret_only,
    const TrainingRangeView& player_a_hands_view,
    const TrainingRangeView& player_b_hands_view) {
  TrainingRunStats& stats = last_training_run_stats_;
  const bool should_prebuild_public_states =
      !storage_.frozen && (num_threads > 1 || can_use_frozen_regret_only) &&
      (config_.max_public_states > 0 || max_depth > 0);
  if (!should_prebuild_public_states) {
    return false;
  }

  auto record_public_counts = [&] {
    stats.prebuild_public_states =
        static_cast<int64_t>(get_public_state_count());
    stats.prebuild_betting_histories =
        static_cast<int64_t>(tables().betting_history_rows.size());
  };
  auto record_action_counts = [&] {
    stats.prebuild_info_sets = static_cast<int64_t>(get_info_set_count());
    stats.prebuild_action_entries =
        static_cast<int64_t>(
            arrays().cumulative_regrets.size());
  };

  VLOG(1) << "Prebuilding compact public-state rows...";
  const auto prebuild_start = std::chrono::steady_clock::now();
  stats.public_state_prebuild_complete =
      public_graph_.prebuild_public_state_rows(root_public_state_id,
                                               max_depth);
  const auto prebuild_end = std::chrono::steady_clock::now();
  stats.prebuild_seconds =
      std::chrono::duration<double>(prebuild_end - prebuild_start).count();
  record_public_counts();
  if (!stats.public_state_prebuild_complete) {
    return false;
  }

  if (!public_graph_.validate_prebuilt_transitions(root_public_state_id,
                                                   max_depth, stats)) {
    return false;
  }
  record_action_counts();

  VLOG(1) << "Prebuilding infoset rows...";
  const auto info_set_prebuild_start = std::chrono::steady_clock::now();
  stats.info_set_prebuild_complete =
      prebuild_info_set_rows(player_a_hands_view, player_b_hands_view);
  const auto info_set_prebuild_end = std::chrono::steady_clock::now();
  stats.info_set_prebuild_seconds =
      std::chrono::duration<double>(info_set_prebuild_end -
                                    info_set_prebuild_start)
          .count();
  record_action_counts();
  if (!stats.info_set_prebuild_complete) {
    return false;
  }

  stats.private_bucket_prebuild_complete =
      strategy_store_.prebuild_private_bucket_rows();
  if (!stats.private_bucket_prebuild_complete) {
    return false;
  }
  stats.prebuild_private_bucket_rows =
      static_cast<int64_t>(tables().private_bucket_rows.size());

  stats.frozen_info_set_lookup_prebuild_complete =
      strategy_store_.prebuild_frozen_info_set_action_offsets();
  if (!stats.frozen_info_set_lookup_prebuild_complete) {
    return false;
  }
  stats.prebuild_frozen_info_set_lookup_rows =
      static_cast<int64_t>(
          tables().frozen_info_set_action_offsets.size());
  return true;
}

int CFRSolver::run_warmup_phase(
    int iterations,
    uint32_t root_public_state_id,
    const CompactPublicState& root_state,
    RangeSampler& range_sampler,
    const TrainingRangeView& player_a_hands_view,
    const TrainingRangeView& player_b_hands_view,
    int max_depth,
    bool should_run_frozen_phase,
    bool can_use_frozen_regret_only) {
  const bool auto_warmup =
      should_run_frozen_phase && !storage_.frozen && config_.warmup_iterations <= 0;
  int warmup_count = iterations;
  if (should_run_frozen_phase && !storage_.frozen &&
      config_.warmup_iterations > 0) {
    const int requested_warmup =
        can_use_frozen_regret_only
            ? config_.warmup_iterations
            : std::max(config_.warmup_iterations, kPlayerCount);
    warmup_count = std::min(requested_warmup, iterations);
  }

  LOG(INFO) << "Starting CFR iterations...";
  VLOG(1) << "Warmup phase: "
          << (auto_warmup ? "adaptive" : std::to_string(warmup_count))
          << " single-threaded iterations";
  NodeGraph graph(*this, NodeGraphMode::kGrow);
  const NodeRef root_node{root_public_state_id,
                          ExactBoardFromState(root_state)};
  TraversalScratch scratch(ScratchDepthReserve(config_, max_depth));
  const int64_t warmup_start_updates = cfr_update_count_;
  const auto warmup_start = std::chrono::steady_clock::now();
  int completed_warmup = 0;
  int no_growth_iterations = 0;
  size_t previous_info_sets = get_info_set_count();
  size_t previous_public_states = get_public_state_count();

  for (int i = 0; i < warmup_count; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    const TraversalDeal traversal_deal{{
        PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo),
    }};

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    int cfr_iteration = iterations_run_;
    OptionalTrainingRange player_a_context_range;
    OptionalTrainingRange player_b_context_range;
    if (max_depth > 0) {
      player_a_context_range = std::cref(player_a_hands_view);
      player_b_context_range = std::cref(player_b_hands_view);
    }
    TraversalOptions options;
    options.update_player = cfr_iteration % kPlayerCount;
    options.iteration = cfr_iteration;
    options.max_depth = max_depth;
    options.write_average_strategy = !config_.regret_only_training;
    TraversalContext ctx(traversal_deal, options, scratch,
                         player_a_context_range, player_b_context_range);
    double dealt_value = cfr_with_ranges(root_node, ctx, graph);

    cumulative_root_utility_ += dealt_value;
    ++iterations_run_;
    ++completed_warmup;

    if (auto_warmup) {
      const size_t current_info_sets = get_info_set_count();
      const size_t current_public_states = get_public_state_count();
      if (current_info_sets == previous_info_sets &&
          current_public_states == previous_public_states) {
        ++no_growth_iterations;
      } else {
        no_growth_iterations = 0;
        previous_info_sets = current_info_sets;
        previous_public_states = current_public_states;
      }
      const bool info_set_cap_hit =
          config_.max_info_sets > 0 &&
          current_info_sets >= static_cast<size_t>(config_.max_info_sets);
      const bool public_state_cap_hit =
          config_.max_public_states > 0 &&
          current_public_states >=
              static_cast<size_t>(config_.max_public_states);
      // Cap hits mean the tree is incomplete; do not freeze and turn the
      // frozen phase into mostly skipped missing-child branches.
      if (!info_set_cap_hit && !public_state_cap_hit &&
          no_growth_iterations >= kAutoWarmupNoGrowthLimit) {
        break;
      }
    }
  }

  const auto warmup_end = std::chrono::steady_clock::now();
  last_training_run_stats_.warmup_iterations = completed_warmup;
  last_training_run_stats_.warmup_seconds =
      std::chrono::duration<double>(warmup_end - warmup_start).count();
  last_training_run_stats_.warmup_cfr_updates =
      cfr_update_count_ - warmup_start_updates;
  return completed_warmup;
}

void CFRSolver::maybe_run_frozen_phase(
    int iterations,
    int completed_warmup,
    int num_threads,
    uint32_t root_public_state_id,
    const RangeSampler& range_sampler,
    const TrainingRange& player_a_training_range,
    const TrainingRange& player_b_training_range) {
  const int remaining = iterations - completed_warmup;
  if (remaining <= 0) {
    return;
  }

  storage_.freeze();
  require_frozen_children_ = true;
  LOG(INFO) << "Frozen after warmup: " << get_info_set_count()
            << " info sets, " << iterations_run_
            << " warmup iterations. Starting frozen phase ("
            << remaining << " iterations, " << num_threads << " workers)...";
  const int64_t frozen_start_updates = cfr_update_count_;
  const auto frozen_start = std::chrono::steady_clock::now();
  run_frozen_iterations(remaining, num_threads, root_public_state_id,
                        range_sampler, player_a_training_range,
                        player_b_training_range);
  const auto frozen_end = std::chrono::steady_clock::now();
  last_training_run_stats_.frozen_iterations = remaining;
  last_training_run_stats_.frozen_seconds =
      std::chrono::duration<double>(frozen_end - frozen_start).count();
  last_training_run_stats_.frozen_cfr_updates =
      cfr_update_count_ - frozen_start_updates;
}

template <typename WorkerFn, typename AccumulateFn>
void CFRSolver::run_sharded(int work_count,
                            int worker_count,
                            int first_index,
                            WorkerFn&& worker_fn,
                            AccumulateFn&& accumulate_fn) {
  if (work_count <= 0 || worker_count <= 0) {
    return;
  }
  const int shard_count = std::min(work_count, worker_count);
  ThreadPoolExecutor executor(shard_count);
  std::uniform_int_distribution<unsigned int> seed_dist;
  using WorkerResult = decltype(worker_fn(0, 0, 0u));
  std::vector<std::future<WorkerResult>> futures;
  futures.reserve(shard_count);

  int work_remaining = work_count;
  int next_index = first_index;
  for (int worker_index = 0; worker_index < shard_count; ++worker_index) {
    const int shard = work_remaining / (shard_count - worker_index);
    work_remaining -= shard;
    const int begin = next_index;
    next_index += shard;
    const unsigned int seed = seed_dist(rng_);
    auto worker = worker_fn;
    futures.push_back(executor.submit([worker, begin, shard, seed]() mutable {
      return worker(begin, shard, seed);
    }));
  }

  for (std::future<WorkerResult>& future : futures) {
    accumulate_fn(future.get());
  }
}

void CFRSolver::run_frozen_iterations(
    int iterations,
    int num_threads,
    uint32_t root_public_state_id,
    const RangeSampler& range_sampler,
    const TrainingRange& player_a_training_range,
    const TrainingRange& player_b_training_range) {
  // Each worker shares the frozen strategy tables and writes to the same
  // regret/strategy arrays.
  // Workers use their own RNG and TraversalScratch; no locks needed.
  std::shared_ptr<const StrategyTables> frozen_tables =
      storage_.frozen_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative = storage_.cumulative;
  const bool use_frozen_regret_only =
      storage_.frozen && require_frozen_children_ &&
      config_.regret_only_training && config_.max_depth == 0;
  const bool use_atomic_updates = num_threads > 1;

  struct WorkerResult {
    double utility = 0.0;
    TraversalStats traversal_stats;
    int64_t cfr_updates = 0;
    int iterations = 0;
  };

  int completed_iterations = 0;
  run_sharded(
      iterations, num_threads, iterations_run_,
      [this, root_public_state_id, &range_sampler, frozen_tables, cumulative,
       use_frozen_regret_only, use_atomic_updates, &player_a_training_range,
       &player_b_training_range](int iteration_begin,
                                 int shard,
                                 unsigned int seed) mutable {
          // Build a lightweight worker that shares frozen tables.
          CFRSolver worker(config_, utility_cache_);
          worker.storage_.bind_frozen(frozen_tables, cumulative);
          worker.require_frozen_children_ = true;
          worker.rng_.seed(seed);

          TrainingRangeView player_a_hands_view;
          TrainingRangeView player_b_hands_view;
          const int max_depth = config_.max_depth;
          const size_t scratch_depth =
              use_frozen_regret_only ? 0
                                     : ScratchDepthReserve(config_, max_depth);
          TraversalScratch scratch(scratch_depth);
          if (!use_frozen_regret_only) {
            // Per-worker range views (read-only, built from shared training data).
            player_a_hands_view.reset_to_all(player_a_training_range);
            player_b_hands_view.reset_to_all(player_b_training_range);
          }

          double local_utility = 0.0;
          const CompactPublicState root_state =
              worker.tables()
                  .public_state_rows[root_public_state_id]
                  .state;
          NodeGraph graph(worker, NodeGraphMode::kRequirePresent);
          const NodeRef root_node{root_public_state_id,
                                  ExactBoardFromState(root_state)};
          for (int i = 0; i < shard; ++i) {
            const RangeDeal deal = range_sampler.sample(worker.rng_);
            const TraversalDeal traversal_deal{{
                PrivateCards::FromCombo(deal.player_a_combo),
                PrivateCards::FromCombo(deal.player_b_combo),
            }};

            const int cfr_iteration = iteration_begin + i;
            OptionalTrainingRange player_a_context_range;
            OptionalTrainingRange player_b_context_range;
            if (max_depth > 0) {
              player_a_context_range = std::cref(player_a_hands_view);
              player_b_context_range = std::cref(player_b_hands_view);
            }
            TraversalOptions options;
            options.update_player = cfr_iteration % kPlayerCount;
            options.iteration = cfr_iteration;
            options.max_depth = max_depth;
            options.write_average_strategy = !config_.regret_only_training;
            if (use_frozen_regret_only) {
              options.regret_load_mode =
                  use_atomic_updates ? RegretLoadMode::kAtomic
                                     : RegretLoadMode::kPlain;
              options.regret_update_mode =
                  use_atomic_updates ? RegretUpdateMode::kAtomic
                                     : RegretUpdateMode::kPlain;
              options.write_average_strategy = false;
              options.record_atomic_retry_stats = use_atomic_updates;
            }
            TraversalContext ctx(traversal_deal, options, scratch,
                                 player_a_context_range,
                                 player_b_context_range);
            if (use_frozen_regret_only) {
              local_utility +=
                  worker.cfr_frozen_regret_only(root_node, ctx, graph);
              continue;
            }

            local_utility += worker.cfr_with_ranges(root_node, ctx, graph);
          }
          return WorkerResult{local_utility, worker.get_traversal_stats(),
                              worker.get_cfr_update_count(), shard};
      },
      [&](const WorkerResult& result) {
        cumulative_root_utility_ += result.utility;
        traversal_stats_.add(result.traversal_stats);
        cfr_update_count_ += result.cfr_updates;
        completed_iterations += result.iterations;
      });
  iterations_run_ += completed_iterations;
}

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::CfrTraversal<mode>::value(NodeRef node) {
  const uint32_t public_state_id = node.public_state_id;
  const auto& public_state_rows = solver_.rows();
  if (public_state_id >= public_state_rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = public_state_rows[public_state_id];

  if (row.is_terminal) {
    return terminal(node, row);
  }

  if (row.is_chance_node) {
    return chance(node);
  }

  if constexpr (mode == CfrTraversalMode::kNormal) {
    if (ctx_.depth_limited()) {
      std::optional<NodeCursor> node_cursor = solver_.cursor(node);
      if (!node_cursor.has_value()) {
        return 0.0;
      }
      return depth_limit_value(*node_cursor);
    }
  }

  return decision(node, row);
}

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::CfrTraversal<mode>::terminal(
    NodeRef node,
    const PublicStateRow& row) {
  if constexpr (mode == CfrTraversalMode::kNormal) {
    std::optional<NodeCursor> node_cursor = solver_.cursor(node);
    if (!node_cursor.has_value()) {
      return 0.0;
    }
    const CompactPublicState& state = node_cursor->exact_state();
    solver_.traversal_stats_.record_terminal(state.folded_player < 0);
    if (!ctx_.use_terminal_cache() || ctx_.max_depth() > 0) {
      return solver_.uncached_utility(state, ctx_.cards(0), ctx_.cards(1));
    }
    return solver_.utility(state, ctx_.cards(0), ctx_.cards(1));
  } else {
    solver_.traversal_stats_.record_terminal(row.state.folded_player < 0);
    return solver_.frozen_utility(row, node.exact_board, ctx_.cards(0),
                                  ctx_.cards(1));
  }
}

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::CfrTraversal<mode>::chance(NodeRef node) {
  const int samples = ChanceSamples(solver_.config_);
  solver_.traversal_stats_.record_chance_samples(samples);
  if constexpr (mode == CfrTraversalMode::kFrozenRegretOnly) {
    return solver_.sample_chance_children(
        samples, node, ctx_.known_private_cards(), graph_,
        [&](NodeRef child) {
          return value(child);
        });
  }

  TrainingRangeView* public_player_a_range = nullptr;
  TrainingRangeView* public_player_b_range = nullptr;
  if (ctx_.range(0).has_value() || ctx_.range(1).has_value()) {
    RangeScratchFrame& scratch_frame = ctx_.scratch_frame();
    public_player_a_range = &scratch_frame.public_player_a_range;
    public_player_b_range = &scratch_frame.public_player_b_range;
  }

  return solver_.sample_chance_children(
      samples, node, ctx_.known_private_cards(), graph_,
      [&](NodeRef child) {
        OptionalTrainingRange child_player_a_range = ctx_.range(0);
        OptionalTrainingRange child_player_b_range = ctx_.range(1);
        if (child_player_a_range.has_value()) {
          child_player_a_range =
              std::cref(child_player_a_range->get().without_mask(
                  child.exact_board.mask, *public_player_a_range));
        }
        if (child_player_b_range.has_value()) {
          child_player_b_range =
              std::cref(child_player_b_range->get().without_mask(
                  child.exact_board.mask, *public_player_b_range));
        }

        if (child_player_a_range.has_value() ||
            child_player_b_range.has_value()) {
          auto range_scope =
              ctx_.set_ranges(child_player_a_range, child_player_b_range);
          return value(child);
        }
        return value(child);
      });
}

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::CfrTraversal<mode>::depth_limit_value(
    const NodeCursor& node_cursor) {
  const CompactPublicState& state = node_cursor.exact_state();
  return solver_.game_tree_->is_betting_round_over(state)
             ? solver_.uncached_utility(state, ctx_.cards(0), ctx_.cards(1))
             : 0.0;
}

template <CFRSolver::CfrTraversalMode mode>
double CFRSolver::CfrTraversal<mode>::decision(
    NodeRef node,
    const PublicStateRow& row) {
  const DecisionFrame decision = solver_.make_decision_frame(node, row);
  const int player = decision.player;
  if (!IsPlayer(player) || decision.action_count == 0) {
    return 0.0;
  }
  const PrivateCards& player_cards = ctx_.cards(player);

  const bool is_update_player = ctx_.is_update_player(player);
  const size_t action_count = decision.action_count;
  const absl::Span<const int> legal_action_ids = decision.action_ids_span();
  std::optional<ActionBlock> action_block;
  if constexpr (mode == CfrTraversalMode::kNormal) {
    const InfoSetAddress info_set_address{
        decision.public_state_id, player,
        solver_.card_abstraction_.private_bucket(player_cards.combo,
                                                 row.state)};
    action_block =
        is_update_player
            ? solver_.strategy_store_.get_or_create(info_set_address,
                                                    legal_action_ids)
            : solver_.strategy_store_.find(info_set_address, action_count);
  } else {
    action_block = solver_.strategy_store_.find_frozen(
        decision.public_state_id, player, player_cards.combo, action_count);
  }

  ActionScratch action_scratch;
  absl::Span<double> action_probabilities =
      action_scratch.probs(action_count);
  absl::Span<double> action_values = action_scratch.vals(action_count);
  solver_.strategy_store_.regret_matching_or_uniform(
      action_block, action_count, ctx_.options().regret_load_mode,
      action_probabilities);

  double node_value = 0.0;
  std::optional<ActionRangeConditioning> range_conditioning;
  if constexpr (mode == CfrTraversalMode::kNormal) {
    const bool needs_range_conditioning =
        (player == 0 && ctx_.range(0).has_value()) ||
        (player == 1 && ctx_.range(1).has_value());
    if (needs_range_conditioning) {
      std::optional<NodeCursor> node_cursor = solver_.cursor(node);
      if (!node_cursor.has_value()) {
        return 0.0;
      }
      range_conditioning.emplace(
          solver_, ctx_, *node_cursor, decision.public_state_id, player,
          legal_action_ids);
    }
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    const ChildResult child =
        graph_.action_child(node, static_cast<int>(action_index));
    if (child.status != ChildStatus::kOk) {
      continue;
    }

    double action_value = 0.0;
    {
      auto reach_scope =
          ctx_.enter_action(player, action_probabilities[action_index]);
      auto depth_scope = ctx_.descend();
      if constexpr (mode == CfrTraversalMode::kNormal) {
        if (range_conditioning.has_value() &&
            range_conditioning->enabled()) {
          auto range_scope =
              ctx_.set_ranges(
                  range_conditioning->player_a_range_for(action_index),
                  range_conditioning->player_b_range_for(action_index));
          action_value = value(child.node);
        } else {
          action_value = value(child.node);
        }
      } else {
        action_value = value(child.node);
      }
    }
    action_values[action_index] = action_value;
    node_value += action_probabilities[action_index] * action_value;
  }

  ++solver_.cfr_update_count_;
  solver_.traversal_stats_.record_decision(decision.street, ctx_.depth());

  if (action_block.has_value() && is_update_player) {
    const double opponent_reach_prob = ctx_.opponent_reach(player);
    const RegretUpdateOptions regret_update_options =
        ctx_.regret_update_options();
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double utility_sign = player == 0 ? 1.0 : -1.0;
      const double regret =
          opponent_reach_prob * utility_sign *
          (action_values[action_index] - node_value);

      action_block->add_cfr_plus_regret(
          action_index, static_cast<float>(regret), regret_update_options);
    }

    if constexpr (mode == CfrTraversalMode::kNormal) {
      if (ctx_.options().write_average_strategy) {
        action_block->add_average_strategy(
            action_probabilities, ctx_.average_strategy_weight(player),
            ctx_.options().regret_update_mode);
      }
    }
  }

  return node_value;
}

double CFRSolver::cfr_with_ranges(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  return CfrTraversal<CfrTraversalMode::kNormal>(*this, ctx, graph)
      .value(node);
}

double CFRSolver::cfr_frozen_regret_only(
    NodeRef node,
    TraversalContext& ctx,
    NodeGraph& graph) {
  return CfrTraversal<CfrTraversalMode::kFrozenRegretOnly>(*this, ctx, graph)
      .value(node);
}

template <typename EvalChild>
double CFRSolver::sample_chance_children(
    int samples,
    NodeRef node,
    CardMask known_private_cards,
    NodeGraph& graph,
    EvalChild&& eval_child) {
  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const ChildResult child =
        graph.sample_chance_child(node, known_private_cards);
    if (child.status != ChildStatus::kOk) {
      continue;
    }
    value += eval_child(child.node);
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

CFRSolver::ActionRangeConditioning::ActionRangeConditioning(
    CFRSolver& solver,
    TraversalContext& ctx,
    const NodeCursor& node_cursor,
    uint32_t public_state_id,
    int player,
    absl::Span<const int> legal_action_ids)
    : original_player_a_range_(ctx.range(0)),
      original_player_b_range_(ctx.range(1)),
      condition_player_a_(player == 0 &&
                          original_player_a_range_.has_value()),
      condition_player_b_(player == 1 &&
                          original_player_b_range_.has_value()) {
  if (!enabled()) {
    return;
  }

  RangeScratchFrame& scratch_frame = ctx.scratch_frame();
  const CompactPublicState& state = node_cursor.exact_state();
  if (condition_player_a_) {
    conditioned_ranges_ = solver.condition_ranges_for_actions(
        original_player_a_range_->get(), state, public_state_id, player,
        legal_action_ids, scratch_frame);
  } else {
    conditioned_ranges_ = solver.condition_ranges_for_actions(
        original_player_b_range_->get(), state, public_state_id, player,
        legal_action_ids, scratch_frame);
  }
}

CFRSolver::OptionalTrainingRange
CFRSolver::ActionRangeConditioning::player_a_range_for(
    size_t action_index) const {
  if (condition_player_a_) {
    return std::cref(conditioned_ranges_.for_action(action_index));
  }
  return original_player_a_range_;
}

CFRSolver::OptionalTrainingRange
CFRSolver::ActionRangeConditioning::player_b_range_for(
    size_t action_index) const {
  if (condition_player_b_) {
    return std::cref(conditioned_ranges_.for_action(action_index));
  }
  return original_player_b_range_;
}

CFRSolver::ActionConditionedRanges CFRSolver::condition_ranges_for_actions(
    const TrainingRangeView& range,
    const CompactPublicState& state,
    uint32_t public_state_id,
    int player,
    absl::Span<const int> conditioned_action_ids,
    RangeScratchFrame& scratch_frame) {
  const size_t action_count = conditioned_action_ids.size();
  if (action_count == 0) {
    return ActionConditionedRanges();
  }

  std::vector<TrainingRangeView>& conditioned_ranges =
      scratch_frame.conditioned_ranges;
  if (conditioned_ranges.size() < action_count) {
    conditioned_ranges.resize(action_count);
  }
  for (size_t i = 0; i < action_count; ++i) {
    conditioned_ranges[i].reset_to_filtered();
  }
  const size_t range_size = range.size();
  if (range_size == 0) {
    return ActionConditionedRanges(
        absl::Span<TrainingRangeView>(conditioned_ranges.data(),
                                      action_count));
  }

  const CardMask board_mask = state.board_mask;
  ActionScratch action_scratch;
  absl::Span<double> action_probabilities =
      action_scratch.probs(action_count);
  for (size_t i = 0; i < range_size; ++i) {
    const float range_weight = range.weight(i);
    const ComboId combo_id = range.combo(i);
    if (range_weight <= 0.0 || (ComboMask(combo_id) & board_mask) != 0) {
      continue;
    }

    const PrivateBucketId private_bucket =
        card_abstraction_.private_bucket(combo_id, state);
    strategy_store_.regret_matching_for_bucket(
        public_state_id, player, private_bucket, conditioned_action_ids,
        action_probabilities);

    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double conditioned_weight =
          range_weight * action_probabilities[action_index];
      if (conditioned_weight > 0.0) {
        conditioned_ranges[action_index].add(
            combo_id, static_cast<float>(conditioned_weight));
      }
    }
  }

  return ActionConditionedRanges(
      absl::Span<TrainingRangeView>(conditioned_ranges.data(), action_count));
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  const std::optional<uint32_t> root_public_state_id =
      public_graph_.get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }
  const CompactPublicState root_state =
      rows()[*root_public_state_id].state;
  const NodeRef root_node{*root_public_state_id,
                          ExactBoardFromState(root_state)};
  NodeGraph graph(*this, default_node_graph_mode());
  EvaluationContext ctx{TraversalDeal{{
      PrivateCards::FromCombo(player_a_hand),
      PrivateCards::FromCombo(player_b_hand),
  }}};
  return evaluate_strategy_node(root_node, ctx, graph);
}

double CFRSolver::evaluate_strategy(int samples, const HandRange& player_a_range,
                                    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  const TrainingRange player_a_training_range =
      BuildTrainingRange(player_a_range);
  const TrainingRange player_b_training_range =
      BuildTrainingRange(player_b_range);
  RangeSampler range_sampler(player_a_training_range, player_b_training_range);
  const std::optional<uint32_t> root_public_state_id =
      public_graph_.get_or_create_public_state_row(initial_state_);
  if (!root_public_state_id.has_value()) {
    return 0.0;
  }

  if (samples < kParallelEvaluationSampleThreshold) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  int worker_count = WorkerCountForSamples(samples);
  if (worker_count <= 1) {
    return evaluate_strategy_samples(samples, *root_public_state_id,
                                     range_sampler);
  }

  SolverConfig config = config_;
  std::shared_ptr<const StrategyTables> frozen_tables =
      storage_.frozen_tables;
  std::shared_ptr<MutableCumulativeArrays> cumulative = storage_.cumulative;
  double total = 0.0;
  run_sharded(
      samples, worker_count, 0,
      [config, range_sampler, root_public_state_id = *root_public_state_id,
       frozen_tables, cumulative](int, int shard_samples,
                                  unsigned int seed) mutable {
        CFRSolver worker(config, std::make_shared<TerminalUtilityCache>());
        worker.storage_.bind_frozen(frozen_tables, cumulative);
        worker.rng_.seed(seed);
        const double value = worker.evaluate_strategy_samples(
            shard_samples, root_public_state_id, range_sampler);
        return std::make_pair(
            value * shard_samples,
            worker.get_traversal_stats().action_entry_touches);
      },
      [&](const std::pair<double, int64_t>& result) {
        total += result.first;
        traversal_stats_.record_action_entries(result.second);
      });
  return total / samples;
}

double CFRSolver::evaluate_strategy_samples(
    int samples,
    uint32_t root_public_state_id,
    RangeSampler range_sampler) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  std::optional<NodeRef> root_node = root_node_ref(root_public_state_id);
  if (!root_node.has_value()) {
    return 0.0;
  }
  NodeGraph graph(*this, default_node_graph_mode());
  for (int i = 0; i < samples; ++i) {
    const RangeDeal deal = range_sampler.sample(rng_);
    EvaluationContext ctx{TraversalDeal{{
        PrivateCards::FromCombo(deal.player_a_combo),
        PrivateCards::FromCombo(deal.player_b_combo),
    }}};
    total += evaluate_strategy_node(*root_node, ctx, graph);
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    NodeRef node,
    EvaluationContext& ctx,
    NodeGraph& graph) {
  const std::optional<NodeCursor> node_cursor = cursor(node);
  if (!node_cursor.has_value()) {
    return 0.0;
  }
  const uint32_t public_state_id = node.public_state_id;
  const PublicStateRow& row = node_cursor->row();

  if (row.is_terminal) {
    const CompactPublicState& state = node_cursor->exact_state();
    return utility(state, ctx.cards(0), ctx.cards(1));
  }
  if (row.is_chance_node) {
    const int samples = ChanceSamples(config_);
    return sample_chance_children(
        samples, node, ctx.known_private_cards(), graph,
        [&](NodeRef child) {
          return evaluate_strategy_node(child, ctx, graph);
        });
  }
  if (row.action_count == 0) {
    return 0.0;
  }

  const int player = row.player_to_act;
  if (!IsPlayer(player)) {
    return 0.0;
  }

  const PrivateCards& player_cards = ctx.cards(player);
  ActionScratch action_scratch;
  absl::Span<double> probabilities =
      action_scratch.probs(row.action_count);
  const PrivateBucketId private_bucket =
      card_abstraction_.private_bucket(player_cards.combo, row.state);
  strategy_store_.average_strategy(
      public_state_id, row, player, private_bucket,
      config_.regret_only_training, probabilities);

  double value = 0.0;
  const int action_count = row.action_count;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    const ChildResult child =
        graph.action_child(node, action_index);
    if (child.status != ChildStatus::kOk) {
      continue;
    }
    value += probabilities[action_index] *
             evaluate_strategy_node(child.node, ctx, graph);
  }
  return value;
}

double CFRSolver::get_expected_value(int player_id) const {
  const int iters = iterations_run_;
  if (iters == 0) {
    return 0.0;
  }
  double player_a_ev = cumulative_root_utility_ / iters;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

CFRSolver::UtilityCacheStats CFRSolver::get_utility_cache_stats() const {
  TerminalUtilityCache::Stats stats = utility_cache_->stats();
  return {stats.hits, stats.misses, stats.entries};
}

bool CFRSolver::traversal_stats_enabled() {
  return kTraversalStatsEnabled;
}

double CFRSolver::utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards) {
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, state.board_count)) {
    return *utility;
  }

  return utility_cache_->get_or_compute(
      state, player_a_cards.combo, player_b_cards.combo, [&]() {
        return game_tree_->get_utility(
            state, player_a_cards.combo, player_b_cards.combo);
      });
}

double CFRSolver::frozen_utility(const PublicStateRow& row,
                                 const ExactBoardState& exact_board,
                                 const PrivateCards& player_a_cards,
                                 const PrivateCards& player_b_cards) {
  const CompactPublicState& state = row.state;
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, exact_board.count)) {
    return *utility;
  }

  // Frozen sampled training sees mostly one-off showdowns; direct evaluation
  // is faster than paying shared cache lookup/mutation overhead.
  HandEvaluator evaluator;
  const int comparison =
      evaluator.compare_hands(player_a_cards.combo, player_b_cards.combo,
                              exact_board.cards, exact_board.count);
  return ShowdownUtilityFromComparison(state, comparison);
}

double CFRSolver::uncached_utility(const CompactPublicState& state,
                                   const PrivateCards& player_a_cards,
                                   const PrivateCards& player_b_cards) {
  if (const std::optional<double> utility =
          UtilityBeforeShowdown(state, state.board_count)) {
    return *utility;
  }

  return game_tree_->get_utility(
      state, player_a_cards.combo, player_b_cards.combo);
}

} // namespace poker
