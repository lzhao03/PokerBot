#include "src/cfr_solver.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "src/build_flags.h"
#include "src/card_abstraction.h"
#include "src/card_utils.h"
#include "src/game_tree.h"
#include "src/hand_evaluator.h"
#include "src/hand_range.h"
#include "src/strategy_tables.h"
#include "src/thread_pool.h"
#include "src/training_range.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace poker {

namespace {

constexpr int kParallelEvaluationSampleThreshold = 32;

SolverConfig NormalizedSolverConfig(SolverConfig config) {
  config.num_training_threads = std::max(1, config.num_training_threads);
  return config;
}

std::optional<double> UtilityBeforeShowdown(const BettingState& state,
                                            uint8_t board_count) {
  const double player_a_committed = state.committed[0];
  if (state.folded_player == 0) {
    return -player_a_committed;
  }
  if (state.folded_player == 1) {
    return Pot(state) - player_a_committed;
  }
  if (board_count + 2 < 5) {
    return 0.0;
  }
  return std::nullopt;
}

double ShowdownUtilityFromComparison(const BettingState& state,
                                     int comparison) {
  const double player_a_committed = state.committed[0];
  if (comparison > 0) {
    return Pot(state) - player_a_committed;
  }
  if (comparison < 0) {
    return -player_a_committed;
  }
  return (Pot(state) / 2.0) - player_a_committed;
}

size_t ScratchDepthReserve(const SolverConfig& config, int max_depth) {
  if (max_depth > 0) {
    return static_cast<size_t>(max_depth) + kMaxBoardCards + 4;
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

ExactGameState DefaultInitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const int starting_stack = config.starting_stack_size;

  BettingState betting;
  betting.stack[0] = std::max(0, starting_stack - small_blind);
  betting.stack[1] = std::max(0, starting_stack - big_blind);
  betting.folded_player = -1;
  betting.street = StreetKind::kPreflop;
  betting.player_to_act = 0;
  betting.committed = {small_blind, big_blind};
  return ExactGameState{betting, Board{}};
}

}  // namespace

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, DefaultInitialState(config)) {
}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const ExactGameState& initial_state)
    : config_(NormalizedSolverConfig(config)),
      initial_state_(initial_state),
      rng_(12345),
      cumulative_root_utility_(0.0),
      betting_abstraction_(config_),
      storage_(),
      strategy_store_(config_, storage_, &traversal_stats_),
      graph_builder_(config_, storage_, betting_abstraction_,
                     traversal_stats_) {
  ValidateBettingState(initial_state_.betting);
  // Pre-allocate strategy table storage when limits are known upfront.
  // This gives fully deterministic peak memory: no reallocation after init.
  if (config_.max_info_sets > 0) {
    // avg ~4 actions per info set (between 2 and kMaxActionsPerNode).
    constexpr int kAvgActionsPerInfoSet = 4;
    const size_t info_set_cap = static_cast<size_t>(config_.max_info_sets);
    const size_t action_cap = info_set_cap * kAvgActionsPerInfoSet;
    arrays().cumulative_regrets.reserve(action_cap);
    arrays().cumulative_strategies.reserve(action_cap);
  }
  if (config_.max_public_states > 0) {
    const auto public_state_cap = static_cast<size_t>(config_.max_public_states);
    storage_.mutable_ref().public_state_ids.reserve(public_state_cap);
    storage_.mutable_ref().public_state_rows.reserve(public_state_cap);
    storage_.mutable_ref().public_chance_child_ids.reserve(public_state_cap);
    storage_.mutable_ref().action_child_ids.reserve(public_state_cap * 4);
    storage_.mutable_ref().chance_child_entries.reserve(public_state_cap);
    storage_.mutable_ref().frozen_info_set_action_offsets.reserve(
        public_state_cap);
    storage_.mutable_ref().public_info_set_slabs.reserve(public_state_cap);
    storage_.mutable_ref().betting_nodes.reserve(public_state_cap);
    storage_.mutable_ref().betting_edges.reserve(public_state_cap);
  }
}

CFRSolver::Deal CFRSolver::traversal_deal(RangeDeal deal) const {
  const std::array<ComboId, kPlayerCount> hands = {
      deal.player_a_combo,
      deal.player_b_combo,
  };
  return Deal{hands, ComboMask(hands[0]) | ComboMask(hands[1])};
}

CFRSolver::TraversalOptions CFRSolver::traversal_options(
    int iteration,
    int max_depth) const {
  TraversalOptions options;
  options.update_player = iteration % kPlayerCount;
  options.iteration = iteration;
  options.max_depth = max_depth;
  options.write_average_strategy = !config_.regret_only_training;
  return options;
}

void CFRSolver::log_training_summary() const {
  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << iterations_run_;
  LOG(INFO) << "Information sets: " << get_info_set_count();
  LOG(INFO) << "Public states: " << get_public_state_count();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

CFRSolver::TraversalScratch::TraversalScratch(size_t depth_count) {
  frames.reserve(depth_count);
}

CFRSolver::RangeScratchFrame& CFRSolver::TraversalScratch::frame(
    size_t depth) {
  if (depth >= frames.capacity()) {
    throw std::logic_error("TraversalScratch depth reserve exhausted");
  }
  while (frames.size() <= depth) {
    frames.emplace_back();
  }
  return frames[depth];
}

CFRSolver::MutableTraversalGraph::MutableTraversalGraph(CFRSolver& solver)
    : solver_(solver) {}

std::optional<CFRSolver::NodeRef>
CFRSolver::MutableTraversalGraph::action_child(NodeRef parent,
                                               int action_index) {
  const auto child_id = solver_.graph_builder_.get_or_create_action_child(
      parent.public_state_id, action_index, parent.exact_board);
  if (!child_id.has_value() ||
      *child_id == kInvalidPublicStateId ||
      *child_id == kCappedPublicStateId ||
      *child_id >= solver_.rows().size()) {
    return std::nullopt;
  }
  return NodeRef{*child_id, parent.exact_board, parent.board_features};
}

std::optional<CFRSolver::NodeRef>
CFRSolver::MutableTraversalGraph::sample_chance_child(
    NodeRef parent,
    CardMask known_private_cards) {
  const auto& public_rows = solver_.rows();
  if (parent.public_state_id >= public_rows.size()) {
    return std::nullopt;
  }
  const PublicStateRow& row = public_rows[parent.public_state_id];
  if (row.betting_node_id >= solver_.tables().betting_nodes.size()) {
    return std::nullopt;
  }
  ExactGameState exact_parent_state{
      solver_.tables().betting_nodes[row.betting_node_id].state,
      parent.exact_board,
  };
  const auto cards = SampleStreetCards(exact_parent_state.betting.street,
                                       exact_parent_state.board,
                                       known_private_cards, solver_.rng_);
  ExactGameState exact_child_state = ApplyChance(exact_parent_state, cards);
  const auto child_id = solver_.graph_builder_.get_or_create_chance_child(
      parent.public_state_id, exact_child_state);
  if (!child_id.has_value() ||
      *child_id == kInvalidPublicStateId ||
      *child_id == kCappedPublicStateId ||
      *child_id >= public_rows.size()) {
    return std::nullopt;
  }
  return NodeRef{
      *child_id,
      exact_child_state.board,
      board_features(exact_child_state.board),
  };
}

CFRSolver::FrozenTraversalGraph::FrozenTraversalGraph(CFRSolver& solver)
    : solver_(solver) {}

uint32_t CFRSolver::FrozenTraversalGraph::required_action_child_id(
    uint32_t parent_public_state_id,
    int action_index) const {
  const auto& rows = solver_.rows();
  if (parent_public_state_id >= rows.size()) {
    throw std::logic_error("frozen action parent public state is invalid");
  }
  const PublicStateRow& row = rows[parent_public_state_id];
  if (row.betting_node_id >= solver_.tables().betting_nodes.size()) {
    throw std::logic_error("frozen action parent betting node is invalid");
  }
  const auto& node = solver_.tables().betting_nodes[row.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error("frozen action child index out of range");
  }
  const size_t child_slot = static_cast<size_t>(row.action_child_offset) +
                            static_cast<size_t>(action_index);
  if (child_slot >= solver_.tables().action_child_ids.size()) {
    throw std::logic_error("required action child public state is missing");
  }
  const uint32_t child_id = solver_.tables().action_child_ids[child_slot];
  if (child_id == kInvalidPublicStateId ||
      child_id == kCappedPublicStateId ||
      child_id >= rows.size()) {
    throw std::logic_error("required action child public state is missing");
  }
  return child_id;
}

uint32_t CFRSolver::FrozenTraversalGraph::required_chance_child_id(
    uint32_t parent_public_state_id,
    const ExactGameState& child_state) const {
  const auto child_id =
      solver_.tables().chance_child(
          parent_public_state_id,
          public_bucket(child_state.betting.street, child_state.board));
  if (!child_id.has_value() ||
      *child_id == kInvalidPublicStateId ||
      *child_id == kCappedPublicStateId ||
      *child_id >= solver_.rows().size()) {
    throw std::logic_error("required chance child public state is missing");
  }
  return *child_id;
}

CFRSolver::NodeRef CFRSolver::FrozenTraversalGraph::action_child(
    NodeRef parent,
    int action_index) const {
  return NodeRef{
      required_action_child_id(parent.public_state_id, action_index),
      parent.exact_board,
      parent.board_features,
  };
}

CFRSolver::NodeRef CFRSolver::FrozenTraversalGraph::sample_chance_child(
    NodeRef parent,
    CardMask known_private_cards) {
  const auto& public_rows = solver_.rows();
  if (parent.public_state_id >= public_rows.size()) {
    throw std::logic_error("frozen chance parent public state is invalid");
  }
  const PublicStateRow& row = public_rows[parent.public_state_id];
  if (row.betting_node_id >= solver_.tables().betting_nodes.size()) {
    throw std::logic_error("frozen chance parent betting node is invalid");
  }
  ExactGameState exact_parent_state{
      solver_.tables().betting_nodes[row.betting_node_id].state,
      parent.exact_board,
  };
  const auto cards = SampleStreetCards(exact_parent_state.betting.street,
                                       exact_parent_state.board,
                                       known_private_cards, solver_.rng_);
  ExactGameState exact_child_state = ApplyChance(exact_parent_state, cards);
  return NodeRef{
      required_chance_child_id(parent.public_state_id, exact_child_state),
      exact_child_state.board,
      board_features(exact_child_state.board),
  };
}

bool CFRSolver::prebuild_info_set_rows(
    const TrainingRangeView& a_view,
    const TrainingRangeView& b_view,
    absl::Span<const std::optional<Board>> row_boards) {
  if (storage_.frozen) {
    return true;
  }

  if constexpr (kCoarsePublicBuckets) {
    for (uint32_t node_id = 0; node_id < rows().size(); ++node_id) {
      if (node_id >= row_boards.size() || !row_boards[node_id].has_value()) {
        continue;
      }
      const PublicStateRow& row = rows()[node_id];
      if (row.betting_node_id >= tables().betting_nodes.size()) {
        continue;
      }
      const auto& node = tables().betting_nodes[row.betting_node_id];
      if (node.kind != StrategyTables::NodeKind::kDecision ||
          node.action_count == 0 || !IsPlayer(node.state.player_to_act)) {
        continue;
      }
      for (uint32_t bucket = 0; bucket < StrategyTables::kPrivateBucketCount;
           ++bucket) {
        const PrivateBucketId private_bucket = static_cast<PrivateBucketId>(bucket);
        const InfoSetAddress infoset{node_id, private_bucket};
        if (!strategy_store_.get_or_create(infoset, node.action_count)
                 .has_value()) {
          return false;
        }
      }
    }
    return true;
  }

  std::vector<uint32_t> seen_buckets;
  uint32_t seen_generation = 1;

  for (uint32_t public_state_id = 0;
       public_state_id < rows().size();
       ++public_state_id) {
    const uint32_t node_id = public_state_id;
    if (node_id >= row_boards.size() || !row_boards[node_id].has_value()) {
      continue;
    }
    const PublicStateRow& row = rows()[node_id];
    if (row.betting_node_id >= tables().betting_nodes.size()) {
      continue;
    }
    const auto& node = tables().betting_nodes[row.betting_node_id];
    if (node.kind != StrategyTables::NodeKind::kDecision ||
        node.action_count == 0 || !IsPlayer(node.state.player_to_act)) {
      continue;
    }

    const int player = node.state.player_to_act;
    const TrainingRangeView& range = player == 0 ? a_view : b_view;
    if (range.empty()) {
      continue;
    }

    const Board& board = *row_boards[node_id];
    const BoardFeatures features = board_features(board);
    const uint32_t bucket_count = private_bucket_count(node.state.street);
    if (bucket_count == 0 ||
        bucket_count > StrategyTables::kPrivateBucketCount) {
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
    const CardMask board_mask = board.mask;
    for (size_t i = 0; i < range.size(); ++i) {
      if (range.weight(i) <= 0.0f) {
        continue;
      }
      const ComboId combo_id = range.combo(i);
      if ((ComboMask(combo_id) & board_mask) != 0) {
        continue;
      }
      const PrivateBucketId bucket =
          private_bucket(combo_id, node.state.street, features);
      if (bucket >= bucket_count) {
        return false;
      }
      if (seen_buckets[bucket] == generation) {
        continue;
      }
      seen_buckets[bucket] = generation;
      if (!strategy_store_
               .get_or_create({node_id, bucket}, node.action_count)
               .has_value()) {
        return false;
      }
    }
  }

  return true;
}

void CFRSolver::run(int iterations,
                    const HandRange& a_range_spec,
                    const HandRange& b_range_spec) {
  last_training_run_stats_ = {};
  if (iterations <= 0) {
    return;
  }

  const auto a_range = BuildTrainingRange(a_range_spec);
  const auto b_range = BuildTrainingRange(b_range_spec);
  TrainingRangeView a_view(a_range);
  TrainingRangeView b_view(b_range);
  RangeSampler sampler(a_range, b_range);

  VLOG(1) << "Preparing compact public-state rows...";
  const auto maybe_root_id = graph_builder_.get_or_create_row(initial_state_);
  if (!maybe_root_id.has_value()) {
    return;
  }
  const uint32_t root_id = *maybe_root_id;
  const auto& root_row = rows()[root_id];
  int root_actions = 0;
  if (root_row.betting_node_id < tables().betting_nodes.size()) {
    root_actions = tables().betting_nodes[root_row.betting_node_id].action_count;
  }
  VLOG(1) << "Root row has " << root_actions << " actions";

  const int threads = config_.num_training_threads;
  const int max_depth = config_.max_depth;
  const bool parallel = threads > 1;
  const bool regret_only = config_.regret_only_training && max_depth == 0;
  const bool bounded = config_.max_public_states > 0 || max_depth > 0;

  bool fixed_storage = storage_.frozen;
  if (!fixed_storage && (parallel || regret_only) && bounded) {
    fixed_storage = prepare_prebuilt_training(root_id, max_depth, a_view, b_view);
  }

  if (fixed_storage) {
    run_fixed_storage_iterations(
        iterations, threads, root_id, initial_state_.board, sampler, a_range,
        b_range);
  } else {
    run_growing_iterations(iterations, root_id, sampler, a_view, b_view,
                           max_depth);
  }

  log_training_summary();
}

bool CFRSolver::prepare_prebuilt_training(
    uint32_t root_id,
    int max_depth,
    const TrainingRangeView& a_view,
    const TrainingRangeView& b_view) {
  TrainingRunStats& stats = last_training_run_stats_;
  if (storage_.frozen) {
    return true;
  }

  auto record_public_counts = [&] {
    const int64_t public_states = static_cast<int64_t>(get_public_state_count());
    stats.prebuild_public_states = public_states;
  };
  auto record_action_counts = [&] {
    const int64_t info_sets = static_cast<int64_t>(get_info_set_count());
    const auto& regrets = arrays().cumulative_regrets;
    const int64_t actions = static_cast<int64_t>(regrets.size());
    stats.prebuild_info_sets = info_sets;
    stats.prebuild_action_entries = actions;
  };

  VLOG(1) << "Prebuilding compact public-state rows...";
  GraphBuilder& graph = graph_builder_;
  std::vector<std::optional<Board>> row_boards;
  const auto prebuild_start = std::chrono::steady_clock::now();
  const bool public_rows_complete = graph.prebuild_reachable_rows(
      root_id, initial_state_.board, max_depth, row_boards);
  const auto prebuild_end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> prebuild_seconds =
      prebuild_end - prebuild_start;
  stats.public_state_prebuild_complete = public_rows_complete;
  stats.prebuild_seconds = prebuild_seconds.count();
  record_public_counts();
  if (!stats.public_state_prebuild_complete) {
    return false;
  }

  if (!graph.validate_prebuilt_rows(root_id, initial_state_.board, max_depth,
                                    stats)) {
    return false;
  }
  record_action_counts();

  VLOG(1) << "Prebuilding infoset rows...";
  const auto info_set_prebuild_start = std::chrono::steady_clock::now();
  const bool infosets_complete =
      prebuild_info_set_rows(a_view, b_view, row_boards);
  const auto info_set_prebuild_end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> info_set_seconds =
      info_set_prebuild_end - info_set_prebuild_start;
  stats.info_set_prebuild_complete = infosets_complete;
  stats.info_set_prebuild_seconds = info_set_seconds.count();
  record_action_counts();
  if (!stats.info_set_prebuild_complete) {
    return false;
  }

  StrategyStore& store = strategy_store_;
  const bool frozen_lookup_complete = store.prebuild_frozen_info_set_action_offsets();
  stats.frozen_info_set_lookup_prebuild_complete = frozen_lookup_complete;
  if (!stats.frozen_info_set_lookup_prebuild_complete) {
    return false;
  }
  const int64_t frozen_lookup_rows =
      static_cast<int64_t>(tables().frozen_info_set_action_offsets.size());
  stats.prebuild_frozen_info_set_lookup_rows = frozen_lookup_rows;
  return true;
}

void CFRSolver::run_growing_iterations(
    int iterations,
    uint32_t root_id,
    RangeSampler& sampler,
    const TrainingRangeView& a_view,
    const TrainingRangeView& b_view,
    int max_depth) {
  LOG(INFO) << "Starting CFR iterations...";
  VLOG(1) << "Growing storage backend: " << iterations
          << " single-threaded iterations";
  MutableTraversalGraph graph(*this);
  const NodeRef root_node{
      root_id,
      initial_state_.board,
      board_features(initial_state_.board),
  };
  TraversalScratch scratch(ScratchDepthReserve(config_, max_depth));
  const int64_t warmup_start_updates = cfr_update_count_;
  const auto warmup_start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; ++i) {
    const RangeDeal sampled = sampler.sample(rng_);
    const Deal deal = traversal_deal(sampled);

    VLOG(2) << "Iteration " << i + 1 << "/" << iterations;
    const int cfr_iteration = iterations_run_;
    TraversalFrame frame;
    if (max_depth > 0) {
      frame.ranges[0] = &a_view;
      frame.ranges[1] = &b_view;
    }
    TraversalOptions options = traversal_options(cfr_iteration, max_depth);
    TraversalRun run{deal, options, &scratch};
    const double dealt_value = cfr(root_node, run, frame, graph);

    cumulative_root_utility_ += dealt_value;
    ++iterations_run_;
  }

  const auto warmup_end = std::chrono::steady_clock::now();
  last_training_run_stats_.warmup_iterations = iterations;
  last_training_run_stats_.warmup_seconds =
      std::chrono::duration<double>(warmup_end - warmup_start).count();
  last_training_run_stats_.warmup_cfr_updates =
      cfr_update_count_ - warmup_start_updates;
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

void CFRSolver::run_fixed_storage_iterations(
    int iterations,
    int num_threads,
    uint32_t root_id,
    const Board& root_board,
    const RangeSampler& sampler,
    const TrainingRange& a_range,
    const TrainingRange& b_range) {
  if (!storage_.frozen) {
    storage_.freeze();
  }

  LOG(INFO) << "Starting fixed-storage CFR iterations ("
            << iterations << " iterations, " << num_threads << " workers)...";
  const int64_t frozen_start_updates = cfr_update_count_;
  const auto frozen_start = std::chrono::steady_clock::now();

  // Each worker shares the frozen strategy tables and writes to the same
  // regret/strategy arrays.
  // Workers use their own RNG and TraversalScratch; no locks needed.
  auto fixed_tables = storage_.frozen_tables;
  auto cumulative = storage_.cumulative;
  const bool depth_zero = config_.max_depth == 0;
  const bool regret_only_config = config_.regret_only_training && depth_zero;
  const bool use_fixed_infoset_lookup =
      storage_.frozen && regret_only_config && !kCoarsePublicBuckets;
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
      [this, root_id, root_board, &sampler, fixed_tables, cumulative,
       use_fixed_infoset_lookup, use_atomic_updates, &a_range, &b_range](
          int iteration_begin, int shard, unsigned int seed) mutable {
          // Build a lightweight worker that shares frozen tables.
          CFRSolver worker(config_);
          worker.storage_.bind_frozen(fixed_tables, cumulative);
          worker.rng_.seed(seed);

          TrainingRangeView a_view;
          TrainingRangeView b_view;
          const int max_depth = config_.max_depth;
          size_t scratch_depth = 0;
          if (!use_fixed_infoset_lookup) {
            scratch_depth = ScratchDepthReserve(config_, max_depth);
          }
          TraversalScratch scratch(scratch_depth);
          if (!use_fixed_infoset_lookup) {
            // Per-worker range views (read-only, built from shared training data).
            a_view.reset_to_all(a_range);
            b_view.reset_to_all(b_range);
          }

          double local_utility = 0.0;
          FrozenTraversalGraph graph(worker);
          const NodeRef root_node{
              root_id,
              root_board,
              board_features(root_board),
          };
          for (int i = 0; i < shard; ++i) {
            const RangeDeal sampled = sampler.sample(worker.rng_);
            const Deal deal = worker.traversal_deal(sampled);

            const int cfr_iteration = iteration_begin + i;
            TraversalFrame frame;
            if (max_depth > 0) {
              frame.ranges[0] = &a_view;
              frame.ranges[1] = &b_view;
            }
            auto options = worker.traversal_options(cfr_iteration, max_depth);
            if (use_fixed_infoset_lookup) {
              if (use_atomic_updates) {
                options.regret_load_mode = RegretLoadMode::kAtomic;
                options.regret_update_mode = RegretUpdateMode::kAtomic;
              } else {
                options.regret_load_mode = RegretLoadMode::kPlain;
                options.regret_update_mode = RegretUpdateMode::kPlain;
              }
              options.use_fixed_infoset_lookup = true;
              options.write_average_strategy = false;
              options.record_atomic_retry_stats = use_atomic_updates;
            }
            TraversalRun run{deal, options, &scratch};
            local_utility += worker.cfr(root_node, run, frame, graph);
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

  const auto frozen_end = std::chrono::steady_clock::now();
  last_training_run_stats_.frozen_iterations = iterations;
  last_training_run_stats_.frozen_seconds =
      std::chrono::duration<double>(frozen_end - frozen_start).count();
  last_training_run_stats_.frozen_cfr_updates =
      cfr_update_count_ - frozen_start_updates;
}

template <typename Graph>
double CFRSolver::CfrTraversal<Graph>::value(
    NodeRef node,
    const TraversalFrame& frame) {
  const uint32_t node_id = node.public_state_id;
  const auto& rows = solver_.rows();
  if (node_id >= rows.size()) {
    return 0.0;
  }
  const PublicStateRow& row = rows[node_id];
  if (row.betting_node_id >= solver_.tables().betting_nodes.size()) {
    return 0.0;
  }
  const auto& betting_node =
      solver_.tables().betting_nodes[row.betting_node_id];

  switch (betting_node.kind) {
    case StrategyTables::NodeKind::kTerminal:
      return terminal(node, row);
    case StrategyTables::NodeKind::kChance:
      return chance(node, frame);
    case StrategyTables::NodeKind::kFrontier:
      return depth_limit_value(node, row);
    case StrategyTables::NodeKind::kDecision:
      break;
  }

  const int max_depth = run_.options.max_depth;
  if (max_depth > 0 && frame.decision_depth >= max_depth) {
    return depth_limit_value(node, row);
  }

  return decision(node, row, frame);
}

template <typename Graph>
double CFRSolver::CfrTraversal<Graph>::terminal(
    NodeRef node,
    const PublicStateRow& row) {
  const auto& betting_node =
      solver_.tables().betting_nodes[row.betting_node_id];
  solver_.traversal_stats_.record_terminal(
      betting_node.state.folded_player < 0);
  return solver_.terminal_utility(row, node.exact_board,
                                  run_.deal.hand(0),
                                  run_.deal.hand(1));
}

template <typename Graph>
double CFRSolver::CfrTraversal<Graph>::chance(
    NodeRef node,
    const TraversalFrame& frame) {
  const int samples = ChanceSamples(solver_.config_);
  solver_.traversal_stats_.record_chance_samples(samples);
  return solver_.sample_chance_children(
      samples, node, run_.deal.known_private_cards(), graph_,
      [&](NodeRef child) {
        TraversalFrame child_frame = frame;
        ++child_frame.scratch_depth;
        if (frame.ranges[0] != nullptr || frame.ranges[1] != nullptr) {
          RangeScratchFrame& scratch =
              run_.scratch->frame(child_frame.scratch_depth);
          for (size_t player = 0; player < kPlayerCount; ++player) {
            if (frame.ranges[player] == nullptr) {
              continue;
            }
            child_frame.ranges[player] =
                &frame.ranges[player]->copy_without_mask_into(
                    child.exact_board.mask, scratch.filtered_ranges[player]);
          }
        }
        return value(child, child_frame);
      });
}

template <typename Graph>
double CFRSolver::CfrTraversal<Graph>::depth_limit_value(
    NodeRef node,
    const PublicStateRow& row) {
  const auto& betting_node =
      solver_.tables().betting_nodes[row.betting_node_id];
  return IsBettingRoundOver(betting_node.state)
             ? solver_.terminal_utility(row, node.exact_board,
                                        run_.deal.hand(0),
                                        run_.deal.hand(1))
             : 0.0;
}

template <typename Graph>
double CFRSolver::CfrTraversal<Graph>::decision(
    NodeRef node,
    const PublicStateRow& row,
    const TraversalFrame& frame) {
  const uint32_t node_id = node.public_state_id;
  if (row.betting_node_id >= solver_.tables().betting_nodes.size()) {
    return 0.0;
  }
  const auto& betting_node =
      solver_.tables().betting_nodes[row.betting_node_id];
  if (betting_node.kind != StrategyTables::NodeKind::kDecision ||
      betting_node.action_count == 0 ||
      !IsPlayer(betting_node.state.player_to_act)) {
    return 0.0;
  }
  const int player = betting_node.state.player_to_act;
  const ComboId hand = run_.deal.hand(player);

  const bool update_player = player == run_.options.update_player;
  const size_t action_count = betting_node.action_count;
  const PrivateBucketId bucket =
      private_bucket(hand, betting_node.state.street, node.board_features);
  std::optional<ActionBlock> actions;
  if (run_.options.use_fixed_infoset_lookup) {
    actions = solver_.strategy_store_.find_frozen(node_id, bucket,
                                                  action_count);
  } else {
    const InfoSetAddress infoset{node_id, bucket};
    if (update_player) {
      actions = solver_.strategy_store_.get_or_create(infoset, action_count);
    } else {
      actions = solver_.strategy_store_.find(infoset, action_count);
    }
  }

  std::array<double, kMaxActionsPerNode> action_probabilities_storage{};
  std::array<double, kMaxActionsPerNode> action_values_storage{};
  absl::Span<double> action_probabilities(
      action_probabilities_storage.data(), action_count);
  absl::Span<double> action_values(action_values_storage.data(), action_count);
  std::fill(action_values.begin(), action_values.end(), 0.0);
  solver_.strategy_store_.regret_matching_or_uniform(
      actions, action_count, run_.options.regret_load_mode,
      action_probabilities);

  double node_value = 0.0;
  absl::Span<TrainingRangeView> conditioned_ranges;
  const bool override_range = frame.ranges[static_cast<size_t>(player)] != nullptr;
  if (override_range) {
    RangeScratchFrame& scratch =
        run_.scratch->frame(static_cast<size_t>(frame.scratch_depth));
    conditioned_ranges = solver_.condition_ranges_for_actions(
        *frame.ranges[static_cast<size_t>(player)], betting_node.state.street,
        node.exact_board, node.board_features, node_id, player, action_count,
        scratch);
  }

  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    const int action = static_cast<int>(action_index);
    const auto child = graph_.action_child(node, action);
    auto visit_child = [&](NodeRef child_node) {
      TraversalFrame child_frame = frame;
      child_frame.reach[static_cast<size_t>(player)] *=
          action_probabilities[action_index];
      ++child_frame.decision_depth;
      ++child_frame.scratch_depth;
      if (override_range) {
        child_frame.ranges[static_cast<size_t>(player)] =
            &conditioned_ranges[action_index];
      }
      return value(child_node, child_frame);
    };

    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(child)>,
                                 std::optional<NodeRef>>) {
      if (!child.has_value()) {
        continue;
      }
      const double action_value = visit_child(*child);
      action_values[action_index] = action_value;
      node_value += action_probabilities[action_index] * action_value;
    } else {
      const double action_value = visit_child(child);
      action_values[action_index] = action_value;
      node_value += action_probabilities[action_index] * action_value;
    }
  }

  ++solver_.cfr_update_count_;
  solver_.traversal_stats_.record_decision(betting_node.state.street,
                                           frame.decision_depth);

  if (actions.has_value() && update_player) {
    const double opponent_reach =
        frame.reach[static_cast<size_t>(1 - player)];
    const RegretUpdateOptions regret_options{
        run_.options.regret_update_mode,
        run_.options.record_atomic_retry_stats,
    };
    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double sign = player == 0 ? 1.0 : -1.0;
      const double delta = action_values[action_index] - node_value;
      const double regret = opponent_reach * sign * delta;

      actions->add_cfr_plus_regret(
          action_index, static_cast<float>(regret), regret_options);
    }

    if (run_.options.write_average_strategy) {
      const double weight =
          frame.reach[static_cast<size_t>(player)] *
          static_cast<double>(run_.options.iteration + 1);
      actions->add_average_strategy(
          action_probabilities, weight, run_.options.regret_update_mode);
    }
  }

  return node_value;
}

template <typename Graph>
double CFRSolver::cfr(
    NodeRef node,
    TraversalRun& run,
    const TraversalFrame& frame,
    Graph& graph) {
  return CfrTraversal<Graph>(*this, run, graph).value(node, frame);
}

template <typename Graph, typename EvalChild>
double CFRSolver::sample_chance_children(
    int samples,
    NodeRef node,
    CardMask known_private_cards,
    Graph& graph,
    EvalChild&& eval_child) {
  double value = 0.0;
  int evaluated = 0;
  for (int i = 0; i < samples; ++i) {
    const auto child = graph.sample_chance_child(node, known_private_cards);
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(child)>,
                                 std::optional<NodeRef>>) {
      if (!child.has_value()) {
        continue;
      }
      value += eval_child(*child);
    } else {
      value += eval_child(child);
    }
    ++evaluated;
  }

  return evaluated > 0 ? value / evaluated : 0.0;
}

template double CFRSolver::cfr<CFRSolver::MutableTraversalGraph>(
    NodeRef node,
    TraversalRun& run,
    const TraversalFrame& frame,
    MutableTraversalGraph& graph);

template double CFRSolver::cfr<CFRSolver::FrozenTraversalGraph>(
    NodeRef node,
    TraversalRun& run,
    const TraversalFrame& frame,
    FrozenTraversalGraph& graph);

absl::Span<TrainingRangeView> CFRSolver::condition_ranges_for_actions(
    const TrainingRangeView& range,
    StreetKind street,
    const Board& board,
    const BoardFeatures& features,
    uint32_t node_id,
    int player,
    size_t action_count,
    RangeScratchFrame& scratch_frame) {
  if (action_count == 0) {
    return {};
  }

  auto& ranges = scratch_frame.conditioned_ranges;
  for (size_t i = 0; i < action_count; ++i) {
    ranges[i].reset_to_filtered();
  }
  const size_t range_size = range.size();
  if (range_size == 0) {
    return absl::Span<TrainingRangeView>(ranges.data(), action_count);
  }

  const CardMask board_mask = board.mask;
  std::array<double, kMaxActionsPerNode> action_probabilities_storage{};
  absl::Span<double> action_probabilities(
      action_probabilities_storage.data(), action_count);
  for (size_t i = 0; i < range_size; ++i) {
    const float range_weight = range.weight(i);
    const ComboId combo_id = range.combo(i);
    if (range_weight <= 0.0 || (ComboMask(combo_id) & board_mask) != 0) {
      continue;
    }

    const PrivateBucketId bucket = private_bucket(combo_id, street, features);
    strategy_store_.regret_matching_for_bucket(
        node_id, bucket, action_count, action_probabilities);

    for (size_t action_index = 0; action_index < action_count; ++action_index) {
      const double probability = action_probabilities[action_index];
      const double conditioned_weight = range_weight * probability;
      if (conditioned_weight > 0.0) {
        ranges[action_index].add(combo_id,
                                 static_cast<float>(conditioned_weight));
      }
    }
  }

  return absl::Span<TrainingRangeView>(ranges.data(), action_count);
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand) {
  const auto maybe_root_id = graph_builder_.get_or_create_row(initial_state_);
  if (!maybe_root_id.has_value()) {
    return 0.0;
  }
  const uint32_t root_id = *maybe_root_id;
  const NodeRef root_node{
      root_id,
      initial_state_.board,
      board_features(initial_state_.board),
  };
  const Deal deal{{player_a_hand, player_b_hand},
                  ComboMask(player_a_hand) | ComboMask(player_b_hand)};
  if (storage_.frozen) {
    FrozenTraversalGraph graph(*this);
    return evaluate_strategy_node(root_node, deal, graph);
  }
  MutableTraversalGraph graph(*this);
  return evaluate_strategy_node(root_node, deal, graph);
}

double CFRSolver::evaluate_strategy(int samples, const HandRange& player_a_range,
                                    const HandRange& player_b_range) {
  if (samples <= 0) {
    return 0.0;
  }

  const TrainingRange a_range = BuildTrainingRange(player_a_range);
  const TrainingRange b_range = BuildTrainingRange(player_b_range);
  RangeSampler sampler(a_range, b_range);
  const auto maybe_root_id = graph_builder_.get_or_create_row(initial_state_);
  if (!maybe_root_id.has_value()) {
    return 0.0;
  }
  return evaluate_strategy_samples(samples, *maybe_root_id,
                                   initial_state_.board, sampler, true);
}

double CFRSolver::evaluate_strategy_samples(
    int samples,
    uint32_t root_id,
    const Board& root_board,
    const RangeSampler& sampler,
    bool allow_parallel) {
  if (samples <= 0) {
    return 0.0;
  }
  const bool enough_samples = samples >= kParallelEvaluationSampleThreshold;
  const bool parallel = allow_parallel && enough_samples;
  const int worker_count = parallel ? WorkerCountForSamples(samples) : 1;
  if (worker_count > 1) {
    SolverConfig config = config_;
    auto frozen_tables = storage_.frozen_tables;
    std::shared_ptr<MutableCumulativeArrays> cumulative = storage_.cumulative;
    double total = 0.0;
    run_sharded(
        samples, worker_count, 0,
        [config, &sampler, root_id, root_board, frozen_tables, cumulative](
            int, int shard_samples, unsigned int seed) mutable {
          CFRSolver worker(config);
          worker.storage_.bind_frozen(frozen_tables, cumulative);
          worker.rng_.seed(seed);
          const double value = worker.evaluate_strategy_samples(
              shard_samples, root_id, root_board, sampler, false);
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

  double total = 0.0;
  if (root_id >= rows().size()) {
    return 0.0;
  }
  const NodeRef root_node{
      root_id,
      root_board,
      board_features(root_board),
  };
  if (storage_.frozen) {
    FrozenTraversalGraph graph(*this);
    for (int i = 0; i < samples; ++i) {
      const RangeDeal sampled = sampler.sample(rng_);
      const Deal deal = traversal_deal(sampled);
      total += evaluate_strategy_node(root_node, deal, graph);
    }
  } else {
    MutableTraversalGraph graph(*this);
    for (int i = 0; i < samples; ++i) {
      const RangeDeal sampled = sampler.sample(rng_);
      const Deal deal = traversal_deal(sampled);
      total += evaluate_strategy_node(root_node, deal, graph);
    }
  }
  return total / samples;
}

double CFRSolver::evaluate_strategy_node(
    NodeRef node,
    const Deal& deal,
    MutableTraversalGraph& graph) {
  return evaluate_strategy_node_impl(node, deal, graph);
}

double CFRSolver::evaluate_strategy_node(
    NodeRef node,
    const Deal& deal,
    FrozenTraversalGraph& graph) {
  return evaluate_strategy_node_impl(node, deal, graph);
}

template <typename Graph>
double CFRSolver::evaluate_strategy_node_impl(
    NodeRef node,
    const Deal& deal,
    Graph& graph) {
  const auto& public_rows = rows();
  if (node.public_state_id >= public_rows.size()) {
    return 0.0;
  }
  const uint32_t node_id = node.public_state_id;
  const PublicStateRow& row = public_rows[node_id];
  if (row.betting_node_id >= tables().betting_nodes.size()) {
    return 0.0;
  }
  const auto& betting_node = tables().betting_nodes[row.betting_node_id];

  switch (betting_node.kind) {
    case StrategyTables::NodeKind::kTerminal:
      return terminal_utility(row, node.exact_board, deal.hand(0),
                              deal.hand(1));
    case StrategyTables::NodeKind::kChance: {
      const int samples = ChanceSamples(config_);
      return sample_chance_children(
          samples, node, deal.known_private_cards(), graph,
          [&](NodeRef child) {
            return evaluate_strategy_node_impl(child, deal, graph);
          });
    }
    case StrategyTables::NodeKind::kFrontier:
      return 0.0;
    case StrategyTables::NodeKind::kDecision:
      break;
  }
  if (betting_node.kind != StrategyTables::NodeKind::kDecision ||
      betting_node.action_count == 0 ||
      !IsPlayer(betting_node.state.player_to_act)) {
    return 0.0;
  }
  const int player = betting_node.state.player_to_act;

  std::array<double, kMaxActionsPerNode> probabilities_storage{};
  absl::Span<double> probabilities(
      probabilities_storage.data(), betting_node.action_count);
  const PrivateBucketId bucket =
      private_bucket(deal.hand(player), betting_node.state.street,
                     node.board_features);
  strategy_store_.average_strategy(
      node_id, bucket, betting_node.action_count,
      config_.regret_only_training, probabilities);

  double value = 0.0;
  const int action_count = betting_node.action_count;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    const auto child = graph.action_child(node, action_index);
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(child)>,
                                 std::optional<NodeRef>>) {
      if (!child.has_value()) {
        continue;
      }
      value += probabilities[action_index] *
               evaluate_strategy_node_impl(*child, deal, graph);
    } else {
      value += probabilities[action_index] *
               evaluate_strategy_node_impl(child, deal, graph);
    }
  }
  return value;
}

double CFRSolver::get_expected_value(int player_id) const {
  if (iterations_run_ == 0) {
    return 0.0;
  }
  const double player_a_ev = cumulative_root_utility_ / iterations_run_;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

bool CFRSolver::traversal_stats_enabled() {
  return kTraversalStatsEnabled;
}

double CFRSolver::terminal_utility(const PublicStateRow& row,
                                   const Board& board,
                                   ComboId player_a_hand,
                                   ComboId player_b_hand) {
  if (row.betting_node_id >= tables().betting_nodes.size()) {
    return 0.0;
  }
  const BettingState& state =
      tables().betting_nodes[row.betting_node_id].state;
  const auto utility = UtilityBeforeShowdown(state, board.count);
  if (utility.has_value()) {
    return *utility;
  }

  // Frozen sampled training sees mostly one-off showdowns; direct evaluation
  // is faster than paying shared cache lookup/mutation overhead.
  HandEvaluator evaluator;
  const int comparison = evaluator.compare_hands(
      player_a_hand, player_b_hand, board);
  return ShowdownUtilityFromComparison(state, comparison);
}

} // namespace poker
