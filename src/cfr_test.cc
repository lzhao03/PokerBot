#include "src/cfr_solver.h"
#include "src/combo.h"
#include "src/strategy_store.h"

#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

HandRange ExactRange(ComboId combo) {
  HandRange range;
  range.add_combo(combo, 1.0);
  return range;
}

HandRange TwoComboRange(ComboId first, ComboId second) {
  HandRange range = ExactRange(first);
  range.add_combo(second, 1.0);
  return range;
}

SolverConfig BaseConfig() {
  SolverConfig config;
  config.starting_stack_size = 12;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.chance_samples = 1;
  return config;
}

SolverConfig EquivalenceConfig() {
  SolverConfig config = BaseConfig();
  config.max_depth = 1;
  config.max_public_states = 5000;
  config.max_info_sets = 5000;
  config.regret_only_training = false;
  return config;
}

ExactGameState RiverState() {
  ExactGameState state;
  state.betting.stack = {20, 20};
  state.betting.committed = {10, 10};
  state.betting.street = StreetKind::kRiver;
  state.betting.player_to_act = 1;
  state.board.add(MakeCardId(3, SuitKind::kHearts));
  state.board.add(MakeCardId(7, SuitKind::kDiamonds));
  state.board.add(MakeCardId(12, SuitKind::kClubs));
  state.board.add(MakeCardId(9, SuitKind::kSpades));
  state.board.add(MakeCardId(4, SuitKind::kHearts));
  return state;
}

struct SolverSnapshot {
  double value = 0.0;
  int iterations = 0;
  size_t info_sets = 0;
  size_t nodes = 0;
  int64_t cfr_updates = 0;
  TraversalStats stats;
  std::vector<uint64_t> graph;
  std::vector<float> regrets;
  std::vector<float> strategies;
};

struct RangeScratchProbe {
  size_t root_size = 0;
  size_t after_flop_size = 0;
  size_t after_turn_size = 0;
  size_t after_flop_size_after_turn = 0;
};

}  // namespace

struct CFRSolverTestAccess {
  static uint64_t Mix(uint64_t hash, uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
  }

  static std::vector<uint64_t> GraphFingerprint(const CFRSolver& solver) {
    std::vector<uint64_t> out;
    const StrategyTables& tables = solver.tables();
    out.reserve(tables.nodes.size() + tables.chance_child_entries.size() +
                tables.betting_nodes.size() + tables.betting_edges.size());

    for (size_t node_id = 0; node_id < tables.nodes.size(); ++node_id) {
      const StrategyTables::Node& row = tables.nodes[node_id];
      uint64_t hash = 0;
      hash = Mix(hash, row.betting_node_id);
      hash = Mix(hash, row.board_bucket);
      hash = Mix(hash, row.action_child_offset);
      hash = Mix(hash, row.chance_child_offset);
      hash = Mix(hash, row.chance_child_count);
      const auto& node = tables.betting_nodes[row.betting_node_id];
      for (int action = 0; action < node.action_count; ++action) {
        hash = Mix(
            hash,
            tables.action_child(static_cast<uint32_t>(node_id), action)
                .value_or(kInvalidNodeId));
      }
      out.push_back(hash);
    }

    for (const StrategyTables::BettingNode& node : tables.betting_nodes) {
      uint64_t hash = 0;
      hash = Mix(hash, node.action_begin);
      hash = Mix(hash, node.action_count);
      hash = Mix(hash, node.chance_child);
      hash = Mix(hash, static_cast<uint64_t>(node.kind));
      hash = Mix(hash, static_cast<uint64_t>(node.state.player_to_act + 2));
      hash = Mix(hash, Pot(node.state));
      hash = Mix(hash, node.state.stack[0]);
      hash = Mix(hash, node.state.stack[1]);
      out.push_back(hash);
    }

    for (const StrategyTables::BettingEdge& edge : tables.betting_edges) {
      uint64_t hash = 0;
      hash = Mix(hash, static_cast<uint64_t>(edge.action.kind));
      hash = Mix(hash, edge.action.amount);
      hash = Mix(hash, edge.child);
      out.push_back(hash);
    }
    for (const StrategyTables::ChanceChildEntry& entry :
         tables.chance_child_entries) {
      out.push_back(Mix(entry.outcome_id, entry.node_id));
    }
    return out;
  }

  static SolverSnapshot Snapshot(const CFRSolver& solver,
                                 double value = 0.0) {
    const MutableCumulativeArrays& arrays =
        solver.storage_.cumulative_ref();
    return SolverSnapshot{
        value,
        solver.get_iterations_run(),
        solver.get_info_set_count(),
        solver.get_public_state_count(),
        solver.cfr_update_count_,
        solver.traversal_stats_,
        GraphFingerprint(solver),
        std::vector<float>(arrays.cumulative_regrets.begin(),
                           arrays.cumulative_regrets.end()),
        std::vector<float>(arrays.cumulative_strategies.begin(),
                           arrays.cumulative_strategies.end()),
    };
  }

  static SolverSnapshot RunPrepared(bool freeze_storage) {
    const ComboId a_combo =
        Combo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
    const ComboId b_combo =
        Combo(13, SuitKind::kHearts, 13, SuitKind::kSpades);
    const TrainingRange a_range = BuildTrainingRange(ExactRange(a_combo));
    const TrainingRange b_range = BuildTrainingRange(ExactRange(b_combo));
    TrainingRangeView a_view(a_range);
    TrainingRangeView b_view(b_range);

    CFRSolver solver(EquivalenceConfig());
    const std::optional<NodeId> maybe_root =
        solver.graph_builder_.get_or_create_node(solver.initial_state_);
    REQUIRE(maybe_root.has_value());
    const NodeId root_id = *maybe_root;
    REQUIRE(solver.prepare_prebuilt_training(root_id, solver.config_.max_depth,
                                             a_view, b_view));
    if (freeze_storage) {
      solver.storage_.freeze();
    }

    const CFRSolver::Position root{root_id, solver.initial_state_.board};
    CFRSolver::TraversalScratch scratch(32);
    const CFRSolver::Deal deal{
        {a_combo, b_combo},
        ComboMask(a_combo) | ComboMask(b_combo),
    };
    const CFRSolver::TraversalOptions options =
        solver.traversal_options(0, solver.config_.max_depth);
    CFRSolver::TraversalRun run{deal, options, &scratch};
    CFRSolver::TraversalFrame frame;
    frame.ranges[0] = &a_view;
    frame.ranges[1] = &b_view;

    double value = 0.0;
    if (freeze_storage) {
      CFRSolver::FrozenTraversalGraph graph(solver);
      value = solver.cfr(root, run, frame, graph);
    } else {
      CFRSolver::MutableTraversalGraph graph(solver);
      value = solver.cfr(root, run, frame, graph);
    }
    return Snapshot(solver, value);
  }

  static RangeScratchProbe ProbeNestedChanceScratch() {
    TrainingRange source;
    source.add(Combo(14, SuitKind::kHearts, 14, SuitKind::kSpades), 1.0f);
    source.add(Combo(13, SuitKind::kHearts, 13, SuitKind::kSpades), 1.0f);
    source.add(Combo(12, SuitKind::kHearts, 12, SuitKind::kSpades), 1.0f);

    TrainingRangeView root(source);
    CFRSolver::TraversalScratch scratch(3);
    CFRSolver::TraversalFrame root_frame;
    root_frame.ranges[0] = &root;

    CFRSolver::RangeScratchFrame& flop_scratch = scratch.frame(1);
    const TrainingRangeView& after_flop =
        root_frame.ranges[0]->copy_without_mask_into(
            CardBit(MakeCardId(14, SuitKind::kHearts)),
            flop_scratch.filtered_ranges[0]);

    CFRSolver::TraversalFrame turn_frame = root_frame;
    turn_frame.scratch_depth = 1;
    turn_frame.ranges[0] = &after_flop;
    CFRSolver::RangeScratchFrame& turn_scratch = scratch.frame(2);
    const TrainingRangeView& after_turn =
        turn_frame.ranges[0]->copy_without_mask_into(
            CardBit(MakeCardId(13, SuitKind::kHearts)),
            turn_scratch.filtered_ranges[0]);

    return {root.size(), after_flop.size(), after_turn.size(),
            after_flop.size()};
  }
};

namespace {

struct StoreFixture {
  StoreFixture() {
    storage.mutable_tables->nodes.resize(1);
    storage.mutable_tables->betting_nodes.resize(1);
    storage.mutable_tables->nodes[0].betting_node_id = 0;
    storage.mutable_tables->betting_nodes[0].kind =
        StrategyTables::NodeKind::kDecision;
    storage.mutable_tables->betting_nodes[0].state.player_to_act = 0;
  }

  SolverConfig config;
  SolverStorage storage;
  TraversalStats stats;
  StrategyStore store{config, storage, &stats};
};

ActionBlock CreateBlock(StoreFixture& fixture, size_t action_count) {
  fixture.storage.mutable_tables->betting_nodes[0].action_count =
      static_cast<uint8_t>(action_count);
  std::optional<ActionBlock> block =
      fixture.store.get_or_create({0, 0}, action_count);
  REQUIRE(block.has_value());
  return *block;
}

TEST_CASE("prepared mutable and frozen traversals are exactly equivalent") {
  const SolverSnapshot mutable_run = CFRSolverTestAccess::RunPrepared(false);
  const SolverSnapshot frozen_run = CFRSolverTestAccess::RunPrepared(true);

  CHECK(mutable_run.value == doctest::Approx(frozen_run.value).epsilon(1e-12));
  CHECK(mutable_run.cfr_updates == frozen_run.cfr_updates);
  CHECK(mutable_run.stats.terminal_utility_calls ==
        frozen_run.stats.terminal_utility_calls);
  CHECK(mutable_run.stats.action_entry_touches ==
        frozen_run.stats.action_entry_touches);
  CHECK(mutable_run.graph == frozen_run.graph);
  CHECK(mutable_run.regrets == frozen_run.regrets);
  CHECK(mutable_run.strategies == frozen_run.strategies);
}

TEST_CASE("nested chance filtering does not overwrite its parent range") {
  const RangeScratchProbe probe =
      CFRSolverTestAccess::ProbeNestedChanceScratch();
  CHECK(probe.root_size == 3);
  CHECK(probe.after_flop_size == 2);
  CHECK(probe.after_turn_size == 1);
  CHECK(probe.after_flop_size_after_turn == 2);
}

TEST_CASE("strategy evaluation is read-only") {
  const ComboId a_combo =
      Combo(14, SuitKind::kHearts, 2, SuitKind::kHearts);
  const ComboId b_combo =
      Combo(13, SuitKind::kClubs, 2, SuitKind::kClubs);
  SolverConfig config = EquivalenceConfig();
  config.max_depth = 0;
  CFRSolver solver(config, RiverState());
  solver.run(3, ExactRange(a_combo), ExactRange(b_combo));
  const SolverSnapshot before = CFRSolverTestAccess::Snapshot(solver);

  solver.reset_traversal_stats();
  const double value = solver.evaluate_strategy(a_combo, b_combo);
  const SolverSnapshot after = CFRSolverTestAccess::Snapshot(solver);

  CHECK(std::isfinite(value));
  CHECK(after.iterations == before.iterations);
  CHECK(after.info_sets == before.info_sets);
  CHECK(after.nodes == before.nodes);
  CHECK(after.cfr_updates == before.cfr_updates);
  CHECK(after.graph == before.graph);
  CHECK(after.regrets == before.regrets);
  CHECK(after.strategies == before.strategies);
  CHECK(after.stats.cfr_updates == 0);
}

TEST_CASE("public-state limits keep training in mutable warmup") {
  SolverConfig config = BaseConfig();
  config.regret_only_training = true;
  config.max_public_states = 1;
  config.max_info_sets = 500000;
  CFRSolver solver(config);
  solver.run(5,
             ExactRange(Combo(14, SuitKind::kDiamonds,
                              12, SuitKind::kDiamonds)),
             ExactRange(Combo(11, SuitKind::kSpades,
                              10, SuitKind::kSpades)));

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  CHECK(!stats.public_state_prebuild_complete);
  CHECK(stats.warmup_iterations == 5);
  CHECK(stats.frozen_iterations == 0);
  CHECK(solver.get_public_state_count() <= 1);
}

TEST_CASE("solver accepts terminal roots and rejects invalid roots") {
  SolverConfig config = BaseConfig();
  config.regret_only_training = true;
  ExactGameState terminal;
  terminal.betting.stack = {0, 0};
  terminal.betting.street = StreetKind::kRiver;
  terminal.betting.player_to_act = -1;
  terminal.betting.folded_player = 1;
  terminal.betting.committed = {5, 5};

  CFRSolver solver(config, terminal);
  solver.run(1,
             ExactRange(Combo(14, SuitKind::kHearts,
                              14, SuitKind::kSpades)),
             ExactRange(Combo(13, SuitKind::kHearts,
                              13, SuitKind::kSpades)));
  CHECK(solver.get_iterations_run() == 1);
  CHECK(solver.get_expected_value(0) == doctest::Approx(5.0));

  ExactGameState invalid;
  invalid.betting.stack[0] = -1;
  CHECK_THROWS_AS(CFRSolver(config, invalid), std::invalid_argument);
}

TEST_CASE("CFR traversal respects warmup and decision-depth guards") {
  {
    SolverConfig config = BaseConfig();
    config.max_depth = 1;
    config.regret_only_training = false;
    CFRSolver solver(config);
    solver.run(3,
               ExactRange(Combo(14, SuitKind::kHearts,
                                14, SuitKind::kSpades)),
               ExactRange(Combo(13, SuitKind::kHearts,
                                13, SuitKind::kSpades)));

    const CFRSolver::TrainingRunStats stats =
        solver.get_last_training_run_stats();
    CHECK(stats.warmup_iterations == 3);
    CHECK(stats.frozen_iterations == 0);
    CHECK(solver.get_expected_value(0) == doctest::Approx(-1.0 / 12.0));
    CHECK(solver.get_cfr_update_count() > 0);
    CHECK(solver.get_traversal_stats().max_decision_depth <=
          config.max_depth);
  }

  {
    SolverConfig config = BaseConfig();
    config.max_depth = 2;
    config.regret_only_training = true;
    CFRSolver solver(config);
    solver.run(
        4,
        TwoComboRange(Combo(14, SuitKind::kHearts,
                            14, SuitKind::kSpades),
                      Combo(13, SuitKind::kHearts,
                            13, SuitKind::kSpades)),
        TwoComboRange(Combo(12, SuitKind::kClubs,
                            12, SuitKind::kDiamonds),
                      Combo(11, SuitKind::kClubs,
                            11, SuitKind::kDiamonds)));

    const CFRSolver::TrainingRunStats run_stats =
        solver.get_last_training_run_stats();
    const CFRSolver::TraversalStats traversal_stats =
        solver.get_traversal_stats();
    CHECK(run_stats.warmup_iterations == 4);
    CHECK(run_stats.frozen_iterations == 0);
    CHECK(solver.get_cfr_update_count() > 0);
    CHECK(traversal_stats.max_decision_depth == config.max_depth - 1);
  }
}

TEST_CASE("regret matching handles uniform, normalization, and CFR+ clipping") {
  {
    StoreFixture fixture;
    double probabilities[3] = {};
    fixture.store.regret_matching_or_uniform(
        std::nullopt, 3, RegretLoadMode::kPlain,
        absl::Span<double>(probabilities));
    CHECK(probabilities[0] == doctest::Approx(1.0 / 3.0));
    CHECK(probabilities[1] == doctest::Approx(1.0 / 3.0));
    CHECK(probabilities[2] == doctest::Approx(1.0 / 3.0));

    ActionBlock block = CreateBlock(fixture, 3);
    const RegretUpdateOptions plain{RegretUpdateMode::kPlain, false};
    block.add_cfr_plus_regret(1, 2.0f, plain);
    block.add_cfr_plus_regret(2, 6.0f, plain);
    block.regret_matching(RegretLoadMode::kPlain,
                          absl::Span<double>(probabilities));
    CHECK(probabilities[0] == doctest::Approx(0.0));
    CHECK(probabilities[1] == doctest::Approx(0.25));
    CHECK(probabilities[2] == doctest::Approx(0.75));
  }

  {
    StoreFixture fixture;
    ActionBlock block = CreateBlock(fixture, 2);
    const RegretUpdateOptions plain{RegretUpdateMode::kPlain, false};
    block.add_cfr_plus_regret(0, 1.0f, plain);
    block.add_cfr_plus_regret(0, -3.0f, plain);
    block.add_cfr_plus_regret(0, 1.0f, plain);
    block.add_cfr_plus_regret(1, 0.5f, plain);
    double probabilities[2] = {};
    block.regret_matching(RegretLoadMode::kPlain,
                          absl::Span<double>(probabilities));
    CHECK(probabilities[0] == doctest::Approx(2.0 / 3.0));
    CHECK(probabilities[1] == doctest::Approx(1.0 / 3.0));
  }
}

TEST_CASE("average strategy is uniform when empty and normalized otherwise") {
  StoreFixture fixture;
  ActionBlock block = CreateBlock(fixture, 2);
  double probabilities[2] = {};
  block.average_strategy(false, 0.5, absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.5));
  CHECK(probabilities[1] == doctest::Approx(0.5));

  const double strategy[2] = {0.25, 0.75};
  block.add_average_strategy(absl::Span<const double>(strategy), 4.0,
                             RegretUpdateMode::kPlain);
  block.average_strategy(false, 0.5, absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.25));
  CHECK(probabilities[1] == doctest::Approx(0.75));
}

TEST_CASE("frozen strategy lookup matches mutable slab lookup") {
  StoreFixture fixture;
  fixture.storage.mutable_tables->betting_nodes[0].action_count = 2;
  ActionBlock block = CreateBlock(fixture, 2);
  const RegretUpdateOptions plain{RegretUpdateMode::kPlain, false};
  block.add_cfr_plus_regret(0, 1.0f, plain);
  block.add_cfr_plus_regret(1, 3.0f, plain);

  REQUIRE(fixture.store.prebuild_frozen_info_set_action_offsets());
  const std::optional<ActionBlock> frozen_block =
      fixture.store.find_frozen(0, 0, 2);
  REQUIRE(frozen_block.has_value());

  double expected[2] = {};
  double actual[2] = {};
  block.regret_matching(RegretLoadMode::kPlain, absl::Span<double>(expected));
  frozen_block->regret_matching(RegretLoadMode::kPlain,
                                absl::Span<double>(actual));
  CHECK(actual[0] == doctest::Approx(expected[0]));
  CHECK(actual[1] == doctest::Approx(expected[1]));
}

}  // namespace
}  // namespace poker
