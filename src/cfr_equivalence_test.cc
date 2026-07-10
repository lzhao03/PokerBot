#include "src/cfr_solver.h"

#include "doctest/doctest.h"
#include "src/combo.h"

#include <cstdint>
#include <optional>
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

SolverConfig EquivalenceConfig() {
  SolverConfig config;
  config.starting_stack_size = 12;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.max_depth = 1;
  config.max_public_states = 5000;
  config.max_info_sets = 5000;
  config.chance_samples = 1;
  config.regret_only_training = false;
  return config;
}

struct SolverSnapshot {
  double value = 0.0;
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
    out.reserve(tables.public_state_rows.size() +
                tables.chance_child_entries.size() +
                tables.betting_nodes.size() + tables.betting_edges.size());
    for (size_t row_id = 0; row_id < tables.public_state_rows.size();
         ++row_id) {
      const StrategyTables::PublicStateRow& row =
          tables.public_state_rows[row_id];
      uint64_t hash = 0;
      hash = Mix(hash, row.betting_node_id);
      hash = Mix(hash, row.public_bucket);
      hash = Mix(hash, row.action_child_offset);
      hash = Mix(hash, row.chance_child_offset);
      hash = Mix(hash, row.chance_child_count);
      const auto& node = tables.betting_nodes[row.betting_node_id];
      for (int action = 0; action < node.action_count; ++action) {
        const uint32_t child_id =
            tables.action_child(static_cast<uint32_t>(row_id), action)
                .value_or(kInvalidPublicStateId);
        hash = Mix(hash, child_id);
      }
      out.push_back(hash);
    }
    for (const StrategyTables::BettingNode& node : tables.betting_nodes) {
      uint64_t hash = 0;
      hash = Mix(hash, node.action_begin);
      hash = Mix(hash, node.action_count);
      hash = Mix(hash, node.chance_child);
      hash = Mix(hash, static_cast<uint64_t>(node.kind));
      hash = Mix(hash,
                 static_cast<uint64_t>(node.state.player_to_act + 2));
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
      out.push_back(Mix(entry.outcome_id, entry.public_state_id));
    }
    return out;
  }

  static SolverSnapshot RunPrepared(bool freeze_storage) {
    const ComboId a_combo =
        Combo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
    const ComboId b_combo =
        Combo(13, SuitKind::kHearts, 13, SuitKind::kSpades);
    const HandRange a_spec = ExactRange(a_combo);
    const HandRange b_spec = ExactRange(b_combo);
    const TrainingRange a_range = BuildTrainingRange(a_spec);
    const TrainingRange b_range = BuildTrainingRange(b_spec);
    TrainingRangeView a_view(a_range);
    TrainingRangeView b_view(b_range);

    CFRSolver solver(EquivalenceConfig());
    const std::optional<uint32_t> maybe_root =
        solver.graph_builder_.get_or_create_row(solver.initial_state_);
    REQUIRE(maybe_root.has_value());
    const uint32_t root_id = *maybe_root;
    REQUIRE(solver.prepare_prebuilt_training(root_id, solver.config_.max_depth,
                                             a_view, b_view));

    if (freeze_storage) {
      solver.storage_.freeze();
    }

    const CFRSolver::NodeRef root{
        root_id,
        solver.initial_state_.board,
    };
    CFRSolver::TraversalScratch scratch(32);
    const CFRSolver::Deal deal{
        {a_combo, b_combo},
        ComboMask(a_combo) | ComboMask(b_combo),
    };
    CFRSolver::TraversalOptions options =
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
    const MutableCumulativeArrays& arrays = solver.storage_.cumulative_ref();
    return SolverSnapshot{
        value,
        solver.cfr_update_count_,
        solver.traversal_stats_,
        GraphFingerprint(solver),
        std::vector<float>(arrays.cumulative_regrets.begin(),
                           arrays.cumulative_regrets.end()),
        std::vector<float>(arrays.cumulative_strategies.begin(),
                           arrays.cumulative_strategies.end()),
    };
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

    const CardMask flop_mask = CardBit(MakeCardId(14, SuitKind::kHearts));
    CFRSolver::RangeScratchFrame& flop_scratch = scratch.frame(1);
    const TrainingRangeView& after_flop =
        root_frame.ranges[0]->copy_without_mask_into(
            flop_mask, flop_scratch.filtered_ranges[0]);

    CFRSolver::TraversalFrame turn_frame = root_frame;
    turn_frame.scratch_depth = 1;
    turn_frame.ranges[0] = &after_flop;

    const CardMask turn_mask = CardBit(MakeCardId(13, SuitKind::kHearts));
    CFRSolver::RangeScratchFrame& turn_scratch = scratch.frame(2);
    const TrainingRangeView& after_turn =
        turn_frame.ranges[0]->copy_without_mask_into(
            turn_mask, turn_scratch.filtered_ranges[0]);

    return RangeScratchProbe{
        root.size(),
        after_flop.size(),
        after_turn.size(),
        after_flop.size(),
    };
  }
};

namespace {

TEST_CASE("prepared mutable and frozen traversal produce the same storage") {
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

TEST_CASE("nested chance range filtering does not overwrite parent range") {
  const RangeScratchProbe probe =
      CFRSolverTestAccess::ProbeNestedChanceScratch();

  CHECK(probe.root_size == 3);
  CHECK(probe.after_flop_size == 2);
  CHECK(probe.after_turn_size == 1);
  CHECK(probe.after_flop_size_after_turn == 2);
}

}  // namespace
}  // namespace poker
