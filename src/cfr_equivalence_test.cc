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
  std::vector<int> action_ids;
  std::vector<float> regrets;
  std::vector<float> strategies;
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
                tables.chance_child_entries.size());
    for (const StrategyTables::PublicStateRow& row :
         tables.public_state_rows) {
      uint64_t hash = 0;
      hash = Mix(hash, row.betting_history_id);
      hash = Mix(hash, row.public_bucket);
      hash = Mix(hash, row.is_terminal);
      hash = Mix(hash, row.is_chance_node);
      hash = Mix(hash, static_cast<uint64_t>(row.player_to_act + 2));
      hash = Mix(hash, row.action_count);
      hash = Mix(hash, row.chance_child_offset);
      hash = Mix(hash, row.chance_child_count);
      hash = Mix(hash, row.betting.pot);
      hash = Mix(hash, row.betting.stack[0]);
      hash = Mix(hash, row.betting.stack[1]);
      hash = Mix(hash, row.board.mask);
      for (uint8_t i = 0; i < row.action_count; ++i) {
        hash = Mix(hash, row.action_ids[static_cast<size_t>(i)]);
        hash = Mix(hash, row.action_child_ids[static_cast<size_t>(i)]);
      }
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
        solver.public_graph_.get_or_create_row(solver.initial_state_);
    REQUIRE(maybe_root.has_value());
    const uint32_t root_id = *maybe_root;
    REQUIRE(solver.prepare_prebuilt_training(root_id, solver.config_.max_depth,
                                             a_view, b_view));

    if (freeze_storage) {
      solver.storage_.freeze();
      solver.require_frozen_children_ = true;
    }

    const StrategyTables::PublicStateRow& root_row =
        solver.tables().public_state_rows[root_id];
    const CFRSolver::NodeRef root{
        root_id,
        root_row.board,
    };
    CFRSolver::NodeGraph graph(
        solver,
        freeze_storage ? CFRSolver::NodeGraphMode::kRequirePresent
                       : CFRSolver::NodeGraphMode::kGrow);
    CFRSolver::TraversalScratch scratch(32);
    const CFRSolver::TraversalDeal deal{{
        CFRSolver::PrivateCards::FromCombo(a_combo),
        CFRSolver::PrivateCards::FromCombo(b_combo),
    }};
    CFRSolver::TraversalOptions options =
        solver.traversal_options(0, solver.config_.max_depth);
    CFRSolver::TraversalContext ctx(deal, options, scratch, std::cref(a_view),
                                    std::cref(b_view));

    const double value = solver.cfr(root, ctx, graph);
    const MutableCumulativeArrays& arrays = solver.storage_.cumulative_ref();
    return SolverSnapshot{
        value,
        solver.cfr_update_count_,
        solver.traversal_stats_,
        GraphFingerprint(solver),
        solver.tables().action_ids,
        std::vector<float>(arrays.cumulative_regrets.begin(),
                           arrays.cumulative_regrets.end()),
        std::vector<float>(arrays.cumulative_strategies.begin(),
                           arrays.cumulative_strategies.end()),
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
  CHECK(mutable_run.action_ids == frozen_run.action_ids);
  CHECK(mutable_run.regrets == frozen_run.regrets);
  CHECK(mutable_run.strategies == frozen_run.strategies);
}

}  // namespace
}  // namespace poker
