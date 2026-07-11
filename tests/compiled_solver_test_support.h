#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cfr_solver.h"
#include "src/combo.h"
#include "src/game_rules.h"
#include "src/hand_range.h"

namespace poker {

namespace test {

using S = SuitKind;

inline ComboId Hand(int r0, S s0, int r1, S s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

inline HandRange ExactRange(ComboId hand) {
  HandRange range;
  range.add_combo(hand, 1.0);
  return range;
}

inline SolverConfig CompiledConfig() {
  SolverConfig config;
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {1.0};
  config.chance_samples = 1;
  config.max_depth = 0;
  config.max_info_sets = 0;
  config.max_public_states = 0;
  config.num_training_threads = 1;
  config.regret_only_training = false;
  return config;
}

inline ExactPublicState TurnRoot() {
  const BettingRules rules{2};
  ExactPublicState state = MakeInitialState(rules, {8, 8}, {1, 2});
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      MakeCardId(2, S::kHearts),
      MakeCardId(7, S::kDiamonds),
      MakeCardId(9, S::kClubs),
  };
  state = ApplyChance(state, flop, rules);
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck, 0});
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck, 0});
  const std::array<CardId, 1> turn = {MakeCardId(4, S::kSpades)};
  return ApplyChance(state, turn, rules);
}

inline const ComboId kPlayerAHand =
    Hand(14, S::kHearts, 14, S::kSpades);
inline const ComboId kPlayerBHand =
    Hand(13, S::kClubs, 13, S::kDiamonds);
inline constexpr uint32_t kTrainingSeed = 0x5eed1234u;
inline constexpr uint32_t kEvaluationSeed = 0x1234abcdu;
inline constexpr int kTrainingIterations = 6;

struct InfoSetOffset {
  NodeId node_id = kInvalidNodeId;
  PrivateObservationId private_observation = 0;
  uint32_t action_offset = StrategyTables::kInvalidActionOffset;

  friend bool operator==(const InfoSetOffset&, const InfoSetOffset&) = default;
};

struct SolverSnapshot {
  NodeId root_node_id = kInvalidNodeId;
  size_t node_count = 0;
  size_t betting_node_count = 0;
  size_t action_transition_count = 0;
  size_t chance_transition_count = 0;
  size_t info_set_count = 0;
  size_t action_entry_count = 0;
  std::vector<GameAction> root_actions;
  std::vector<InfoSetOffset> selected_info_set_offsets;
  std::vector<float> regrets;
  std::vector<float> strategy_sums;
};

struct StorageIdentity {
  const StrategyTables* mutable_tables = nullptr;
  const StrategyTables* frozen_tables = nullptr;
  const CfrState* cfr_state = nullptr;
};

}  // namespace test

struct CFRSolverTestAccess {
  static NodeId prepare(CFRSolver& solver,
                        const HandRange& a_spec,
                        const HandRange& b_spec) {
    const TrainingRange a_range = BuildTrainingRange(a_spec);
    const TrainingRange b_range = BuildTrainingRange(b_spec);
    const TrainingRangeView a_view(a_range);
    const TrainingRangeView b_view(b_range);
    const auto root = solver.graph_builder_.get_or_create_node(
        solver.initial_state_);
    if (!root.has_value() ||
        !solver.prepare_prebuilt_training(*root, solver.config_.max_depth,
                                          a_view, b_view)) {
      throw std::logic_error("failed to prepare complete solver storage");
    }
    return *root;
  }

  static void freeze(CFRSolver& solver) {
    solver.storage_.freeze();
  }

  static void seed(CFRSolver& solver, uint32_t value) {
    solver.rng_.seed(value);
  }

  static void run_growing(CFRSolver& solver,
                          int iterations,
                          uint32_t seed_value,
                          const HandRange& a_spec,
                          const HandRange& b_spec) {
    const TrainingRange a_range = BuildTrainingRange(a_spec);
    const TrainingRange b_range = BuildTrainingRange(b_spec);
    TrainingRangeView a_view(a_range);
    TrainingRangeView b_view(b_range);
    RangeSampler sampler(a_range, b_range);
    solver.rng_.seed(seed_value);
    solver.run_growing_iterations(
        iterations, solver.tables().graph.root, sampler, a_view, b_view,
        solver.config_.max_depth);
  }

  static void run_fixed(CFRSolver& solver,
                        int iterations,
                        uint32_t seed_value,
                        const HandRange& a_spec,
                        const HandRange& b_spec) {
    const TrainingRange a_range = BuildTrainingRange(a_spec);
    const TrainingRange b_range = BuildTrainingRange(b_spec);
    const RangeSampler sampler(a_range, b_range);
    solver.rng_.seed(seed_value);
    solver.run_fixed_storage_iterations(
        iterations, 1, solver.tables().graph.root,
        solver.initial_state_.board, sampler, a_range, b_range);
  }

  static uint32_t first_worker_seed(uint32_t parent_seed) {
    std::mt19937 rng(parent_seed);
    std::uniform_int_distribution<unsigned int> distribution;
    return distribution(rng);
  }

  static test::StorageIdentity storage_identity(const CFRSolver& solver) {
    return {
        solver.storage_.mutable_tables.get(),
        solver.storage_.frozen_tables.get(),
        solver.storage_.cfr_state.get(),
    };
  }

  static const PublicGraph& public_graph(const CFRSolver& solver) {
    return solver.tables().graph;
  }

  static std::shared_ptr<CfrState> copy_cfr_state(
      const CFRSolver& solver) {
    return std::make_shared<CfrState>(*solver.storage_.cfr_state);
  }

  static void bind_frozen(CFRSolver& target,
                          const CFRSolver& source,
                          std::shared_ptr<CfrState> state) {
    target.storage_.bind_frozen(source.storage_.frozen_tables,
                                std::move(state));
  }

  static test::SolverSnapshot snapshot(const CFRSolver& solver,
                                       ComboId a_hand,
                                       ComboId b_hand) {
    const StrategyTables& tables = solver.tables();
    const PublicGraph& graph = tables.graph;
    const CfrState& state = solver.storage_.cfr_state_ref();
    test::SolverSnapshot result;
    result.root_node_id = graph.root;
    result.node_count = graph.nodes.size();
    result.betting_node_count = graph.betting_nodes.size();
    result.action_transition_count = std::count_if(
        graph.action_children.begin(), graph.action_children.end(),
        [&](NodeId child) { return child < graph.nodes.size(); });
    result.chance_transition_count = graph.chance_children.size();
    result.info_set_count = tables.info_set_count;
    result.action_entry_count = state.regret_sum.size();
    result.regrets.assign(state.regret_sum.begin(), state.regret_sum.end());
    result.strategy_sums.assign(state.strategy_sum.begin(),
                                state.strategy_sum.end());

    if (graph.root >= graph.nodes.size()) {
      return result;
    }
    const auto& root = graph.nodes[graph.root];
    if (root.betting_node_id >= graph.betting_nodes.size()) {
      return result;
    }
    const auto& root_betting = graph.betting_nodes[root.betting_node_id];
    for (uint8_t i = 0; i < root_betting.action_count; ++i) {
      result.root_actions.push_back(
          graph.betting_edges[root_betting.action_begin + i].action);
    }

    std::vector<NodeId> selected_nodes = {graph.root};
    for (uint8_t i = 0; i < root_betting.action_count; ++i) {
      const NodeId child = graph.action_child(graph.root, i);
      if (child != kInvalidNodeId) {
        selected_nodes.push_back(child);
      }
    }
    for (NodeId node_id : selected_nodes) {
      const auto offset = selected_offset(solver, node_id, a_hand, b_hand);
      if (offset.has_value()) {
        result.selected_info_set_offsets.push_back(*offset);
      }
    }
    return result;
  }

  static double cumulative_utility(const CFRSolver& solver) {
    return solver.cfr_state().cumulative_root_utility;
  }

 private:
  static std::optional<test::InfoSetOffset> selected_offset(
      const CFRSolver& solver,
      NodeId node_id,
      ComboId a_hand,
      ComboId b_hand) {
    const StrategyTables& tables = solver.tables();
    const PublicGraph& graph = tables.graph;
    if (node_id >= graph.nodes.size() ||
        node_id >= tables.frozen_info_set_ranges.size()) {
      return std::nullopt;
    }
    const auto& node = graph.nodes[node_id];
    if (node.betting_node_id >= graph.betting_nodes.size()) {
      return std::nullopt;
    }
    const auto& betting = graph.betting_nodes[node.betting_node_id];
    const int player = betting.state.player_to_act;
    if (betting.kind != PublicGraph::NodeKind::kDecision ||
        !IsPlayer(player)) {
      return std::nullopt;
    }
    const ComboId hand = player == 0 ? a_hand : b_hand;
    const PrivateObservationId observation = private_observation_for_runout(
        hand, solver.initial_state_.board, node.public_observation);
    const auto& range = tables.frozen_info_set_ranges[node_id];
    const auto first = tables.frozen_info_set_entries.begin() + range.begin;
    const auto last = first + range.count;
    const auto entry = std::lower_bound(
        first, last, observation,
        [](const StrategyTables::FrozenInfoSetEntry& candidate,
           PrivateObservationId value) {
          return candidate.private_observation < value;
        });
    if (entry == last || entry->private_observation != observation) {
      return std::nullopt;
    }
    return test::InfoSetOffset{node_id, observation, entry->action_offset};
  }
};

}  // namespace poker
