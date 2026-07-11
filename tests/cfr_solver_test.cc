#include "src/solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

namespace poker {

struct CFRSolverTestAccess {
  static const HistoryTree& history(const CFRSolver& solver) {
    return solver.history_;
  }
  static const CfrState& state(const CFRSolver& solver) {
    return solver.state_;
  }
};

namespace {

using S = SuitKind;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

ComboRange R(ComboId hand) {
  return SingleComboRange(hand);
}

SolverConfig Config() {
  SolverConfig config;
  config.starting_stack = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  for (auto& sizes : config.bet_sizes) {
    sizes = {0.5, 1.0};
  }
  config.chance_samples = 1;
  config.max_info_sets = 500000;
  return config;
}

const ComboId kA = H(14, S::kHearts, 14, S::kSpades);
const ComboId kB = H(13, S::kClubs, 13, S::kDiamonds);

TEST_CASE("history tree stores direct rule transitions") {
  CFRSolver solver(Config());
  const HistoryTree& tree = CFRSolverTestAccess::history(solver);
  REQUIRE(tree.root < tree.nodes.size());

  for (HistoryId id = 0; id < tree.nodes.size(); ++id) {
    const HistoryNode& node = tree.nodes[id];
    if (node.kind == HistoryNodeKind::kDecision) {
      REQUIRE(node.action_count > 0);
      for (uint8_t action = 0; action < node.action_count; ++action) {
        const HistoryEdge& edge = tree.edges[node.action_begin + action];
        REQUIRE(edge.child < tree.nodes.size());
        CHECK(tree.nodes[edge.child].state ==
              ApplyAction(node.state, edge.action));
      }
    } else if (node.kind == HistoryNodeKind::kChance) {
      REQUIRE(node.chance_child < tree.nodes.size());
      CHECK(tree.nodes[node.chance_child].state ==
            AdvanceBettingStreet(node.state, BettingRules{2}));
    }
  }

  const HistoryNode& root = tree.nodes[tree.root];
  REQUIRE(root.action_count >= 2);
  CHECK(tree.edges[root.action_begin].child !=
        tree.edges[root.action_begin + 1].child);
}

TEST_CASE("training mutates only CFR state") {
  CFRSolver solver(Config());
  const size_t history_count = solver.get_history_count();
  solver.run(4, R(kA), R(kB));
  CHECK(solver.get_iterations_run() == 4);
  CHECK(solver.get_info_set_count() > 0);
  CHECK(solver.get_cfr_update_count() > 0);
  CHECK(std::isfinite(solver.get_expected_value(0)));
  CHECK(solver.get_history_count() == history_count);

  const CfrState before = CFRSolverTestAccess::state(solver);
  const int64_t updates = solver.get_cfr_update_count();
  CHECK(std::isfinite(
      solver.evaluate_strategy(kA, kB, StrategySource::kAverage)));
  CHECK(solver.get_history_count() == history_count);
  CHECK(solver.get_cfr_update_count() == updates);
  CHECK(CFRSolverTestAccess::state(solver).rows == before.rows);
  CHECK(CFRSolverTestAccess::state(solver).regret_sum == before.regret_sum);
  CHECK(CFRSolverTestAccess::state(solver).strategy_sum == before.strategy_sum);
}

TEST_CASE("infoset action rows are contiguous") {
  CFRSolver solver(Config());
  solver.run(4, R(kA), R(kB));
  const CfrState& state = CFRSolverTestAccess::state(solver);

  std::vector<InfoSetRow> rows;
  rows.reserve(state.rows.size());
  for (const auto& entry : state.rows) {
    rows.push_back(entry.second);
  }
  std::sort(rows.begin(), rows.end(), [](InfoSetRow left, InfoSetRow right) {
    return left.action_offset < right.action_offset;
  });

  size_t offset = 0;
  for (const InfoSetRow row : rows) {
    CHECK(row.action_offset == offset);
    offset += row.action_count;
  }
  CHECK(offset == state.regret_sum.size());
  CHECK(state.strategy_sum.size() == state.regret_sum.size());
}

TEST_CASE("postflop roots use full observation identity") {
  SolverConfig config = Config();
  const BettingRules rules{config.big_blind};
  ExactPublicState root = MakeInitialState(rules, {8, 8}, {1, 2});
  root.betting = ApplyAction(root.betting, {ActionKind::kCall, 2});
  root.betting = ApplyAction(root.betting, {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      MakeCardId(2, S::kHearts), MakeCardId(7, S::kDiamonds),
      MakeCardId(12, S::kClubs)};
  root = ApplyChance(root, flop, rules);

  CFRSolver solver(config, root);
  solver.run(2, R(kA), R(kB));
  const HistoryTree& tree = CFRSolverTestAccess::history(solver);
  const int player = tree.nodes[tree.root].state.player_to_act;
  const ComboId hand = player == 0 ? kA : kB;
  const PublicObservationId public_id =
      public_observation_id(root.betting.street, root.board);
  const PrivateObservationId private_id =
      private_observation_for_runout(hand, root.board, public_id);
  CHECK(CFRSolverTestAccess::state(solver).rows.contains(
      {tree.root, public_id, private_id}));
}

TEST_CASE("unsupported execution modes and caps fail explicitly") {
  SolverConfig capped = Config();
  capped.max_info_sets = 1;
  CFRSolver solver(capped);
  CHECK_THROWS_AS(solver.run(2, R(kA), R(kB)), std::runtime_error);
}

TEST_CASE("average strategy storage is optional") {
  SolverConfig config = Config();
  config.accumulate_average_strategy = false;
  CFRSolver solver(config);
  solver.run(2, R(kA), R(kB));

  CHECK(CFRSolverTestAccess::state(solver).strategy_sum.empty());
  CHECK(std::isfinite(
      solver.evaluate_strategy(kA, kB, StrategySource::kCurrent)));
  CHECK_THROWS_AS(
      solver.evaluate_strategy(kA, kB, StrategySource::kAverage),
      std::logic_error);
}

}  // namespace
}  // namespace poker
