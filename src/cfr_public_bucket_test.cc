#include "src/cfr_solver.h"
#include "src/hand_range.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace poker {

#if !POKER_STREET_ONLY_PUBLIC_BUCKETS
#error "cfr_public_bucket_test must be compiled with street-only public buckets"
#endif

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

class CFRSolverRegretTestPeer {
 public:
  static CFRSolver::PublicBucketId PublicBucket(
      const CFRSolver& solver,
      const GameState& state) {
    return solver.card_abstraction_.public_bucket(state);
  }

  static uint32_t PublicStateId(CFRSolver& solver, const GameState& state) {
    const std::optional<uint32_t> public_state_id =
        solver.get_or_create_public_state_row(state);
    if (!public_state_id.has_value()) {
      throw std::runtime_error("public state was not created");
    }
    return *public_state_id;
  }

  static bool PublicStateIsExact(CFRSolver& solver, uint32_t public_state_id) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .public_state_is_exact;
  }

  static uint32_t CompactBettingHistoryId(CFRSolver& solver,
                                          const GameState& state) {
    CompactPublicState compact =
        solver.compact_public_state_from_game_state(state);
    return solver.get_or_create_betting_history_id(compact);
  }

  static uint32_t CompactPublicStateId(CFRSolver& solver,
                                       const GameState& state) {
    CompactPublicState compact =
        solver.compact_public_state_from_game_state(state);
    const uint32_t betting_history_id =
        solver.get_or_create_betting_history_id(compact);
    std::optional<uint32_t> public_state_id =
        solver.get_or_create_public_state_row(betting_history_id,
                                              std::move(compact));
    if (!public_state_id.has_value()) {
      throw std::runtime_error("public state was not created");
    }
    return *public_state_id;
  }

  static int CompactPublicStateActionCount(const CFRSolver& solver,
                                           uint32_t public_state_id) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .action_count;
  }

  static ActionKind CompactPublicStateActionKind(const CFRSolver& solver,
                                                 uint32_t public_state_id,
                                                 int action_index) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .actions[static_cast<size_t>(action_index)]
        .kind;
  }

  static int CompactPublicStateActionAmount(const CFRSolver& solver,
                                            uint32_t public_state_id,
                                            int action_index) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .actions[static_cast<size_t>(action_index)]
        .amount;
  }

  static int CompactPublicStateActionId(const CFRSolver& solver,
                                        uint32_t public_state_id,
                                        int action_index) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .action_ids[static_cast<size_t>(action_index)];
  }

  static uint32_t CompactActionChild(CFRSolver& solver,
                                     uint32_t public_state_id,
                                     int action_index) {
    std::optional<uint32_t> child_id =
        solver.get_or_create_action_child_public_state(public_state_id,
                                                       action_index);
    if (!child_id.has_value()) {
      throw std::runtime_error("action child was not created");
    }
    return *child_id;
  }

  static std::optional<uint32_t> CompactActionChildOptional(
      CFRSolver& solver,
      uint32_t public_state_id,
      int action_index) {
    return solver.get_or_create_action_child_public_state(public_state_id,
                                                          action_index);
  }

  static void ClearCompactActionChild(CFRSolver& solver,
                                      uint32_t public_state_id,
                                      int action_index) {
    solver.mutable_tables_->public_state_rows[public_state_id]
        .action_child_ids[static_cast<size_t>(action_index)] =
        GameTree::Node::kInvalidPublicStateId;
  }

  static uint32_t CompactPublicStateBettingHistoryId(
      const CFRSolver& solver,
      uint32_t public_state_id) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .betting_history_id;
  }

  static int BettingHistoryPot(const CFRSolver& solver,
                               uint32_t betting_history_id) {
    return solver.frozen_tables_->betting_history_rows[betting_history_id].pot;
  }

  static int BettingHistoryStack(const CFRSolver& solver,
                                 uint32_t betting_history_id,
                                 int player) {
    return solver.frozen_tables_->betting_history_rows[betting_history_id]
        .stack[static_cast<size_t>(player)];
  }

  static int BettingHistoryContribution(const CFRSolver& solver,
                                        uint32_t betting_history_id,
                                        int player) {
    return solver.frozen_tables_->betting_history_rows[betting_history_id]
        .player_contributions[static_cast<size_t>(player)];
  }
};

GameState PublicState(StreetKind street,
                      CardId first,
                      CardId second,
                      CardId third,
                      CardId fourth = 0,
                      CardId fifth = 0) {
  GameState state;
  state.stack[0] = 18;
  state.stack[1] = 18;
  state.pot = 4;
  state.street = street;
  state.player_to_act = 0;
  state.player_contribution = {0, 0};
  state.player_contribution_count = 2;
  AddBoardCard(state, first);
  AddBoardCard(state, second);
  AddBoardCard(state, third);
  if (street == StreetKind::kTurn || street == StreetKind::kRiver) {
    AddBoardCard(state, fourth);
  }
  if (street == StreetKind::kRiver) {
    AddBoardCard(state, fifth);
  }
  return state;
}

GameState BettingState(int pot,
                       int stack_a,
                       int stack_b,
                       int contribution_a,
                       int contribution_b) {
  GameState state;
  state.stack[0] = stack_a;
  state.stack[1] = stack_b;
  state.pot = pot;
  state.street = StreetKind::kPreflop;
  state.player_to_act = 0;
  state.player_contribution = {contribution_a, contribution_b};
  state.player_contribution_count = 2;
  state.folded_player = -1;
  return state;
}

void CheckStreetOnlyPublicBucketsMergeBoardsByStreet() {
  SolverConfig config;
  CFRSolver solver(config);

  const GameState first_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs));
  const GameState second_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(3, SuitKind::kHearts),
                  MakeCardId(8, SuitKind::kDiamonds),
                  MakeCardId(12, SuitKind::kClubs));
  const GameState turn =
      PublicState(StreetKind::kTurn,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs),
                  MakeCardId(14, SuitKind::kSpades));

  Expect(CFRSolverRegretTestPeer::PublicBucket(solver, first_flop) == 1,
         "flop bucket should be street id 1");
  Expect(CFRSolverRegretTestPeer::PublicBucket(solver, turn) == 2,
         "turn bucket should be street id 2");
  Expect(CFRSolverRegretTestPeer::PublicBucket(solver, first_flop) ==
             CFRSolverRegretTestPeer::PublicBucket(solver, second_flop),
         "street-only buckets should merge different flops");
  Expect(CFRSolverRegretTestPeer::PublicStateId(solver, first_flop) ==
             CFRSolverRegretTestPeer::PublicStateId(solver, second_flop),
         "same-street public states should reuse the representative row");
  Expect(!CFRSolverRegretTestPeer::PublicStateIsExact(solver, 0),
         "street-only public bucket rows should be representative");
}

void CheckCoarseBettingHistoryBucketsChipState() {
  SolverConfig config;
  CFRSolver solver(config);

  const GameState first = BettingState(8, 18, 23, 4, 4);
  const GameState same_bucket = BettingState(15, 19, 22, 6, 6);
  const GameState different_pot_bucket = BettingState(16, 18, 23, 4, 4);
  const GameState different_call_bucket = BettingState(8, 18, 23, 4, 8);
  const uint32_t first_id =
      CFRSolverRegretTestPeer::CompactBettingHistoryId(solver, first);
  const uint32_t same_bucket_id =
      CFRSolverRegretTestPeer::CompactBettingHistoryId(solver, same_bucket);

  Expect(first_id == same_bucket_id,
         "coarse betting key should merge chip states in the same buckets");
  Expect(CFRSolverRegretTestPeer::BettingHistoryPot(solver, first_id) == 4,
         "coarse betting row should store pot bucket");
  Expect(CFRSolverRegretTestPeer::BettingHistoryStack(solver, first_id, 0) ==
             5,
         "coarse betting row should store effective stack bucket");
  Expect(CFRSolverRegretTestPeer::BettingHistoryStack(solver, first_id, 1) ==
             0,
         "coarse betting row should clear unused stack slot");
  Expect(CFRSolverRegretTestPeer::BettingHistoryContribution(solver, first_id,
                                                             0) == 0,
         "coarse betting row should store to-call bucket");
  Expect(CFRSolverRegretTestPeer::BettingHistoryContribution(solver, first_id,
                                                             1) == 0,
         "coarse betting row should clear unused contribution slot");
  Expect(CFRSolverRegretTestPeer::CompactBettingHistoryId(solver, first) !=
             CFRSolverRegretTestPeer::CompactBettingHistoryId(
                 solver, different_pot_bucket),
         "coarse betting key should split pot bucket boundaries");
  Expect(CFRSolverRegretTestPeer::CompactBettingHistoryId(solver, first) !=
             CFRSolverRegretTestPeer::CompactBettingHistoryId(
                 solver, different_call_bucket),
         "coarse betting key should split outstanding-call buckets");
}

void CheckCoarseBettingHistoryKeepsActionSlotsDistinct() {
  SolverConfig config;
  CFRSolver solver(config);
  const uint32_t root_id = CFRSolverRegretTestPeer::CompactPublicStateId(
      solver, BettingState(3, 99, 98, 1, 2));

  Expect(CFRSolverRegretTestPeer::CompactPublicStateActionCount(solver,
                                                                root_id) >= 2,
         "root should have at least two legal actions");

  const uint32_t first_child =
      CFRSolverRegretTestPeer::CompactActionChild(solver, root_id, 0);
  const uint32_t second_child =
      CFRSolverRegretTestPeer::CompactActionChild(solver, root_id, 1);
  Expect(CFRSolverRegretTestPeer::CompactPublicStateBettingHistoryId(
             solver, first_child) !=
             CFRSolverRegretTestPeer::CompactPublicStateBettingHistoryId(
                 solver, second_child),
         "coarse betting key should keep action slots distinct");
}

void CheckCoarseLegalActionsUseAbstractBettingState() {
  SolverConfig config;
  CFRSolver solver(config);
  const GameState first = BettingState(8, 18, 23, 4, 4);
  const GameState same_bucket = BettingState(15, 19, 22, 6, 6);
  const uint32_t first_public_id =
      CFRSolverRegretTestPeer::CompactPublicStateId(solver, first);
  const uint32_t same_bucket_public_id =
      CFRSolverRegretTestPeer::CompactPublicStateId(solver, same_bucket);

  Expect(first_public_id == same_bucket_public_id,
         "same coarse betting state should reuse the public row");

  int all_in_index = -1;
  const int action_count =
      CFRSolverRegretTestPeer::CompactPublicStateActionCount(
          solver, first_public_id);
  for (int i = 0; i < action_count; ++i) {
    if (CFRSolverRegretTestPeer::CompactPublicStateActionKind(
            solver, first_public_id, i) == ActionKind::kAllIn) {
      all_in_index = i;
      break;
    }
  }

  Expect(all_in_index >= 0, "coarse action row should include all-in");
  Expect(CFRSolverRegretTestPeer::CompactPublicStateActionAmount(
             solver, first_public_id, all_in_index) == 5,
         "coarse action row should use effective-stack bucket as all-in size");
  Expect(CFRSolverRegretTestPeer::CompactPublicStateActionAmount(
             solver, first_public_id, all_in_index) != first.stack[0],
         "coarse action row should not use first representative stack");
  Expect(CFRSolverRegretTestPeer::CompactPublicStateActionAmount(
             solver, first_public_id, all_in_index) != same_bucket.stack[0],
         "coarse action row should not use later equivalent stack");

  const uint32_t first_child =
      CFRSolverRegretTestPeer::CompactActionChild(solver, first_public_id,
                                                  all_in_index);
  const uint32_t second_child =
      CFRSolverRegretTestPeer::CompactActionChild(solver,
                                                  same_bucket_public_id,
                                                  all_in_index);
  Expect(first_child == second_child,
         "same coarse action slot should reuse the action child row");
  Expect(CFRSolverRegretTestPeer::CompactPublicStateActionId(
             solver, first_public_id, all_in_index) ==
             CFRSolverRegretTestPeer::CompactPublicStateActionId(
                 solver, same_bucket_public_id, all_in_index),
         "same coarse action slot should reuse action id");
}

void CheckCoarseActionChildUsesBettingHistoryTransition() {
  SolverConfig config;
  config.max_public_states = 2;
  CFRSolver solver(config);
  const GameState state = BettingState(8, 18, 23, 4, 4);
  const uint32_t public_id =
      CFRSolverRegretTestPeer::CompactPublicStateId(solver, state);

  int all_in_index = -1;
  const int action_count =
      CFRSolverRegretTestPeer::CompactPublicStateActionCount(solver,
                                                             public_id);
  for (int i = 0; i < action_count; ++i) {
    if (CFRSolverRegretTestPeer::CompactPublicStateActionKind(
            solver, public_id, i) == ActionKind::kAllIn) {
      all_in_index = i;
      break;
    }
  }
  Expect(all_in_index >= 0, "fixture should include all-in");

  const uint32_t child_id =
      CFRSolverRegretTestPeer::CompactActionChild(solver, public_id,
                                                  all_in_index);
  CFRSolverRegretTestPeer::ClearCompactActionChild(solver, public_id,
                                                   all_in_index);

  const std::optional<uint32_t> repeated_child_id =
      CFRSolverRegretTestPeer::CompactActionChildOptional(solver, public_id,
                                                          all_in_index);
  Expect(repeated_child_id.has_value(),
         "cached betting-history transition should recover existing public child at cap");
  Expect(*repeated_child_id == child_id,
         "cached betting-history transition should return existing public child");
}

void CheckStreetOnlyPublicBucketsEnterFrozenParallelPhase() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.max_depth = 1;
  config.max_public_states = 1000;
  config.num_training_threads = 2;
  config.warmup_iterations = 1;
  config.regret_only_training = true;

  HandRange player_a_range;
  player_a_range.set_from_string("AA");
  HandRange player_b_range;
  player_b_range.set_from_string("KK");

  CFRSolver solver(config);
  solver.run(3, player_a_range, player_b_range);

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.public_state_prebuild_complete,
         "street-only public buckets should complete shallow prebuild");
  Expect(stats.parallel_iterations == 2,
         "complete shallow prebuild should enter the frozen parallel phase");
  Expect(stats.parallel_cfr_updates > 0,
         "frozen parallel phase should do CFR work");
}

}  // namespace poker

int main() {
  poker::CheckStreetOnlyPublicBucketsMergeBoardsByStreet();
  poker::CheckCoarseBettingHistoryBucketsChipState();
  poker::CheckCoarseBettingHistoryKeepsActionSlotsDistinct();
  poker::CheckCoarseLegalActionsUseAbstractBettingState();
  poker::CheckCoarseActionChildUsesBettingHistoryTransition();
  poker::CheckStreetOnlyPublicBucketsEnterFrozenParallelPhase();
  return 0;
}
