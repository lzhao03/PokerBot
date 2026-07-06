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
  poker::CheckStreetOnlyPublicBucketsEnterFrozenParallelPhase();
  return 0;
}
