#include "src/cfr_solver.h"
#include "src/combo.h"
#include "src/hand_range.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace poker {

#if !POKER_COARSE_PUBLIC_BUCKETS
#error "cfr_public_bucket_test must be compiled with coarse public buckets"
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

  static uint32_t CompactChanceChild(CFRSolver& solver,
                                     uint32_t public_state_id,
                                     absl::Span<const CardId> cards) {
    std::optional<uint32_t> child_id =
        solver.get_or_create_chance_child_public_state(public_state_id, cards);
    if (!child_id.has_value()) {
      throw std::runtime_error("chance child was not created");
    }
    return *child_id;
  }

  static void ClearCompactChanceChildren(CFRSolver& solver) {
    solver.mutable_tables_->public_chance_child_ids.clear();
  }

  static bool PrebuildPublicStates(CFRSolver& solver,
                                   uint32_t root_public_state_id,
                                   int max_depth) {
    return solver.prebuild_public_state_rows(root_public_state_id, max_depth);
  }

  static void FreezeTables(CFRSolver& solver) {
    solver.frozen_tables_ = solver.mutable_tables_;
    solver.mutable_tables_.reset();
    solver.frozen_ = true;
    solver.require_frozen_children_ = true;
  }

  static std::optional<uint32_t> FrozenChanceChildOptional(
      CFRSolver& solver,
      uint32_t public_state_id,
      const GameState& state,
      absl::Span<const CardId> cards) {
    const CompactPublicState parent =
        solver.compact_public_state_from_game_state(state);
    return solver.chance_child_public_state(
        public_state_id, solver.game_tree_->apply_chance(parent, cards), cards);
  }

  static uint32_t RequiredChanceChild(CFRSolver& solver,
                                      uint32_t public_state_id,
                                      const GameState& state,
                                      absl::Span<const CardId> cards) {
    const CompactPublicState parent =
        solver.compact_public_state_from_game_state(state);
    return solver.required_chance_child_public_state(
        public_state_id, solver.game_tree_->apply_chance(parent, cards), cards);
  }

  static double EvaluateExactBoard(CFRSolver& solver,
                                   const GameState& state,
                                   ComboId player_a_combo,
                                   ComboId player_b_combo) {
    const std::optional<uint32_t> public_state_id =
        solver.get_or_create_public_state_row(state);
    if (!public_state_id.has_value()) {
      throw std::runtime_error("public state was not created");
    }
    return solver.evaluate_strategy_node(
        *public_state_id, solver.compact_public_state_from_game_state(state),
        CFRSolver::PrivateCards::FromCombo(player_a_combo),
        CFRSolver::PrivateCards::FromCombo(player_b_combo));
  }

  static double EvaluatePublicState(CFRSolver& solver,
                                    uint32_t public_state_id,
                                    const GameState& state,
                                    ComboId player_a_combo,
                                    ComboId player_b_combo) {
    return solver.evaluate_strategy_node(
        public_state_id, solver.compact_public_state_from_game_state(state),
        CFRSolver::PrivateCards::FromCombo(player_a_combo),
        CFRSolver::PrivateCards::FromCombo(player_b_combo));
  }

  static uint32_t CompactPublicStateBettingHistoryId(
      const CFRSolver& solver,
      uint32_t public_state_id) {
    return solver.frozen_tables_->public_state_rows[public_state_id]
        .betting_history_id;
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

GameState ChanceState() {
  GameState state = BettingState(8, 18, 23, 4, 4);
  state.history.push_back({ActionKind::kCall, 0, 0});
  state.history.push_back({ActionKind::kCheck, 0, 1});
  return state;
}

GameState FlopChanceState(CardId first, CardId second, CardId third) {
  GameState state = PublicState(StreetKind::kFlop, first, second, third);
  state.player_to_act = 1;
  state.history.push_back({ActionKind::kCheck, 0, 0});
  state.history.push_back({ActionKind::kCheck, 0, 1});
  return state;
}

ComboId ExactCombo(int first_rank,
                   SuitKind first_suit,
                   int second_rank,
                   SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

GameState TerminalRiverState(CardId first,
                             CardId second,
                             CardId third,
                             CardId fourth,
                             CardId fifth) {
  GameState state = PublicState(StreetKind::kRiver, first, second, third,
                                fourth, fifth);
  state.stack[0] = 0;
  state.stack[1] = 0;
  state.pot = 20;
  state.all_in = true;
  state.folded_player = -1;
  state.player_to_act = 0;
  state.player_contribution = {10, 10};
  state.player_contribution_count = 2;
  return state;
}

void CheckTextureBucketTerminalUtilityUsesExactBoard() {
  SolverConfig config;
  CFRSolver solver(config);

  const ComboId aces = ExactCombo(14, SuitKind::kHearts,
                                  14, SuitKind::kSpades);
  const ComboId kings = ExactCombo(13, SuitKind::kClubs,
                                   13, SuitKind::kDiamonds);
  const GameState aces_win =
      TerminalRiverState(MakeCardId(2, SuitKind::kHearts),
                         MakeCardId(7, SuitKind::kDiamonds),
                         MakeCardId(9, SuitKind::kClubs),
                         MakeCardId(11, SuitKind::kSpades),
                         MakeCardId(12, SuitKind::kDiamonds));
  const GameState kings_win =
      TerminalRiverState(MakeCardId(3, SuitKind::kHearts),
                         MakeCardId(8, SuitKind::kDiamonds),
                         MakeCardId(10, SuitKind::kClubs),
                         MakeCardId(12, SuitKind::kSpades),
                         MakeCardId(13, SuitKind::kHearts));

  Expect(CFRSolverRegretTestPeer::PublicBucket(solver, aces_win) ==
             CFRSolverRegretTestPeer::PublicBucket(solver, kings_win),
         "fixture boards should share one texture bucket");
  Expect(CFRSolverRegretTestPeer::PublicStateId(solver, aces_win) ==
             CFRSolverRegretTestPeer::PublicStateId(solver, kings_win),
         "same texture bucket should reuse one public row");

  const double first_value =
      CFRSolverRegretTestPeer::EvaluateExactBoard(solver, aces_win, aces,
                                                  kings);
  const double second_value =
      CFRSolverRegretTestPeer::EvaluateExactBoard(solver, kings_win, aces,
                                                  kings);
  Expect(std::abs(first_value - 10.0) < 0.000001,
         "first exact board should value aces as winning");
  Expect(std::abs(second_value + 10.0) < 0.000001,
         "second exact board should value kings as winning");
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

void CheckFrozenChanceLookupCoversTextureBuckets() {
  SolverConfig config;
  CFRSolver solver(config);
  const GameState state = ChanceState();
  const uint32_t public_id =
      CFRSolverRegretTestPeer::CompactPublicStateId(solver, state);
  Expect(CFRSolverRegretTestPeer::PrebuildPublicStates(solver, public_id, 0),
         "texture prebuild should complete from a chance node");
  CFRSolverRegretTestPeer::ClearCompactChanceChildren(solver);
  CFRSolverRegretTestPeer::FreezeTables(solver);

  const std::array<CardId, 3> rainbow = {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs),
  };
  const std::array<CardId, 3> paired = {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(2, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs),
  };
  const std::array<CardId, 3> monotone = {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kHearts),
      MakeCardId(11, SuitKind::kHearts),
  };

  Expect(CFRSolverRegretTestPeer::FrozenChanceChildOptional(
             solver, public_id, state, rainbow)
             .has_value(),
         "frozen lookup should contain rainbow flop texture");
  Expect(CFRSolverRegretTestPeer::FrozenChanceChildOptional(
             solver, public_id, state, paired)
             .has_value(),
         "frozen lookup should contain paired flop texture");
  Expect(CFRSolverRegretTestPeer::FrozenChanceChildOptional(
             solver, public_id, state, monotone)
             .has_value(),
         "frozen lookup should contain monotone flop texture");

  const size_t public_states_before_eval = solver.get_public_state_count();
  const double chance_value = CFRSolverRegretTestPeer::EvaluatePublicState(
      solver, public_id, state,
      ExactCombo(14, SuitKind::kHearts, 14, SuitKind::kSpades),
      ExactCombo(13, SuitKind::kClubs, 13, SuitKind::kDiamonds));
  Expect(std::isfinite(chance_value),
         "frozen chance evaluation should use row-local chance children");
  Expect(solver.get_public_state_count() == public_states_before_eval,
         "frozen chance evaluation should not allocate public states");
}

void CheckCoarseFrozenChanceLookupCoversEquivalentParentBoards() {
  SolverConfig config;
  CFRSolver solver(config);
  const GameState representative = FlopChanceState(
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs));
  const uint32_t public_id =
      CFRSolverRegretTestPeer::CompactPublicStateId(solver, representative);
  Expect(CFRSolverRegretTestPeer::PrebuildPublicStates(solver, public_id, 0),
         "flop chance prebuild should complete");
  CFRSolverRegretTestPeer::FreezeTables(solver);

  const PublicBucketId bucket =
      CFRSolverRegretTestPeer::PublicBucket(solver, representative);
  for (int first = 0; first < kDeckCardCount; ++first) {
    for (int second = first + 1; second < kDeckCardCount; ++second) {
      for (int third = second + 1; third < kDeckCardCount; ++third) {
        const GameState alternate = FlopChanceState(
            static_cast<CardId>(first), static_cast<CardId>(second),
            static_cast<CardId>(third));
        if (alternate.board_mask == representative.board_mask ||
            CFRSolverRegretTestPeer::PublicBucket(solver, alternate) !=
                bucket) {
          continue;
        }
        for (int turn = 0; turn < kDeckCardCount; ++turn) {
          const CardId turn_card = static_cast<CardId>(turn);
          if ((alternate.board_mask & CardBit(turn_card)) != 0) {
            continue;
          }
          const std::array<CardId, 1> cards = {turn_card};
          CFRSolverRegretTestPeer::RequiredChanceChild(
              solver, public_id, alternate, cards);
          return;
        }
      }
    }
  }
  Expect(false, "fixture should find an equivalent coarse flop board");
}

void CheckTexturePublicBucketsEnterFrozenParallelPhase() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.max_depth = 1;
  config.max_public_states = 1000;
  config.num_training_threads = 2;
  config.warmup_iterations = 1;
  config.regret_only_training = true;

  HandRange player_a_range;
  player_a_range.set_uniform_range();
  HandRange player_b_range;
  player_b_range.set_uniform_range();

  CFRSolver solver(config);
  solver.run(3, player_a_range, player_b_range);

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.public_state_prebuild_complete,
         "texture public buckets should complete shallow prebuild");
  Expect(stats.action_transition_prebuild_complete,
         "texture public buckets should validate action transitions");
  Expect(stats.prebuild_action_transitions > 0,
         "texture prebuild should create action transitions");
  Expect(stats.missing_action_transitions == 0,
         "texture prebuild should not miss action transitions");
  Expect(stats.betting_history_transition_prebuild_complete,
         "texture public buckets should validate betting-history transitions");
  Expect(stats.prebuild_betting_history_transitions > 0,
         "texture prebuild should create betting-history transitions");
  Expect(stats.missing_betting_history_transitions == 0,
         "texture prebuild should not miss betting-history transitions");
  Expect(stats.chance_transition_prebuild_complete,
         "texture public buckets should validate chance transitions");
  Expect(stats.missing_chance_transitions == 0,
         "texture prebuild should not miss chance transitions");
  Expect(stats.info_set_prebuild_complete,
         "texture/private buckets should prebuild full-range infosets before freezing");
  Expect(stats.prebuild_info_sets > 0,
         "shallow coarse prebuild should create infosets");
  Expect(stats.warmup_iterations == 2,
         "alternating updates should warm both players before freezing");
  Expect(stats.frozen_iterations == 1,
         "complete shallow prebuild should enter the frozen phase");
  Expect(stats.frozen_cfr_updates > 0,
         "frozen phase should do CFR work");
  Expect(stats.frozen_cfr_updates == stats.frozen_iterations,
         "shallow frozen work should have stable CFR updates per hand");
  Expect(static_cast<int64_t>(solver.get_public_state_count()) ==
             stats.prebuild_public_states,
         "frozen phase should not allocate public states");
}

void CheckFullDepthTexturePublicBucketsEnterFrozenParallelPhase() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.max_depth = 0;
  config.max_public_states = 10000;
  config.num_training_threads = 2;
  config.warmup_iterations = 1;
  config.regret_only_training = true;

  HandRange player_a_range;
  player_a_range.set_from_string("AA");
  HandRange player_b_range;
  player_b_range.set_from_string("KK");

  CFRSolver solver(config);
  solver.run(4, player_a_range, player_b_range);

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.public_state_prebuild_complete,
         "full-depth texture public buckets should complete prebuild");
  Expect(stats.betting_history_transition_prebuild_complete,
         "full-depth texture public buckets should validate betting histories");
  Expect(stats.action_transition_prebuild_complete,
         "full-depth texture public buckets should validate action transitions");
  Expect(stats.chance_transition_prebuild_complete,
         "full-depth texture public buckets should validate chance transitions");
  Expect(stats.info_set_prebuild_complete,
         "full-depth texture buckets should prebuild infosets");
  Expect(stats.private_bucket_prebuild_complete,
         "full-depth texture buckets should prebuild private-bucket rows");
  Expect(stats.frozen_info_set_lookup_prebuild_complete,
         "full-depth texture buckets should prebuild frozen infoset lookup");
  Expect(stats.prebuild_private_bucket_rows ==
             static_cast<int64_t>(solver.get_public_state_count()),
         "private-bucket rows should cover every public state");
  Expect(stats.prebuild_frozen_info_set_lookup_rows ==
             static_cast<int64_t>(solver.get_public_state_count()),
         "frozen infoset lookup should cover every public state");
  Expect(stats.frozen_iterations > 0,
         "full-depth texture run should enter frozen phase");
  Expect(stats.frozen_cfr_updates > stats.frozen_iterations,
         "full-depth frozen phase should do nontrivial CFR work");
  Expect(static_cast<int64_t>(solver.get_public_state_count()) ==
             stats.prebuild_public_states,
         "full-depth frozen phase should not allocate public states");
}

void CheckSingleWorkerFullDepthTexturePublicBucketsEnterFrozenPhase() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.max_depth = 0;
  config.max_public_states = 10000;
  config.num_training_threads = 1;
  config.warmup_iterations = 1;
  config.regret_only_training = true;

  HandRange player_a_range;
  player_a_range.set_from_string("AA");
  HandRange player_b_range;
  player_b_range.set_from_string("KK");

  CFRSolver solver(config);
  solver.run(4, player_a_range, player_b_range);

  const CFRSolver::TrainingRunStats stats =
      solver.get_last_training_run_stats();
  Expect(stats.public_state_prebuild_complete,
         "single-worker full-depth texture public buckets should complete prebuild");
  Expect(stats.info_set_prebuild_complete,
         "single-worker full-depth texture buckets should prebuild infosets");
  Expect(stats.private_bucket_prebuild_complete,
         "single-worker full-depth texture buckets should prebuild private buckets");
  Expect(stats.frozen_info_set_lookup_prebuild_complete,
         "single-worker full-depth texture buckets should prebuild frozen lookup");
  Expect(stats.warmup_iterations == 1,
         "single-worker frozen training should use configured warmup");
  Expect(stats.frozen_iterations == 3,
         "single-worker frozen training should report remaining frozen iterations");
  Expect(stats.warmup_cfr_updates > 0,
         "single-worker frozen training should report warmup CFR work");
  Expect(stats.frozen_cfr_updates > stats.frozen_iterations,
         "single-worker frozen training should do nontrivial frozen CFR work");
  Expect(solver.get_traversal_stats().atomic_regret_update_retries == 0,
         "single-worker frozen training should not use atomic regret updates");
  Expect(static_cast<int64_t>(solver.get_public_state_count()) ==
             stats.prebuild_public_states,
         "single-worker frozen phase should not allocate public states");
}

}  // namespace poker

int main() {
  poker::CheckTextureBucketTerminalUtilityUsesExactBoard();
  poker::CheckCoarseBettingHistoryBucketsChipState();
  poker::CheckCoarseBettingHistoryKeepsActionSlotsDistinct();
  poker::CheckCoarseLegalActionsUseAbstractBettingState();
  poker::CheckFrozenChanceLookupCoversTextureBuckets();
  poker::CheckCoarseFrozenChanceLookupCoversEquivalentParentBoards();
  poker::CheckTexturePublicBucketsEnterFrozenParallelPhase();
  poker::CheckFullDepthTexturePublicBucketsEnterFrozenParallelPhase();
  poker::CheckSingleWorkerFullDepthTexturePublicBucketsEnterFrozenPhase();
  return 0;
}
