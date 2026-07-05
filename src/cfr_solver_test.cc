#include "src/cfr_solver.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "src/continuation_value.h"
#include "src/cfr_solver_proto_adapter.h"
#include "src/hand_range.h"
#include "src/subgame_value.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace poker;

namespace poker {

SuitKind TestSuitKind(Suit suit) {
  switch (suit) {
    case Suit::DIAMONDS:
      return SuitKind::kDiamonds;
    case Suit::CLUBS:
      return SuitKind::kClubs;
    case Suit::SPADES:
      return SuitKind::kSpades;
    case Suit::HEARTS:
    default:
      return SuitKind::kHearts;
  }
}

ActionKind TestActionKind(ActionType action_type) {
  switch (action_type) {
    case ActionType::BET:
      return ActionKind::kBet;
    case ActionType::FOLD:
      return ActionKind::kFold;
    case ActionType::CALL:
      return ActionKind::kCall;
    case ActionType::RAISE:
      return ActionKind::kRaise;
    case ActionType::CHECK:
      return ActionKind::kCheck;
    case ActionType::ALL_IN:
      return ActionKind::kAllIn;
    case ActionType::NO_ACTION:
    default:
      return ActionKind::kNoAction;
  }
}

ComboId TestComboId(const Hand& hand) {
  return ComboIdFromProtoHand(hand);
}

GameState TestGameState(const BoardState& state) {
  return GameStateFromProto(state);
}

SolverConfig TestSolverConfig(const PokerConfig& config) {
  return SolverConfigFromProto(config);
}

class CFRSolverRegretTestPeer {
 public:
  static double Regret(CFRSolver& solver,
                       const GameTree::Node& node,
                       int player,
                       const Hand& hand,
                       int action_id) {
    auto public_state =
        solver.get_or_create_public_state_id(node.state);
    const CFRSolver::InfoSetRow* row =
        solver.find_info_set_row(public_state, player,
                                 TestComboId(hand));
    if (row == nullptr) {
      throw std::runtime_error("Regret info set was not created");
    }
    for (uint16_t i = 0; i < row->action_count; ++i) {
      const size_t table_index = row->action_offset + i;
      if (solver.action_ids_[table_index] == action_id) {
        return solver.cumulative_regrets_[table_index];
      }
    }
    throw std::runtime_error("Regret action was not created");
  }

  static bool SamePublicStateKey(const CFRSolver& solver,
                                 const BoardState& left,
                                 const BoardState& right) {
    const GameState left_state = TestGameState(left);
    const GameState right_state = TestGameState(right);
    return solver.make_betting_history_key(left_state) ==
               solver.make_betting_history_key(right_state) &&
           solver.card_abstraction_.public_bucket(left_state) ==
               solver.card_abstraction_.public_bucket(right_state);
  }

  static CFRSolver::PublicBucketId PublicBucket(const CFRSolver& solver,
                                                const BoardState& state) {
    return solver.card_abstraction_.public_bucket(TestGameState(state));
  }

  static CFRSolver::PrivateBucketId PrivateBucket(const CFRSolver& solver,
                                                  const BoardState& state,
                                                  const Hand& hand) {
    return solver.card_abstraction_.private_bucket(TestComboId(hand),
                                                   TestGameState(state));
  }

  static uint32_t PublicStateId(CFRSolver& solver, const BoardState& state) {
    return solver.get_or_create_public_state_id(TestGameState(state));
  }

  static uint32_t BettingHistoryId(CFRSolver& solver,
                                   const BoardState& state) {
    return solver.get_or_create_betting_history_id(TestGameState(state));
  }

  static int InfoSetOffset(CFRSolver& solver,
                           const BoardState& state,
                           int player,
                           const Hand& hand) {
    GameState native_state = TestGameState(state);
    std::vector<GameAction> legal_actions =
        solver.game_tree_->get_legal_actions(native_state);
    int action_id_buf[GameTree::kMaxActionsPerNode];
    int num_actions = 0;
    for (const GameAction& action : legal_actions) {
      action_id_buf[num_actions++] = GameTree::action_key(action);
    }
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(native_state);
    const CFRSolver::PrivateBucketId private_bucket =
        solver.card_abstraction_.private_bucket(TestComboId(hand),
                                                native_state);
    std::optional<CFRSolver::InfoSetRow> row =
        solver.get_or_create_info_set_row(
            public_state_id, player, private_bucket, action_id_buf,
            num_actions);
    if (!row.has_value()) {
      return -1;
    }
    return static_cast<int>(row->action_offset);
  }

  static bool HasPublicInfoSetSlab(CFRSolver& solver,
                                   const BoardState& state) {
    const GameState native_state = TestGameState(state);
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(native_state);
    return public_state_id < solver.public_info_set_slabs_.size() &&
           solver.public_info_set_slabs_[public_state_id] != nullptr;
  }

  static int SlabInfoSetOffset(CFRSolver& solver,
                               const BoardState& state,
                               int player,
                               const Hand& hand) {
    const GameState native_state = TestGameState(state);
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(native_state);
    const CFRSolver::PrivateBucketId private_bucket =
        solver.card_abstraction_.private_bucket(TestComboId(hand),
                                                native_state);
    const CFRSolver::InfoSetRow* row =
        solver.find_info_set_row(public_state_id, player, private_bucket);
    if (row == nullptr) {
      return -1;
    }
    return static_cast<int>(row->action_offset);
  }

  static const GameTree::Node& Root(CFRSolver& solver) {
    return solver.get_or_build_root();
  }

  static GameTree::Node& MutableRoot(CFRSolver& solver) {
    return solver.get_or_build_root();
  }

  static uint32_t NodeBettingHistoryId(CFRSolver& solver,
                                       GameTree::Node& node) {
    return solver.get_or_create_betting_history_id(node);
  }

  static GameTree::Node& ActionChild(CFRSolver& solver,
                                     GameTree::Node& node,
                                     int action_index) {
    GameTree::Node& child =
        solver.game_tree_->create_child_node(node, action_index);
    solver.cache_action_betting_history_transition(
        node, action_index, child);
    return child;
  }

  static uint32_t ActionTransitionId(const CFRSolver& solver,
                                     uint32_t betting_history_id,
                                     int action_index) {
    if (betting_history_id >= solver.betting_history_rows_.size()) {
      return GameTree::Node::kInvalidBettingHistoryId;
    }
    return solver.betting_history_rows_[betting_history_id]
        .action_child_ids[static_cast<size_t>(action_index)];
  }

  static uint8_t BettingHistoryActionCount(const CFRSolver& solver,
                                           uint32_t betting_history_id) {
    if (betting_history_id >= solver.betting_history_rows_.size()) {
      return 0;
    }
    return solver.betting_history_rows_[betting_history_id].action_count;
  }

  static int BettingHistoryActionId(const CFRSolver& solver,
                                    uint32_t betting_history_id,
                                    int action_index) {
    if (betting_history_id >= solver.betting_history_rows_.size()) {
      return 0;
    }
    return solver.betting_history_rows_[betting_history_id]
        .action_ids[static_cast<size_t>(action_index)];
  }

  static int BettingHistoryPot(const CFRSolver& solver,
                               uint32_t betting_history_id) {
    if (betting_history_id >= solver.betting_history_rows_.size()) {
      return -1;
    }
    return solver.betting_history_rows_[betting_history_id].pot;
  }

  static std::pair<uint32_t, CFRSolver::TraversalStats>
  SharedViewActionTransitionChildId(CFRSolver& source,
                                    const GameTree::Node& parent,
                                    const GameTree::Node& child,
                                    int action_index) {
    absl::flat_hash_map<CFRSolver::BettingHistoryKey, uint32_t,
                        CFRSolver::BettingHistoryKeyHash>
        empty_betting_history_ids;
    CFRSolver::StrategyTablesView tables = source.strategy_tables_view();
    tables.betting_history_ids = &empty_betting_history_ids;

    CFRSolver worker(source.config_);
    worker.strategy_tables_view_ = &tables;
    GameTree::Node parent_copy = parent;
    GameTree::Node child_copy = child;
    child_copy.betting_history_id = GameTree::Node::kInvalidBettingHistoryId;
    worker.cache_action_betting_history_transition(parent_copy, action_index,
                                                   child_copy);
    return {child_copy.betting_history_id, worker.get_traversal_stats()};
  }

  static size_t PublicStateInfoSetListSize(CFRSolver& solver,
                                           const GameTree::Node& node,
                                           int player) {
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(node.state);
    const CFRSolver::PublicInfoSetSlab* slab =
        solver.public_info_set_slab(public_state_id);
    if (slab == nullptr) {
      return 0;
    }
    return slab->players[player].rows.size();
  }

  static double EvaluateNode(CFRSolver& solver, GameTree::Node& node,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand) {
    return solver.evaluate_strategy_node(
        node, CFRSolver::PrivateCards::FromCombo(TestComboId(player_a_hand)),
        CFRSolver::PrivateCards::FromCombo(TestComboId(player_b_hand)));
  }

  static double BestResponseNode(CFRSolver& solver, GameTree::Node& node,
                                 const Hand& player_a_hand,
                                 const Hand& player_b_hand,
                                 int best_response_player) {
    return solver.best_response_value(
        node, CFRSolver::PrivateCards::FromCombo(TestComboId(player_a_hand)),
        CFRSolver::PrivateCards::FromCombo(TestComboId(player_b_hand)),
        best_response_player);
  }

  static double BestResponseRangeNode(
      CFRSolver& solver, GameTree::Node& node,
      const Hand& best_response_hand,
      const WeightedHandRange& opponent_hands,
      int best_response_player) {
    WeightedHandRangeView opponent_view(opponent_hands);
    return solver.best_response_value_against_range(
        node, CFRSolver::PrivateCards::FromCombo(TestComboId(best_response_hand)),
        opponent_view, best_response_player);
  }

  static double Utility(CFRSolver& solver,
                        const BoardState& state,
                        const Hand& player_a_hand,
                        const Hand& player_b_hand) {
    return solver.utility(
        TestGameState(state),
        CFRSolver::PrivateCards::FromCombo(TestComboId(player_a_hand)),
        CFRSolver::PrivateCards::FromCombo(TestComboId(player_b_hand)));
  }

  static GameTree::Node& AddChild(CFRSolver& solver,
                                  GameTree::Node& node,
                                  int action_id,
                                  GameTree::Node child) {
    return solver.game_tree_->add_child(node, action_id, std::move(child));
  }

  static void SetRegret(CFRSolver& solver,
                        const BoardState& state,
                        int player,
                        const Hand& hand,
                        int action_id,
                        double regret) {
    GameState native_state = TestGameState(state);
    std::vector<GameAction> legal_actions =
        solver.game_tree_->get_legal_actions(native_state);
    int action_id_buf[GameTree::kMaxActionsPerNode];
    int num_actions = 0;
    for (const GameAction& action : legal_actions) {
      action_id_buf[num_actions++] = GameTree::action_key(action);
    }
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(native_state);
    std::optional<CFRSolver::InfoSetRow> row =
        solver.get_or_create_info_set_row(
        public_state_id, player, TestComboId(hand),
        action_id_buf, num_actions);
    if (!row.has_value()) {
      throw std::runtime_error("Could not create seeded regret row");
    }
    std::optional<size_t> action_table_index;
    for (uint16_t i = 0; i < row->action_count; ++i) {
      const size_t table_index = row->action_offset + i;
      if (solver.action_ids_[table_index] == action_id) {
        action_table_index = table_index;
        break;
      }
    }
    if (!action_table_index.has_value()) {
      throw std::runtime_error("Seeded regret action is not legal");
    }
    solver.cumulative_regrets_[*action_table_index] = regret;
  }

  static double TotalCumulativeStrategy(const CFRSolver& solver) {
    double total = 0.0;
    for (float cumulative_strategy : solver.cumulative_strategies_) {
      total += cumulative_strategy;
    }
    return total;
  }

  static std::vector<double> PlayerASampleWeights(
      const HandRange& player_a_range,
      const HandRange& player_b_range) {
    const TrainingRange player_a_training_range =
        BuildTrainingRange(player_a_range);
    const TrainingRange player_b_training_range =
        BuildTrainingRange(player_b_range);
    CFRSolver::RangeSampler sampler(
        player_a_training_range, player_b_training_range);
    return std::vector<double>(sampler.player_a_sample_weights.begin(),
                               sampler.player_a_sample_weights.end());
  }

  static TrainingRangeView ConditionRangeForAction(
      CFRSolver& solver,
      const TrainingRangeView& range,
      const BoardState& state,
      int player,
      ActionType action_type,
      int amount) {
    GameState native_state = TestGameState(state);
    std::vector<GameAction> legal_actions =
        solver.game_tree_->get_legal_actions(native_state);
    CFRSolver::ActionChoices choices;
    choices.reserve(legal_actions.size());
    size_t selected_index = legal_actions.size();
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      const GameAction& action = legal_actions[i];
      const int action_id = GameTree::action_key(action);
      choices.push_back({std::cref(action), action_id, 0.0, 0.0});
      if (action.kind == TestActionKind(action_type) &&
          action.amount == amount) {
        selected_index = i;
      }
    }
    if (selected_index == legal_actions.size()) {
      throw std::runtime_error("Selected action is not legal");
    }

    CFRSolver::ConditionedRanges conditioned_ranges;
    GameTree::Node& root = solver.get_or_build_root();
    const uint32_t public_state_id =
        solver.get_or_create_public_state_id(native_state);
    solver.condition_ranges_for_actions(
        range, root, public_state_id, player, choices,
        conditioned_ranges);
    return std::move(conditioned_ranges[selected_index]);
  }
};

}  // namespace poker

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class FixedContinuationValueProvider : public ContinuationValueProvider {
 public:
  explicit FixedContinuationValueProvider(double value) : value_(value) {}

  using ContinuationValueProvider::value;

  double value(GameTree& game_tree,
               const ContinuationContext& context) const override {
    (void)game_tree;
    ++calls_;
    saw_empty_ranges_ = !context.has_ranges();
    if (context.has_ranges()) {
      saw_ranges_ = true;
      last_player_a_range_size_ = context.player_a_range.size();
      last_player_b_range_size_ = context.player_b_range.size();
      last_player_a_range_weight_ = 0.0;
      last_player_b_range_weight_ = 0.0;
      for (size_t i = 0; i < context.player_a_range.size(); ++i) {
        last_player_a_range_weight_ += context.player_a_range.weight(i);
      }
      for (size_t i = 0; i < context.player_b_range.size(); ++i) {
        last_player_b_range_weight_ += context.player_b_range.weight(i);
      }
    }
    return value_;
  }

  int calls() const { return calls_; }
  bool saw_empty_ranges() const { return saw_empty_ranges_; }
  bool saw_ranges() const { return saw_ranges_; }
  size_t last_player_a_range_size() const {
    return last_player_a_range_size_;
  }
  size_t last_player_b_range_size() const {
    return last_player_b_range_size_;
  }
  double last_player_a_range_weight() const {
    return last_player_a_range_weight_;
  }
  double last_player_b_range_weight() const {
    return last_player_b_range_weight_;
  }

 private:
  double value_;
  mutable int calls_ = 0;
  mutable bool saw_empty_ranges_ = false;
  mutable bool saw_ranges_ = false;
  mutable size_t last_player_a_range_size_ = 0;
  mutable size_t last_player_b_range_size_ = 0;
  mutable double last_player_a_range_weight_ = 0.0;
  mutable double last_player_b_range_weight_ = 0.0;
};

class CapturingLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    messages_.append(entry.text_message().data(), entry.text_message().size());
    messages_.push_back('\n');
  }

  const std::string& messages() const { return messages_; }

 private:
  std::string messages_;
};

class ScopedLogSink {
 public:
  explicit ScopedLogSink(absl::LogSink& sink) : sink_(sink) {
    absl::AddLogSink(&sink_.get());
  }

  ~ScopedLogSink() {
    absl::RemoveLogSink(&sink_.get());
  }

 private:
  std::reference_wrapper<absl::LogSink> sink_;
};

class ScopedVLogLevel {
 public:
  explicit ScopedVLogLevel(int level)
      : previous_level_(absl::SetGlobalVLogLevel(level)) {}

  ~ScopedVLogLevel() {
    absl::SetGlobalVLogLevel(previous_level_);
  }

 private:
  int previous_level_;
};

Action MakeAction(ActionType type, int amount = 0) {
  Action action;
  action.set_action(type);
  action.set_amount(amount);
  return action;
}

GameAction MakeGameAction(ActionType type, int amount = 0) {
  return {TestActionKind(type), amount, -1};
}

int TestActionKey(ActionType type, int amount = 0) {
  return static_cast<int>(type) * 1000000 + amount;
}

double TestCfr(CFRSolver& solver,
               GameTree::Node& node,
               const Hand& player_a_hand,
               const Hand& player_b_hand,
               std::array<double, 2>& reach_probabilities,
               int iteration,
               int depth = 0,
               int max_depth = 0) {
  return solver.cfr(node, TestComboId(player_a_hand),
                    TestComboId(player_b_hand), reach_probabilities,
                    iteration, depth, max_depth);
}

std::unordered_map<int, double> StrategyActionMap(
    const CFRSolver::StrategyProfile& profile,
    size_t info_set_index = 0) {
  std::unordered_map<int, double> actions;
  const CFRSolver::StrategyInfoSet& info_set = profile.info_sets[info_set_index];
  for (size_t i = 0; i < info_set.action_ids.size(); ++i) {
    actions.emplace(info_set.action_ids[i], info_set.probabilities[i]);
  }
  return actions;
}

bool StrategyHasCombo(const CFRSolver::StrategyProfile& profile,
                      int player,
                      const Hand& hand) {
  const ComboId combo = TestComboId(hand);
  return std::any_of(
      profile.info_sets.begin(), profile.info_sets.end(),
      [&](const CFRSolver::StrategyInfoSet& info_set) {
        return info_set.key.player == player &&
               info_set.key.private_bucket == combo;
      });
}

bool StrategiesApproximatelyEqual(const CFRSolver::StrategyProfile& left,
                                  const CFRSolver::StrategyProfile& right) {
  if (left.info_sets.size() != right.info_sets.size()) {
    return false;
  }
  for (size_t i = 0; i < left.info_sets.size(); ++i) {
    const CFRSolver::StrategyInfoSet& left_info_set = left.info_sets[i];
    const CFRSolver::StrategyInfoSet& right_info_set = right.info_sets[i];
    if (left_info_set.key.public_state_id !=
            right_info_set.key.public_state_id ||
        left_info_set.key.private_bucket != right_info_set.key.private_bucket ||
        left_info_set.key.player != right_info_set.key.player ||
        left_info_set.action_ids != right_info_set.action_ids ||
        left_info_set.probabilities.size() !=
            right_info_set.probabilities.size()) {
      return false;
    }
    for (size_t action_index = 0;
         action_index < left_info_set.probabilities.size(); ++action_index) {
      if (std::abs(left_info_set.probabilities[action_index] -
                   right_info_set.probabilities[action_index]) > 0.000001) {
        return false;
      }
    }
  }
  return true;
}

Card MakeTestCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

void AddCard(BoardState& state, int rank, Suit suit) {
  *state.add_cards() = MakeTestCard(rank, suit);
}

void AddCard(GameState& state, int rank, Suit suit) {
  AddBoardCard(state, MakeCardId(rank, TestSuitKind(suit)));
}

Hand MakeHand(int first_rank, Suit first_suit, int second_rank, Suit second_suit) {
  Hand hand;
  *hand.add_cards() = MakeTestCard(first_rank, first_suit);
  *hand.add_cards() = MakeTestCard(second_rank, second_suit);
  return hand;
}

Suit TestProtoSuit(CardId card_id) {
  switch (SuitFromCardId(card_id)) {
    case SuitKind::kDiamonds:
      return Suit::DIAMONDS;
    case SuitKind::kClubs:
      return Suit::CLUBS;
    case SuitKind::kSpades:
      return Suit::SPADES;
    case SuitKind::kHearts:
      return Suit::HEARTS;
  }
  return Suit::HEARTS;
}

Hand TestHandFromCombo(ComboId combo_id) {
  const ComboInfo& combo = GetComboInfo(combo_id);
  return MakeHand(RankFromCardId(combo.card0), TestProtoSuit(combo.card0),
                  RankFromCardId(combo.card1), TestProtoSuit(combo.card1));
}

void AddHand(HandRange& range, const Hand& hand, double weight) {
  range.add_combo(TestComboId(hand), weight);
}

double TestTotalWeight(const WeightedHandRange& hands) {
  double total = 0.0;
  for (double weight : hands.weights) {
    total += weight;
  }
  return total;
}

double TestTotalWeightForRank(const WeightedHandRange& hands, int rank) {
  double total = 0.0;
  for (size_t i = 0; i < hands.size(); ++i) {
    const ComboInfo& combo = GetComboInfo(hands.combos[i]);
    if (RankFromCardId(combo.card0) == rank &&
        RankFromCardId(combo.card1) == rank) {
      total += hands.weights[i];
    }
  }
  return total;
}

BoardState InitialRootState(const PokerConfig& config) {
  int small_blind = config.small_blind() > 0 ? config.small_blind() : 1;
  int big_blind = config.big_blind() > 0 ? config.big_blind() : 2;
  int starting_stack = config.starting_stack_size();

  BoardState state;
  state.set_stack_a(starting_stack > small_blind ? starting_stack - small_blind : 0);
  state.set_stack_b(starting_stack > big_blind ? starting_stack - big_blind : 0);
  state.set_pot(small_blind + big_blind);
  state.set_folded_player(-1);
  state.set_street(Street::PREFLOP);
  state.set_all_in(false);
  state.set_player_to_act(0);
  state.add_player_contribution(small_blind);
  state.add_player_contribution(big_blind);
  return state;
}

void CheckCfrUsesLegalActions() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_stack_a(10);
  node.state.set_stack_b(10);
  node.state.set_pot(0);
  node.state.set_street(Street::FLOP);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(0);
  node.state.add_player_contribution(0);
  node.state.set_player_to_act(0);
  AddCard(node.state, 2, Suit::HEARTS);
  AddCard(node.state, 3, Suit::DIAMONDS);
  AddCard(node.state, 4, Suit::CLUBS);
  node.player_to_act = 0;
  node.add_action(MakeGameAction(ActionType::CHECK), GameTree::action_key(MakeGameAction(ActionType::CHECK)));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  Expect(strategy.size() == 1, "CFR should visit one info set");
  const auto action_probs = StrategyActionMap(strategy);
  Expect(action_probs.size() == 1, "strategy should only include legal actions");
  Expect(action_probs.count(TestActionKey(ActionType::CHECK)) == 1,
         "strategy should include check");
  Expect(action_probs.at(TestActionKey(ActionType::CHECK)) == 1.0,
         "single legal action should get probability 1");
}

void CheckCfrDistinguishesActionAmounts() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_stack_a(19);
  node.state.set_stack_b(18);
  node.state.set_pot(3);
  node.state.set_street(Street::PREFLOP);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(1);
  node.state.add_player_contribution(2);
  node.state.set_player_to_act(0);
  node.player_to_act = 0;
  node.add_action(MakeGameAction(ActionType::FOLD), GameTree::action_key(MakeGameAction(ActionType::FOLD)));
  node.add_action(MakeGameAction(ActionType::CALL, 1), GameTree::action_key(MakeGameAction(ActionType::CALL, 1)));
  node.add_action(MakeGameAction(ActionType::RAISE, 3), GameTree::action_key(MakeGameAction(ActionType::RAISE, 3)));
  node.add_action(MakeGameAction(ActionType::RAISE, 5), GameTree::action_key(MakeGameAction(ActionType::RAISE, 5)));
  node.add_action(MakeGameAction(ActionType::RAISE, 7), GameTree::action_key(MakeGameAction(ActionType::RAISE, 7)));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  Expect(strategy.size() == 1, "CFR should visit one info set");
  const auto action_probs = StrategyActionMap(strategy);
  Expect(action_probs.size() == 5, "same action type with different amounts should not collide");
  for (const auto& action_prob : action_probs) {
    Expect(std::abs(action_prob.second - 0.2) < 0.000001,
           "initial strategy should be uniform over all legal actions");
  }
}

void CheckPublicStateKeyIgnoresBoardOrder() {
  PokerConfig config;
  CFRSolver solver(TestSolverConfig(config));
  BoardState first = InitialRootState(config);
  BoardState second = InitialRootState(config);
  first.set_street(Street::FLOP);
  second.set_street(Street::FLOP);
  AddCard(first, 2, Suit::HEARTS);
  AddCard(first, 7, Suit::DIAMONDS);
  AddCard(first, 11, Suit::CLUBS);
  AddCard(second, 11, Suit::CLUBS);
  AddCard(second, 2, Suit::HEARTS);
  AddCard(second, 7, Suit::DIAMONDS);

  Expect(CFRSolverRegretTestPeer::SamePublicStateKey(solver, first, second),
         "public state key should not depend on public board card order");
  Expect(CFRSolverRegretTestPeer::PublicBucket(solver, first) ==
             CFRSolverRegretTestPeer::PublicBucket(solver, second),
         "identity public-card abstraction should canonicalize by board mask");
}

void CheckIdentityPrivateAbstractionUsesExactCombo() {
  PokerConfig config;
  CFRSolver solver(TestSolverConfig(config));
  BoardState state = InitialRootState(config);
  Hand aces = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);

  Expect(CFRSolverRegretTestPeer::PrivateBucket(solver, state, aces) ==
             TestComboId(aces),
         "identity private abstraction should use exact combo ids");
}

void CheckPublicStateIdsAreDenseAndKeyedByState() {
  PokerConfig config;
  CFRSolver solver(TestSolverConfig(config));
  BoardState root = InitialRootState(config);
  BoardState flop = InitialRootState(config);
  flop.set_street(Street::FLOP);
  AddCard(flop, 2, Suit::HEARTS);
  AddCard(flop, 7, Suit::DIAMONDS);
  AddCard(flop, 11, Suit::CLUBS);

  const uint32_t root_id =
      CFRSolverRegretTestPeer::PublicStateId(solver, root);
  const uint32_t flop_id =
      CFRSolverRegretTestPeer::PublicStateId(solver, flop);
  const uint32_t repeated_flop_id =
      CFRSolverRegretTestPeer::PublicStateId(solver, flop);

  Expect(root_id == 0, "first public state should get dense id zero");
  Expect(flop_id == 1, "different public state should get next dense id");
  Expect(repeated_flop_id == flop_id,
         "same public state key should reuse its dense id");
}

void CheckBettingHistoryIdsIgnorePublicCards() {
  PokerConfig config;
  CFRSolver solver(TestSolverConfig(config));
  BoardState first_flop = InitialRootState(config);
  first_flop.set_street(Street::FLOP);
  AddCard(first_flop, 2, Suit::HEARTS);
  AddCard(first_flop, 7, Suit::DIAMONDS);
  AddCard(first_flop, 11, Suit::CLUBS);

  BoardState second_flop = InitialRootState(config);
  second_flop.set_street(Street::FLOP);
  AddCard(second_flop, 3, Suit::HEARTS);
  AddCard(second_flop, 8, Suit::DIAMONDS);
  AddCard(second_flop, 12, Suit::CLUBS);

  const uint32_t first_betting_id =
      CFRSolverRegretTestPeer::BettingHistoryId(solver, first_flop);
  const uint32_t second_betting_id =
      CFRSolverRegretTestPeer::BettingHistoryId(solver, second_flop);
  const uint32_t first_public_id =
      CFRSolverRegretTestPeer::PublicStateId(solver, first_flop);
  const uint32_t second_public_id =
      CFRSolverRegretTestPeer::PublicStateId(solver, second_flop);

  Expect(first_betting_id == second_betting_id,
         "betting history ids should ignore public card identity");
  Expect(first_public_id != second_public_id,
         "public state ids should still distinguish public card ids");
}

void CheckBettingHistoryActionTransitionsAreCached() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node& root = CFRSolverRegretTestPeer::MutableRoot(solver);
  Expect(root.action_count > 0, "root should have legal actions");

  const uint32_t root_betting_id =
      CFRSolverRegretTestPeer::NodeBettingHistoryId(solver, root);
  Expect(CFRSolverRegretTestPeer::BettingHistoryPot(
             solver, root_betting_id) == root.state.pot,
         "betting history row should store pot metadata");
  Expect(CFRSolverRegretTestPeer::BettingHistoryActionCount(
             solver, root_betting_id) == root.action_count,
         "betting history row should store legal action count");
  Expect(CFRSolverRegretTestPeer::BettingHistoryActionId(
             solver, root_betting_id, 0) == root.actions[0].key,
         "betting history row should store legal action keys");
  for (int i = 0; i < root.action_count; ++i) {
    Expect(CFRSolverRegretTestPeer::BettingHistoryActionId(
               solver, root_betting_id, i) == root.actions[i].key,
           "betting history row should mirror every legal action key");
  }
  GameTree::Node& child =
      CFRSolverRegretTestPeer::ActionChild(solver, root, 0);
  const uint32_t child_betting_id = child.betting_history_id;

  Expect(root_betting_id != GameTree::Node::kInvalidBettingHistoryId,
         "root should cache its betting history id");
  Expect(child_betting_id != GameTree::Node::kInvalidBettingHistoryId,
         "action child should cache its betting history id");
  Expect(CFRSolverRegretTestPeer::ActionTransitionId(
             solver, root_betting_id, 0) == child_betting_id,
         "parent betting history should cache action child history id");

  GameTree::Node& repeated_child =
      CFRSolverRegretTestPeer::ActionChild(solver, root, 0);
  Expect(&repeated_child == &child,
         "action child lookup should reuse the same game tree node");
  Expect(CFRSolverRegretTestPeer::ActionTransitionId(
             solver, root_betting_id, 0) == child_betting_id,
         "reused action child should preserve cached history transition");
  Expect(solver.get_traversal_stats().betting_history_transition_misses == 1,
         "first local transition lookup should miss before caching child id");
  Expect(solver.get_traversal_stats().betting_history_transition_hits == 1,
         "second local transition lookup should hit cached child id");

  const auto shared_result =
      CFRSolverRegretTestPeer::SharedViewActionTransitionChildId(
          solver, root, child, 0);
  Expect(shared_result.first == child_betting_id,
         "shared strategy view should assign child id from transition table");
  Expect(shared_result.second.betting_history_transition_hits == 1,
         "shared transition table lookup should record a hit");
  Expect(shared_result.second.betting_history_transition_misses == 0,
         "shared transition table lookup should not fall back to key lookup");
}

void CheckSparseSlabRowsUseExactCombos() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  BoardState state = InitialRootState(config);
  Hand aces = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand kings = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);

  const int first_aces_id =
      CFRSolverRegretTestPeer::InfoSetOffset(solver, state, 0, aces);
  const int second_aces_id =
      CFRSolverRegretTestPeer::InfoSetOffset(solver, state, 0, aces);
  const int kings_id =
      CFRSolverRegretTestPeer::InfoSetOffset(solver, state, 0, kings);
  const GameTree::Node& root = CFRSolverRegretTestPeer::Root(solver);

  Expect(first_aces_id == second_aces_id,
         "sparse slab row should reuse the same exact combo");
  Expect(first_aces_id != kings_id,
         "sparse slab row should distinguish different exact combos");
  Expect(CFRSolverRegretTestPeer::PublicStateInfoSetListSize(
             solver, root, 0) == 2,
         "public-state infoset list should track created infosets");
  Expect(solver.get_info_set_count() == 2,
         "sparse slab should only create distinct info sets");
  Expect(solver.get_strategy_profile().size() == 2,
         "sparse slab rows should export through strategy profiles");
}

void CheckSparseSlabTracksInfoSets() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  BoardState state = InitialRootState(config);
  Hand aces = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand kings = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);

  CFRSolverRegretTestPeer::PublicStateId(solver, state);
  Expect(!CFRSolverRegretTestPeer::HasPublicInfoSetSlab(solver, state),
         "sparse slab should not be allocated for a public state alone");

  const int aces_id =
      CFRSolverRegretTestPeer::InfoSetOffset(solver, state, 0, aces);

  Expect(CFRSolverRegretTestPeer::HasPublicInfoSetSlab(solver, state),
         "sparse slab should be allocated when the first infoset is created");
  Expect(CFRSolverRegretTestPeer::SlabInfoSetOffset(solver, state, 0, aces) ==
             aces_id,
         "sparse slab should point at created infosets");
  Expect(CFRSolverRegretTestPeer::SlabInfoSetOffset(solver, state, 0, kings) ==
             -1,
         "sparse slab should mark missing private ids");
  const int kings_id =
      CFRSolverRegretTestPeer::InfoSetOffset(solver, state, 0, kings);
  Expect(CFRSolverRegretTestPeer::SlabInfoSetOffset(solver, state, 0, kings) ==
             kings_id,
         "sparse slab should track new infosets after it is built");
}

void CheckTerminalUtilityCacheReusesScores() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  BoardState state;
  state.set_pot(12);
  state.set_folded_player(-1);
  state.add_player_contribution(6);
  state.add_player_contribution(6);
  AddCard(state, 2, Suit::CLUBS);
  AddCard(state, 7, Suit::DIAMONDS);
  AddCard(state, 9, Suit::HEARTS);
  AddCard(state, 11, Suit::SPADES);
  AddCard(state, 12, Suit::CLUBS);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);

  double first = CFRSolverRegretTestPeer::Utility(
      solver, state, player_a_hand, player_b_hand);
  double second = CFRSolverRegretTestPeer::Utility(
      solver, state, player_a_hand, player_b_hand);
  CFRSolver::UtilityCacheStats stats = solver.get_utility_cache_stats();

  Expect(first == second, "cached utility should preserve score");
  Expect(stats.misses == 1, "first utility lookup should miss cache");
  Expect(stats.hits == 1, "second utility lookup should hit cache");
  Expect(stats.entries == 1, "cache should contain one utility entry");
}

void CheckSingletonRangeMatchesExactEvaluationAndBestResponse() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;
  node.add_action(MakeGameAction(ActionType::FOLD), GameTree::action_key(MakeGameAction(ActionType::FOLD)));
  node.add_action(MakeGameAction(ActionType::CALL, 1), GameTree::action_key(MakeGameAction(ActionType::CALL, 1)));

  auto folded_state = [](int folded_player) {
    GameState state;
    state.set_pot(10);
    state.set_folded_player(folded_player);
    state.add_player_contribution(5);
    state.add_player_contribution(5);
    return state;
  };

  GameTree::Node player_a_loses_child;
  player_a_loses_child.state = folded_state(0);
  player_a_loses_child.is_terminal = true;
  CFRSolverRegretTestPeer::AddChild(
      solver, node, TestActionKey(ActionType::FOLD),
      std::move(player_a_loses_child));

  GameTree::Node player_a_wins_child;
  player_a_wins_child.state = folded_state(1);
  player_a_wins_child.is_terminal = true;
  CFRSolverRegretTestPeer::AddChild(
      solver, node, TestActionKey(ActionType::CALL, 1),
      std::move(player_a_wins_child));

  double exact_best_response = CFRSolverRegretTestPeer::BestResponseNode(
      solver, node, player_a_hand, player_b_hand, 0);
  WeightedHandRange player_b_singleton;
  player_b_singleton.add(TestComboId(player_b_hand), 1.0);
  double range_best_response = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, node, player_a_hand, player_b_singleton, 0);
  Expect(std::abs(exact_best_response - range_best_response) < 0.000001,
         "singleton range best response should match exact-hand best response");
}

void CheckExploitabilityZeroSamples() {
  PokerConfig config;
  CFRSolver solver(TestSolverConfig(config));
  HandRange player_a_range;
  HandRange player_b_range;
  Expect(solver.calculate_exploitability(0) == 0.0,
         "zero exploitability samples should return zero");
  Expect(solver.calculate_player_a_best_response_value(
             0, player_a_range, player_b_range) == 0.0,
         "zero player A best-response samples should return zero");
  Expect(solver.calculate_player_b_best_response_value(
             0, player_a_range, player_b_range) == 0.0,
         "zero player B best-response samples should return zero");
}

void CheckRangeBestResponseWrappersReturnFiniteValues() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  player_a_range.set_from_string("AA,KK");
  HandRange player_b_range;
  player_b_range.set_from_string("QQ,JJ");

  CFRSolver solver(TestSolverConfig(config));
  solver.run(2, player_a_range, player_b_range);

  double player_a_value = solver.calculate_player_a_best_response_value(
      2, player_a_range, player_b_range);
  double player_b_value = solver.calculate_player_b_best_response_value(
      2, player_a_range, player_b_range);
  Expect(std::isfinite(player_a_value),
         "player A range best-response wrapper should return a finite value");
  Expect(std::isfinite(player_b_value),
         "player B range best-response wrapper should return a finite value");

  player_a_value = solver.calculate_player_a_best_response_value(
      64, player_a_range, player_b_range);
  player_b_value = solver.calculate_player_b_best_response_value(
      64, player_a_range, player_b_range);
  Expect(std::isfinite(player_a_value),
         "parallel player A range best-response should return a finite value");
  Expect(std::isfinite(player_b_value),
         "parallel player B range best-response should return a finite value");
}

void CheckParallelEvaluationUsesWorkerLocalUtilityCaches() {
  PokerConfig config;
  config.set_starting_stack_size(6);
  config.set_max_depth(1);
  config.add_bet_sizes(1.0);

  HandRange player_a_range;
  player_a_range.set_from_string("AA,KK");
  HandRange player_b_range;
  player_b_range.set_from_string("QQ,JJ");

  CFRSolver solver(TestSolverConfig(config));
  solver.run(2, player_a_range, player_b_range);

  const CFRSolver::UtilityCacheStats before =
      solver.get_utility_cache_stats();
  const double value =
      solver.evaluate_strategy(64, player_a_range, player_b_range);
  const CFRSolver::UtilityCacheStats after =
      solver.get_utility_cache_stats();

  Expect(std::isfinite(value),
         "parallel range evaluation should return a finite value");
  if (std::thread::hardware_concurrency() > 1) {
    Expect(after.hits == before.hits && after.misses == before.misses &&
               after.entries == before.entries,
           "parallel range evaluation should not mutate the parent utility cache");
  }
}

void CheckRunUsesConfiguredBlinds() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_small_blind(2);
  config.set_big_blind(5);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, player_a_range, player_b_range);

  Expect(solver.get_iterations_run() == 1,
         "run should include completed iteration count");
  bool has_call_three = false;
  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  for (const CFRSolver::StrategyInfoSet& info_set : strategy.info_sets) {
    for (int action_id : info_set.action_ids) {
      if (action_id == TestActionKey(ActionType::CALL, 3)) {
        has_call_three = true;
      }
    }
  }
  Expect(has_call_three, "configured blinds should set the root call amount");
}

void CheckRunUpdatesExpectedValue() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  Expect(solver.get_iterations_run() == 0, "new solver should have no completed iterations");
  Expect(solver.get_cfr_update_count() == 0,
         "new solver should have no CFR updates");
  solver.run(1, player_a_range, player_b_range);
  Expect(solver.get_iterations_run() == 1, "run should record completed iterations");
  Expect(solver.get_cfr_update_count() == 1,
         "range run should update one sampled private-card deal");

  double player_a_ev = solver.get_expected_value(0);
  double player_b_ev = solver.get_expected_value(1);
  Expect(std::isfinite(player_a_ev),
         "root EV should average completed CFR traversals");
  Expect(std::abs(player_a_ev + player_b_ev) < 0.000001,
         "heads-up EV should be zero-sum");

  solver.run(1, player_a_range, player_b_range);
  Expect(solver.get_iterations_run() == 2, "repeated run should accumulate iterations");
  Expect(solver.get_cfr_update_count() == 2,
         "repeated run should accumulate CFR update count");
  player_a_ev = solver.get_expected_value(0);
  player_b_ev = solver.get_expected_value(1);
  Expect(std::abs(player_a_ev + player_b_ev) < 0.000001,
         "continued EV should stay zero-sum");
}

void CheckRunUsesProvidedPrivateRanges() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, player_a_range, player_b_range);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  Expect(strategy.size() == 1,
         "range run should not swap asymmetric player ranges");
  Expect(StrategyHasCombo(strategy, 0, player_a_hand),
         "range run should train the supplied player A hand class");
}

void CheckRunWithoutDepthCutoffTerminates() {
  PokerConfig config;
  config.set_starting_stack_size(6);
  config.set_small_blind(1);
  config.set_big_blind(2);
  config.set_max_depth(0);
  config.add_bet_sizes(1.0);
  config.set_chance_samples(1);

  HandRange player_a_range;
  player_a_range.set_from_string("AA");
  HandRange player_b_range;
  player_b_range.set_from_string("KK");

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, player_a_range, player_b_range);

  Expect(solver.get_iterations_run() == 1,
         "zero max-depth range run should complete one iteration");
  Expect(solver.get_cfr_update_count() > 0,
         "zero max-depth range run should visit CFR decision nodes");
  CFRSolver::TraversalStats stats = solver.get_traversal_stats();
  Expect(stats.cfr_updates == solver.get_cfr_update_count(),
         "traversal stats should track total CFR updates");
  Expect(stats.preflop_updates > 0,
         "zero max-depth range run should visit preflop decision nodes");
  Expect(stats.max_decision_depth > 0,
         "zero max-depth range run should reach deeper decisions");
  Expect(stats.child_nodes_created > 0,
         "traversal stats should count child node creation");
  Expect(stats.chance_samples > 0,
         "traversal stats should count sampled chance outcomes");
  Expect(stats.terminal_utility_calls > 0,
         "traversal stats should count terminal utility calls");
  Expect(stats.fold_utility_calls > 0,
         "traversal stats should count fold utility calls");
  Expect(stats.showdown_utility_calls > 0,
         "traversal stats should count showdown utility calls");
  Expect(stats.terminal_utility_calls ==
             stats.fold_utility_calls + stats.showdown_utility_calls,
         "terminal utility calls should split into fold and showdown calls");
  Expect(!solver.get_strategy_profile().empty(),
         "zero max-depth range run should produce strategy info sets");
}

void CheckRangeExpansionUsesExactCombos() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange aces;
  aces.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);
  WeightedHandRange combos = aces.get_all_weighted_combos();
  Expect(combos.size() == 6, "AA should expand to six exact combos");

  HandRange ace_king_suited;
  ace_king_suited.add_hand_by_index(HandRange::string_to_index("AKs"), 1.0);
  Expect(ace_king_suited.get_all_weighted_combos().size() == 4,
         "AKs should expand to four exact combos");

  HandRange ace_king_offsuit;
  ace_king_offsuit.add_hand_by_index(HandRange::string_to_index("AKo"), 1.0);
  Expect(ace_king_offsuit.get_all_weighted_combos().size() == 12,
         "AKo should expand to twelve exact combos");

  bool has_disjoint_aces = false;
  for (ComboId left : combos.combos) {
    for (ComboId right : combos.combos) {
      if ((ComboMask(left) & ComboMask(right)) == 0) {
        has_disjoint_aces = true;
      }
    }
  }
  Expect(has_disjoint_aces, "AA range should contain disjoint exact combos");

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, aces, aces);
  Expect(!solver.get_strategy_profile().empty(),
         "same hand class ranges should train from disjoint exact combos");
}

void CheckRangeSamplingRejectsEmptyRange() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  HandRange player_b_range;
  player_b_range.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);

  CFRSolver solver(TestSolverConfig(config));
  bool threw = false;
  try {
    solver.run(1, player_a_range, player_b_range);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "range sampling should reject empty ranges");
}

void CheckRangeSamplerWeightsUseCompatibleProducts() {
  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_a_compatible = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  Hand player_b_compatible = MakeHand(12, Suit::SPADES, 12, Suit::HEARTS);

  HandRange player_a_range;
  AddHand(player_a_range, blocked, 2.0);
  AddHand(player_a_range, player_a_compatible, 3.0);
  HandRange player_b_range;
  AddHand(player_b_range, blocked, 5.0);
  AddHand(player_b_range, player_b_compatible, 7.0);

  std::vector<double> weights =
      CFRSolverRegretTestPeer::PlayerASampleWeights(player_a_range,
                                                    player_b_range);
  std::sort(weights.begin(), weights.end());

  Expect(weights.size() == 2,
         "range sampler should weight each player A exact hand");
  Expect(std::abs(weights[0] - 14.0) < 0.000001,
         "range sampler should skip overlapping player B hands");
  Expect(std::abs(weights[1] - 36.0) < 0.000001,
         "range sampler should weight by compatible player B mass");
}

void CheckRangeSamplingRejectsOnlyOverlappingHands() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, blocked, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, blocked, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  bool threw = false;
  try {
    solver.run(1, player_a_range, player_b_range);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "range sampling should reject fully overlapping exact ranges");
}

void CheckRangeSamplingSkipsOverlappingDeals() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand blocked = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand compatible = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, blocked, 1000.0);
  AddHand(player_a_range, compatible, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, blocked, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(3, player_a_range, player_b_range);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  Expect(strategy.size() == 1,
         "range sampling should only train compatible private deals");
  Expect(StrategyHasCombo(strategy, 0, compatible),
         "range sampling should skip overlapping private-card deals");
}

void CheckRunLoggingUsesAbseilLevels() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CapturingLogSink default_output;
  {
    ScopedVLogLevel default_verbosity(0);
    ScopedLogSink capture_logs(default_output);
    CFRSolver solver(TestSolverConfig(config));
    solver.run(1, player_a_range, player_b_range);
  }
  Expect(default_output.messages().find("Starting CFR iterations") !=
             std::string::npos,
         "run should log lifecycle progress at info level");
  Expect(default_output.messages().find("Iteration 1/1") == std::string::npos,
         "run should not log per-iteration progress at default verbosity");

  CapturingLogSink normal_verbose_output;
  {
    ScopedVLogLevel verbose(1);
    ScopedLogSink capture_logs(normal_verbose_output);
    CFRSolver solver(TestSolverConfig(config));
    solver.run(1, player_a_range, player_b_range);
  }
  Expect(normal_verbose_output.messages().find("Iteration 1/1") ==
             std::string::npos,
         "run should not log per-iteration progress at normal verbosity");

  CapturingLogSink detailed_verbose_output;
  {
    ScopedVLogLevel verbose(2);
    ScopedLogSink capture_logs(detailed_verbose_output);
    CFRSolver solver(TestSolverConfig(config));
    solver.run(1, player_a_range, player_b_range);
  }
  Expect(detailed_verbose_output.messages().find("Iteration 1/1") !=
             std::string::npos,
         "run should log per-iteration progress at detailed verbosity");
  Expect(detailed_verbose_output.messages().find("Iterations run: 1") !=
             std::string::npos,
         "run should log completed iteration count");
  Expect(detailed_verbose_output.messages().find("Information sets: 1") !=
             std::string::npos,
         "run should log trained info set count");
  Expect(detailed_verbose_output.messages().find("Player A average EV:") !=
             std::string::npos,
         "run should log average root EV");
}

void CheckRunProducesDeterministicStrategyShape() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver first_solver(TestSolverConfig(config));
  CFRSolver second_solver(TestSolverConfig(config));
  first_solver.run(2, player_a_range, player_b_range);
  second_solver.run(2, player_a_range, player_b_range);

  const CFRSolver::StrategyProfile first = first_solver.get_strategy_profile();
  const CFRSolver::StrategyProfile second = second_solver.get_strategy_profile();
  Expect(StrategiesApproximatelyEqual(first, second),
         "same config should produce the same strategy profile");
}

void CheckRepeatedRunMatchesSingleRun() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  player_a_range.add_hand_by_index(HandRange::string_to_index("AA"), 1.0);
  HandRange player_b_range;
  player_b_range.add_hand_by_index(HandRange::string_to_index("KK"), 1.0);

  CFRSolver continued(TestSolverConfig(config));
  continued.run(10, player_a_range, player_b_range);
  continued.run(10, player_a_range, player_b_range);

  CFRSolver single(TestSolverConfig(config));
  single.run(20, player_a_range, player_b_range);

  const CFRSolver::StrategyProfile continued_equilibrium = continued.get_strategy_profile();
  const CFRSolver::StrategyProfile single_equilibrium = single.get_strategy_profile();
  Expect(StrategiesApproximatelyEqual(continued_equilibrium, single_equilibrium),
         "continued run should match one longer run");
}

BoardState FoldedState(int folded_player) {
  BoardState state;
  state.set_pot(10);
  state.set_folded_player(folded_player);
  state.add_player_contribution(5);
  state.add_player_contribution(5);
  return state;
}

void AddTerminalChild(CFRSolver& solver,
                      GameTree::Node& node,
                      int action_id,
                      const BoardState& state) {
  GameTree::Node child;
  child.state = TestGameState(state);
  child.is_terminal = true;
  CFRSolverRegretTestPeer::AddChild(solver, node, action_id, std::move(child));
}

BoardState FlopRangeCutoffState() {
  BoardState state;
  state.set_stack_a(20);
  state.set_stack_b(20);
  state.set_pot(0);
  state.set_street(Street::FLOP);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.set_player_to_act(1);
  state.add_player_contribution(0);
  state.add_player_contribution(0);
  AddCard(state, 2, Suit::HEARTS);
  AddCard(state, 7, Suit::DIAMONDS);
  AddCard(state, 9, Suit::CLUBS);
  return state;
}

void CheckRunFixedHandsUsesCustomInitialState() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config), TestGameState(FoldedState(1)));
  Hand player_a_hand;
  Hand player_b_hand;
  solver.run(1, TestComboId(player_a_hand), TestComboId(player_b_hand));

  Expect(solver.get_iterations_run() == 1,
         "fixed-hand run should count iterations");
  Expect(solver.get_expected_value(0) == 5.0,
         "fixed-hand run should use the custom initial state");
}

BoardState ShowdownState(const std::vector<Card>& board_cards) {
  BoardState state;
  state.set_pot(20);
  state.set_street(Street::RIVER);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(10);
  state.add_player_contribution(10);
  for (const Card& card : board_cards) {
    *state.add_cards() = card;
  }
  return state;
}

BoardState RiverFacingCallState() {
  BoardState state;
  state.set_stack_a(5);
  state.set_stack_b(0);
  state.set_pot(15);
  state.set_street(Street::RIVER);
  state.set_all_in(true);
  state.set_folded_player(-1);
  state.set_player_to_act(0);
  state.add_player_contribution(5);
  state.add_player_contribution(10);
  *state.mutable_history()->add_actions() = MakeAction(ActionType::BET, 5);
  AddCard(state, 2, Suit::HEARTS);
  AddCard(state, 7, Suit::DIAMONDS);
  AddCard(state, 9, Suit::CLUBS);
  AddCard(state, 11, Suit::SPADES);
  AddCard(state, 12, Suit::DIAMONDS);
  return state;
}

void CheckExactHandNestedCfrContinuationSolvesRiverCallSubgame() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  ExactHandNestedCFRContinuationValueProvider provider(
      TestSolverConfig(config), 1);
  GameTree game_tree(TestSolverConfig(config));
  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  double value = provider.value(
      game_tree, TestGameState(RiverFacingCallState()),
      TestComboId(player_a_hand), TestComboId(player_b_hand));

  Expect(std::abs(value - 2.5) < 0.000001,
         "exact-hand nested CFR continuation should solve from the cutoff state");
}

void CheckExactHandNestedCfrContinuationCachesSubgames() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  ExactHandNestedCFRContinuationValueProvider provider(
      TestSolverConfig(config), 1);
  GameTree game_tree(TestSolverConfig(config));
  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  BoardState state = RiverFacingCallState();

  double first = provider.value(game_tree, TestGameState(state),
                                TestComboId(player_a_hand),
                                TestComboId(player_b_hand));
  ExactHandNestedCFRContinuationValueProvider::Stats first_stats =
      provider.stats();
  double second = provider.value(game_tree, TestGameState(state),
                                 TestComboId(player_a_hand),
                                 TestComboId(player_b_hand));
  ExactHandNestedCFRContinuationValueProvider::Stats second_stats =
      provider.stats();

  Expect(first == second, "cached subgame value should match the first solve");
  Expect(first_stats.misses == 1 && first_stats.hits == 0 &&
             first_stats.entries == 1,
         "first subgame lookup should miss and populate the cache");
  Expect(second_stats.misses == 1 && second_stats.hits == 1 &&
             second_stats.entries == 1,
         "second subgame lookup should hit the cache");
}

void CheckExactHandNestedCfrContinuationSeparatesPrivateHands() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  ExactHandNestedCFRContinuationValueProvider provider(
      TestSolverConfig(config), 1);
  GameTree game_tree(TestSolverConfig(config));
  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand losing_player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  Hand winning_player_b_hand = MakeHand(2, Suit::SPADES, 2, Suit::CLUBS);
  BoardState state = RiverFacingCallState();

  double value_against_loser =
      provider.value(game_tree, TestGameState(state), TestComboId(player_a_hand),
                     TestComboId(losing_player_b_hand));
  double value_against_winner =
      provider.value(game_tree, TestGameState(state), TestComboId(player_a_hand),
                     TestComboId(winning_player_b_hand));
  ExactHandNestedCFRContinuationValueProvider::Stats stats = provider.stats();

  Expect(std::abs(value_against_loser - 2.5) < 0.000001,
         "exact-hand continuation should value player A's winning showdown");
  Expect(std::abs(value_against_winner + 7.5) < 0.000001,
         "exact-hand continuation should value player A's losing showdown");
  Expect(stats.misses == 2 && stats.hits == 0 && stats.entries == 2,
         "exact-hand continuation should cache different private hands separately");
}

void CheckCfrDepthLimitUsesExactHandNestedContinuationProvider() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  auto provider =
      std::make_shared<ExactHandNestedCFRContinuationValueProvider>(
          TestSolverConfig(config), 1);
  CFRSolver solver(TestSolverConfig(config));
  solver.set_continuation_value_provider(provider);

  GameTree::Node node;
  node.state = TestGameState(RiverFacingCallState());
  node.player_to_act = 0;
  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::array<double, 2> reach_probabilities = {1.0, 1.0};

  double first = TestCfr(solver, node, player_a_hand, player_b_hand,
                            reach_probabilities, 0, 1, 1);
  double second = TestCfr(solver, node, player_a_hand, player_b_hand,
                             reach_probabilities, 1, 1, 1);
  ExactHandNestedCFRContinuationValueProvider::Stats stats = provider->stats();

  Expect(std::abs(first - 2.5) < 0.000001 &&
             std::abs(second - 2.5) < 0.000001,
         "CFR depth cutoff should use exact-hand nested continuation values");
  Expect(stats.misses == 1 && stats.hits == 1 && stats.entries == 1,
         "CFR depth cutoff should reuse exact-hand nested continuation values");
}

GameTree::Node TerminalShowdownNode(const std::vector<Card>& board_cards) {
  GameTree::Node node;
  node.state = TestGameState(ShowdownState(board_cards));
  node.is_terminal = true;
  return node;
}

void CheckTerminalUtilityBeatsDepthLimit() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state = TestGameState(FoldedState(1));
  node.is_terminal = true;

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 5.0, "terminal utility should be returned at the depth limit");
}

void CheckDepthLimitUsesShowdownUtility() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_pot(20);
  node.state.set_street(Street::RIVER);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  node.state.set_player_to_act(1);
  node.state.history.push_back(MakeGameAction(ActionType::CHECK));
  node.state.history.push_back(MakeGameAction(ActionType::CHECK));
  AddCard(node.state, 2, Suit::HEARTS);
  AddCard(node.state, 7, Suit::DIAMONDS);
  AddCard(node.state, 9, Suit::CLUBS);
  AddCard(node.state, 11, Suit::SPADES);
  AddCard(node.state, 12, Suit::DIAMONDS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 10.0, "depth cutoff should use showdown utility when available");
}

void CheckDepthLimitDoesNotScoreUncalledBet() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_pot(15);
  node.state.set_street(Street::RIVER);
  node.state.set_all_in(false);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(5);
  node.state.set_player_to_act(1);
  AddCard(node.state, 2, Suit::HEARTS);
  AddCard(node.state, 7, Suit::DIAMONDS);
  AddCard(node.state, 9, Suit::CLUBS);
  AddCard(node.state, 11, Suit::SPADES);
  AddCard(node.state, 12, Suit::DIAMONDS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 0.0, "depth cutoff should not score unresolved bets");
}

void CheckDepthLimitUsesContinuationValueProvider() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  auto provider = std::make_shared<FixedContinuationValueProvider>(7.0);
  solver.set_continuation_value_provider(provider);

  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;
  node.add_action(MakeGameAction(ActionType::CHECK), GameTree::action_key(MakeGameAction(ActionType::CHECK)));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 7.0,
         "depth cutoff should use the continuation value provider");
  Expect(provider->calls() == 1,
         "depth cutoff should pass through the continuation provider");
  Expect(provider->saw_empty_ranges(),
         "exact-hand CFR cutoff context should not include ranges yet");
}

void CheckRangeRunPassesContinuationRanges() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  HandRange player_a_range;
  player_a_range.set_from_string("AA,KK");
  HandRange player_b_range;
  player_b_range.set_from_string("QQ,JJ");

  CFRSolver solver(TestSolverConfig(config),
                   TestGameState(FlopRangeCutoffState()));
  auto provider = std::make_shared<FixedContinuationValueProvider>(7.0);
  solver.set_continuation_value_provider(provider);
  solver.run(1, player_a_range, player_b_range);

  Expect(provider->calls() > 0,
         "range-trained run should hit the continuation provider");
  Expect(provider->saw_ranges(),
         "range-trained cutoff context should include configured ranges");
  Expect(provider->last_player_a_range_size() ==
             player_a_range.get_all_weighted_combos().size(),
         "player A continuation range should include compatible configured hands");
  Expect(provider->last_player_b_range_size() ==
             player_b_range.get_all_weighted_combos().size(),
         "player B continuation range should include compatible configured hands");
  Expect(std::abs(provider->last_player_a_range_weight() -
                  TestTotalWeight(player_a_range.get_all_weighted_combos())) <
             0.000001,
         "non-acting player range weights should pass through unchanged");
  Expect(std::abs(provider->last_player_b_range_weight() -
                  (TestTotalWeight(player_b_range.get_all_weighted_combos()) /
                   2.0)) < 0.000001,
         "acting player range weights should be conditioned by action probability");
}

void CheckRangeRunActionConditionsRangesByHandStrategy() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_max_depth(1);

  BoardState root_state = FlopRangeCutoffState();
  HandRange player_a_range;
  player_a_range.set_from_string("AA");
  HandRange player_b_range;
  player_b_range.set_from_string("QQ,JJ");

  CFRSolver solver(TestSolverConfig(config), TestGameState(root_state));
  auto provider = std::make_shared<FixedContinuationValueProvider>(7.0);
  solver.set_continuation_value_provider(provider);

  WeightedHandRange player_b_combos = player_b_range.get_all_weighted_combos();
  for (ComboId combo : player_b_combos.combos) {
    const ComboInfo& combo_info = GetComboInfo(combo);
    int preferred_action =
        RankFromCardId(combo_info.card0) == 12
            ? TestActionKey(ActionType::ALL_IN, 20)
            : TestActionKey(ActionType::CHECK);
    CFRSolverRegretTestPeer::SetRegret(
        solver, root_state, 1, TestHandFromCombo(combo), preferred_action, 1.0);
  }

  solver.run(1, player_a_range, player_b_range);

  double queens_weight =
      TestTotalWeightForRank(player_b_range.get_all_weighted_combos(), 12);
  Expect(provider->last_player_b_range_size() == 6,
         "all-in branch should keep only hands that choose all-in");
  Expect(std::abs(provider->last_player_b_range_weight() - queens_weight) <
             0.000001,
         "all-in branch should preserve only all-in hand weights");
}

void CheckActionConditioningSkipsBoardBlockedRangeHands() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  BoardState state = FlopRangeCutoffState();
  Hand blocked_by_board = MakeHand(2, Suit::HEARTS, 14, Suit::SPADES);
  Hand compatible = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);

  WeightedHandRange hands;
  hands.add(TestComboId(blocked_by_board), 1.0);
  hands.add(TestComboId(compatible), 1.0);
  TrainingRange training_range = BuildTrainingRange(hands);
  TrainingRangeView range(training_range);

  CFRSolver solver(TestSolverConfig(config), TestGameState(state));
  TrainingRangeView conditioned =
      CFRSolverRegretTestPeer::ConditionRangeForAction(
          solver, range, state, 1, ActionType::CHECK, 0);

  Expect(conditioned.size() == 1,
         "action conditioning should drop hands blocked by public cards");
  Expect(conditioned.combo(0) == TestComboId(compatible),
         "action conditioning should keep the compatible hand");
  Expect(std::abs(conditioned.weight(0) - 0.5) < 0.000001,
         "compatible hand should retain the check action probability");
}

void CheckActionConditioningIndexTracksNewInfoSets() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  BoardState state = FlopRangeCutoffState();
  Hand queens = MakeHand(12, Suit::HEARTS, 12, Suit::SPADES);

  WeightedHandRange hands;
  hands.add(TestComboId(queens), 1.0);
  TrainingRange training_range = BuildTrainingRange(hands);
  TrainingRangeView range(training_range);

  CFRSolver solver(TestSolverConfig(config), TestGameState(state));
  TrainingRangeView initially_conditioned =
      CFRSolverRegretTestPeer::ConditionRangeForAction(
          solver, range, state, 1, ActionType::ALL_IN, 20);
  Expect(std::abs(initially_conditioned.weight(0) - 0.5) < 0.000001,
         "missing combo index entries should use the uniform strategy");

  CFRSolverRegretTestPeer::SetRegret(
      solver, state, 1, queens, TestActionKey(ActionType::ALL_IN, 20), 1.0);
  TrainingRangeView updated_conditioned =
      CFRSolverRegretTestPeer::ConditionRangeForAction(
          solver, range, state, 1, ActionType::ALL_IN, 20);
  Expect(std::abs(updated_conditioned.weight(0) - 1.0) < 0.000001,
         "combo index should track info sets created after the index is built");
}

void CheckZeroMaxDepthDoesNotCutOff() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node root;
  root.state.set_player_to_act(0);
  root.player_to_act = 0;
  root.add_action(MakeGameAction(ActionType::CHECK), GameTree::action_key(MakeGameAction(ActionType::CHECK)));

  GameTree::Node first_child;
  first_child.state.set_player_to_act(1);
  first_child.player_to_act = 1;
  first_child.add_action(MakeGameAction(ActionType::CHECK), GameTree::action_key(MakeGameAction(ActionType::CHECK)));
  GameTree::Node& first_child_ref = CFRSolverRegretTestPeer::AddChild(
      solver, root, TestActionKey(ActionType::CHECK),
      std::move(first_child));

  GameTree::Node second_child;
  second_child.state.set_player_to_act(0);
  second_child.player_to_act = 0;
  second_child.add_action(MakeGameAction(ActionType::CHECK), GameTree::action_key(MakeGameAction(ActionType::CHECK)));
  GameTree::Node& second_child_ref = CFRSolverRegretTestPeer::AddChild(
      solver, first_child_ref, TestActionKey(ActionType::CHECK),
      std::move(second_child));

  GameTree::Node terminal_child;
  terminal_child.state = TestGameState(FoldedState(1));
  terminal_child.is_terminal = true;
  CFRSolverRegretTestPeer::AddChild(
      solver, second_child_ref, TestActionKey(ActionType::CHECK),
      std::move(terminal_child));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, root, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 0);

  Expect(value == 5.0, "zero max depth should not cut off CFR traversal");
}

void CheckChanceDoesNotConsumeDepth() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.is_chance_node = true;
  node.state.set_pot(20);
  node.state.set_street(Street::TURN);
  node.state.set_all_in(true);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  AddCard(node.state, 14, Suit::HEARTS);
  AddCard(node.state, 14, Suit::DIAMONDS);
  AddCard(node.state, 14, Suit::CLUBS);
  AddCard(node.state, 2, Suit::CLUBS);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 3, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  double value =
      TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 1, 1);

  Expect(value == 10.0, "chance nodes should not consume CFR depth");
}

void CheckChanceSamplesVisitMultipleBoards() {
  BoardState state;
  state.set_stack_a(10);
  state.set_stack_b(10);
  state.set_pot(0);
  state.set_street(Street::PREFLOP);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.set_player_to_act(-1);
  state.add_player_contribution(0);
  state.add_player_contribution(0);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 13, Suit::SPADES);
  Hand player_b_hand = MakeHand(12, Suit::HEARTS, 11, Suit::HEARTS);

  PokerConfig one_sample_config;
  one_sample_config.set_starting_stack_size(10);
  CFRSolver one_sample_solver(TestSolverConfig(one_sample_config));
  GameTree::Node one_sample_node;
  one_sample_node.is_chance_node = true;
  one_sample_node.state = TestGameState(state);
  std::array<double, 2> one_sample_reach = {1.0, 1.0};
  TestCfr(one_sample_solver, one_sample_node, player_a_hand, player_b_hand,
                        one_sample_reach, 0, 0, 1);

  PokerConfig three_sample_config;
  three_sample_config.set_starting_stack_size(10);
  three_sample_config.set_chance_samples(3);
  CFRSolver three_sample_solver(TestSolverConfig(three_sample_config));
  GameTree::Node three_sample_node;
  three_sample_node.is_chance_node = true;
  three_sample_node.state = TestGameState(state);
  std::array<double, 2> three_sample_reach = {1.0, 1.0};
  TestCfr(three_sample_solver, three_sample_node, player_a_hand, player_b_hand,
                          three_sample_reach, 0, 0, 1);

  Expect(one_sample_solver.get_strategy_profile().size() == 1,
         "default chance sampling should visit one sampled board");
  Expect(three_sample_solver.get_strategy_profile().size() >
             one_sample_solver.get_strategy_profile().size(),
         "configured chance samples should visit more sampled boards");
}

void CheckEvaluationUsesChanceSamples() {
  auto chance_node = [] {
    GameTree::Node node;
    node.is_chance_node = true;
    node.state.set_pot(20);
    node.state.set_street(Street::TURN);
    node.state.set_all_in(true);
    node.state.set_folded_player(-1);
    node.state.add_player_contribution(10);
    node.state.add_player_contribution(10);
    AddCard(node.state, 2, Suit::HEARTS);
    AddCard(node.state, 7, Suit::DIAMONDS);
    AddCard(node.state, 9, Suit::CLUBS);
    AddCard(node.state, 11, Suit::SPADES);
    return node;
  };

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b_hand = MakeHand(10, Suit::HEARTS, 10, Suit::DIAMONDS);

  PokerConfig one_sample_config;
  one_sample_config.set_starting_stack_size(10);
  CFRSolver one_sample_solver(TestSolverConfig(one_sample_config));
  GameTree::Node one_sample_node = chance_node();
  double first = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, one_sample_node, player_a_hand, player_b_hand);
  double second = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, one_sample_node, player_a_hand, player_b_hand);
  double third = CFRSolverRegretTestPeer::EvaluateNode(
      one_sample_solver, one_sample_node, player_a_hand, player_b_hand);
  double expected_average = (first + second + third) / 3.0;
  Expect(std::abs(first - expected_average) > 0.000001,
         "chance sample fixture should have varied outcomes");

  PokerConfig three_sample_config;
  three_sample_config.set_starting_stack_size(10);
  three_sample_config.set_chance_samples(3);
  CFRSolver three_sample_solver(TestSolverConfig(three_sample_config));
  GameTree::Node three_sample_node = chance_node();
  double sampled_average = CFRSolverRegretTestPeer::EvaluateNode(
      three_sample_solver, three_sample_node, player_a_hand, player_b_hand);
  Expect(std::abs(sampled_average - expected_average) < 0.000001,
         "strategy evaluation should average configured chance samples");

  CFRSolver one_sample_best_response(TestSolverConfig(one_sample_config));
  GameTree::Node one_sample_response_node = chance_node();
  double first_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, one_sample_response_node, player_a_hand,
      player_b_hand, 0);
  double second_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, one_sample_response_node, player_a_hand,
      player_b_hand, 0);
  double third_response = CFRSolverRegretTestPeer::BestResponseNode(
      one_sample_best_response, one_sample_response_node, player_a_hand,
      player_b_hand, 0);
  double expected_response_average =
      (first_response + second_response + third_response) / 3.0;

  CFRSolver three_sample_best_response(TestSolverConfig(three_sample_config));
  GameTree::Node three_sample_response_node = chance_node();
  double sampled_response_average = CFRSolverRegretTestPeer::BestResponseNode(
      three_sample_best_response, three_sample_response_node, player_a_hand,
      player_b_hand, 0);
  Expect(std::abs(sampled_response_average - expected_response_average) <
             0.000001,
         "best response should average configured chance samples");
}

void CheckBestResponseActionSelectsBestLegalAction() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  GameAction player_a_loses = MakeGameAction(ActionType::FOLD);
  GameAction player_a_wins = MakeGameAction(ActionType::CALL, 1);
  node.add_action(player_a_loses, GameTree::action_key(player_a_loses));
  node.add_action(player_a_wins, GameTree::action_key(player_a_wins));

  AddTerminalChild(solver, node, TestActionKey(ActionType::FOLD),
                   FoldedState(0));
  AddTerminalChild(solver, node, TestActionKey(ActionType::CALL, 1),
                   FoldedState(1));

  Hand player_a_hand;
  Hand player_b_hand;
  GameAction best_action =
      solver.get_best_response_action(
          node, TestComboId(player_a_hand), TestComboId(player_b_hand), 0);
  Expect(best_action.kind == ActionKind::kCall,
         "best response should select the highest-value legal action");
  Expect(best_action.amount == 1,
         "best response should preserve selected action amount");

  GameAction opponent_turn_action =
      solver.get_best_response_action(
          node, TestComboId(player_a_hand), TestComboId(player_b_hand), 1);
  Expect(opponent_turn_action.kind == ActionKind::kNoAction,
         "best response should return no action away from its turn");
}

void CheckRangeBestResponseDoesNotKnowOpponentHand() {
  PokerConfig config;
  config.set_starting_stack_size(20);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  GameAction first_board = MakeGameAction(ActionType::CHECK);
  GameAction second_board = MakeGameAction(ActionType::CALL, 1);
  node.add_action(first_board, GameTree::action_key(first_board));
  node.add_action(second_board, GameTree::action_key(second_board));

  CFRSolverRegretTestPeer::AddChild(
      solver, node, TestActionKey(ActionType::CHECK),
      TerminalShowdownNode(
          {MakeTestCard(13, Suit::SPADES), MakeTestCard(13, Suit::HEARTS),
           MakeTestCard(3, Suit::CLUBS), MakeTestCard(4, Suit::DIAMONDS),
           MakeTestCard(5, Suit::SPADES)}));
  CFRSolverRegretTestPeer::AddChild(
      solver, node, TestActionKey(ActionType::CALL, 1),
      TerminalShowdownNode(
          {MakeTestCard(2, Suit::SPADES), MakeTestCard(2, Suit::HEARTS),
           MakeTestCard(3, Suit::CLUBS), MakeTestCard(4, Suit::DIAMONDS),
           MakeTestCard(5, Suit::SPADES)}));

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand kings = MakeHand(13, Suit::CLUBS, 13, Suit::DIAMONDS);
  Hand twos = MakeHand(2, Suit::CLUBS, 2, Suit::DIAMONDS);
  WeightedHandRange opponent_hands;
  opponent_hands.add(TestComboId(kings), 1.0);
  opponent_hands.add(TestComboId(twos), 1.0);

  double clairvoyant_average =
      (CFRSolverRegretTestPeer::BestResponseNode(solver, node, player_a_hand,
                                                 kings, 0) +
       CFRSolverRegretTestPeer::BestResponseNode(solver, node, player_a_hand,
                                                 twos, 0)) /
      2.0;
  double range_value = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, node, player_a_hand, opponent_hands, 0);

  Expect(clairvoyant_average > 0.0,
         "exact-hand best response can overuse hidden opponent cards");
  Expect(std::abs(range_value) < 0.000001,
         "range best response should choose one action for the opponent range");
}

void CheckRangeBestResponseChanceUsesSampledOpponent() {
  PokerConfig config;
  config.set_starting_stack_size(20);
  config.set_chance_samples(1);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.is_chance_node = true;
  node.state.set_pot(20);
  node.state.set_street(Street::TURN);
  node.state.set_all_in(true);
  node.state.set_folded_player(-1);
  node.state.add_player_contribution(10);
  node.state.add_player_contribution(10);
  AddCard(node.state, 2, Suit::CLUBS);
  AddCard(node.state, 2, Suit::DIAMONDS);
  AddCard(node.state, 2, Suit::HEARTS);
  AddCard(node.state, 3, Suit::CLUBS);

  Hand player_a_hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand quads = MakeHand(2, Suit::SPADES, 7, Suit::SPADES);
  Hand air = MakeHand(13, Suit::CLUBS, 12, Suit::DIAMONDS);
  WeightedHandRange opponent_hands;
  opponent_hands.add(TestComboId(quads), 1.0);
  opponent_hands.add(TestComboId(air), 1.0);

  double value = CFRSolverRegretTestPeer::BestResponseRangeNode(
      solver, node, player_a_hand, opponent_hands, 0);

  Expect(std::abs(std::abs(value) - 10.0) < 0.000001,
         "chance range best response should score the sampled opponent only");
}

void CheckPlayerBRegretsUsePlayerBUtility() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(1);
  node.player_to_act = 1;

  GameAction player_b_loses = MakeGameAction(ActionType::FOLD);
  GameAction player_b_wins = MakeGameAction(ActionType::CALL, 1);
  node.add_action(player_b_loses, GameTree::action_key(player_b_loses));
  node.add_action(player_b_wins, GameTree::action_key(player_b_wins));

  AddTerminalChild(solver, node, TestActionKey(ActionType::FOLD),
                   FoldedState(1));
  AddTerminalChild(solver, node, TestActionKey(ActionType::CALL, 1),
                   FoldedState(0));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 2);
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 1, 0, 2);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  const auto action_probs = StrategyActionMap(strategy);
  Expect(action_probs.at(TestActionKey(ActionType::CALL, 1)) >
             action_probs.at(TestActionKey(ActionType::FOLD)),
         "player B should prefer the action that lowers player A utility");
}

void CheckCfrPlusClipsNegativeRegrets() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  GameAction player_a_loses = MakeGameAction(ActionType::FOLD);
  GameAction player_a_wins = MakeGameAction(ActionType::CALL, 1);
  node.add_action(player_a_loses, GameTree::action_key(player_a_loses));
  node.add_action(player_a_wins, GameTree::action_key(player_a_wins));

  AddTerminalChild(solver, node, TestActionKey(ActionType::FOLD),
                   FoldedState(0));
  AddTerminalChild(solver, node, TestActionKey(ActionType::CALL, 1),
                   FoldedState(1));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  Expect(CFRSolverRegretTestPeer::Regret(
             solver, node, 0, player_a_hand,
             TestActionKey(ActionType::FOLD)) == 0.0,
         "CFR+ should clip negative cumulative regrets");
  Expect(CFRSolverRegretTestPeer::Regret(
             solver, node, 0, player_a_hand,
             TestActionKey(ActionType::CALL, 1)) > 0.0,
         "positive cumulative regret should remain positive");
}

void CheckCfrPlusWeightsLaterStrategies() {
  PokerConfig config;
  config.set_starting_stack_size(10);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  GameAction player_a_loses = MakeGameAction(ActionType::FOLD);
  GameAction player_a_wins = MakeGameAction(ActionType::CALL, 1);
  node.add_action(player_a_loses, GameTree::action_key(player_a_loses));
  node.add_action(player_a_wins, GameTree::action_key(player_a_wins));

  AddTerminalChild(solver, node, TestActionKey(ActionType::FOLD),
                   FoldedState(0));
  AddTerminalChild(solver, node, TestActionKey(ActionType::CALL, 1),
                   FoldedState(1));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 1, 0, 1);

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  const auto action_probs = StrategyActionMap(strategy);
  Expect(std::abs(action_probs.at(TestActionKey(ActionType::CALL, 1)) -
                  (5.0 / 6.0)) < 0.000001,
         "CFR+ average strategy should weight later iterations linearly");
}

void CheckRegretOnlyTrainingSkipsAverageStrategyWrites() {
  PokerConfig config;
  config.set_starting_stack_size(10);
  config.set_regret_only_training(true);

  CFRSolver solver(TestSolverConfig(config));
  GameTree::Node node;
  node.state.set_player_to_act(0);
  node.state.set_folded_player(-1);
  node.player_to_act = 0;

  node.add_action(MakeGameAction(ActionType::FOLD), GameTree::action_key(MakeGameAction(ActionType::FOLD)));
  node.add_action(MakeGameAction(ActionType::CALL, 1), GameTree::action_key(MakeGameAction(ActionType::CALL, 1)));

  AddTerminalChild(solver, node, TestActionKey(ActionType::FOLD),
                   FoldedState(0));
  AddTerminalChild(solver, node, TestActionKey(ActionType::CALL, 1),
                   FoldedState(1));

  Hand player_a_hand;
  Hand player_b_hand;
  std::array<double, 2> reach_probabilities = {1.0, 1.0};
  TestCfr(solver, node, player_a_hand, player_b_hand, reach_probabilities, 0, 0, 1);

  Expect(CFRSolverRegretTestPeer::TotalCumulativeStrategy(solver) == 0.0,
         "regret-only training should skip average strategy writes");

  const CFRSolver::StrategyProfile strategy = solver.get_strategy_profile();
  const auto action_probs = StrategyActionMap(strategy);
  Expect(action_probs.at(TestActionKey(ActionType::CALL, 1)) == 1.0,
         "regret-only export should use current regret-matched strategy");
}

void CheckMaxInfoSetsCapsTrainingAllocations() {
  PokerConfig config;
  config.set_starting_stack_size(10);
  config.add_bet_sizes(1.0);
  config.set_regret_only_training(true);
  config.set_max_info_sets(1);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, player_a_range, player_b_range);

  Expect(solver.get_info_set_count() == 1,
         "max_info_sets should cap training info set allocations");
  Expect(solver.get_strategy_profile().size() == 1,
         "capped solver should only export allocated info sets");
}

void CheckMaxTreeNodesCapsTrainingAllocations() {
  PokerConfig config;
  config.set_starting_stack_size(10);
  config.add_bet_sizes(1.0);
  config.set_regret_only_training(true);
  config.set_max_tree_nodes(3);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(20, player_a_range, player_b_range);

  Expect(solver.get_tree_node_count() <= 3,
         "max_tree_nodes should cap total cached training nodes");
}

}  // namespace

int main() {
  CheckCfrUsesLegalActions();
  CheckCfrDistinguishesActionAmounts();
  CheckPublicStateKeyIgnoresBoardOrder();
  CheckIdentityPrivateAbstractionUsesExactCombo();
  CheckPublicStateIdsAreDenseAndKeyedByState();
  CheckBettingHistoryIdsIgnorePublicCards();
  CheckBettingHistoryActionTransitionsAreCached();
  CheckSparseSlabRowsUseExactCombos();
  CheckSparseSlabTracksInfoSets();
  CheckTerminalUtilityCacheReusesScores();
  CheckSingletonRangeMatchesExactEvaluationAndBestResponse();
  CheckExploitabilityZeroSamples();
  CheckRangeBestResponseWrappersReturnFiniteValues();
  CheckParallelEvaluationUsesWorkerLocalUtilityCaches();
  CheckRunUsesConfiguredBlinds();
  CheckRunUpdatesExpectedValue();
  CheckRunUsesProvidedPrivateRanges();
  CheckRunFixedHandsUsesCustomInitialState();
  CheckRunWithoutDepthCutoffTerminates();
  CheckRangeExpansionUsesExactCombos();
  CheckRangeSamplingRejectsEmptyRange();
  CheckRangeSamplerWeightsUseCompatibleProducts();
  CheckRangeSamplingRejectsOnlyOverlappingHands();
  CheckRangeSamplingSkipsOverlappingDeals();
  CheckRunLoggingUsesAbseilLevels();
  CheckRunProducesDeterministicStrategyShape();
  CheckRepeatedRunMatchesSingleRun();
  CheckTerminalUtilityBeatsDepthLimit();
  CheckDepthLimitUsesShowdownUtility();
  CheckDepthLimitDoesNotScoreUncalledBet();
  CheckDepthLimitUsesContinuationValueProvider();
  CheckRangeRunPassesContinuationRanges();
  CheckRangeRunActionConditionsRangesByHandStrategy();
  CheckActionConditioningSkipsBoardBlockedRangeHands();
  CheckActionConditioningIndexTracksNewInfoSets();
  CheckExactHandNestedCfrContinuationSolvesRiverCallSubgame();
  CheckExactHandNestedCfrContinuationCachesSubgames();
  CheckExactHandNestedCfrContinuationSeparatesPrivateHands();
  CheckCfrDepthLimitUsesExactHandNestedContinuationProvider();
  CheckZeroMaxDepthDoesNotCutOff();
  CheckChanceDoesNotConsumeDepth();
  CheckChanceSamplesVisitMultipleBoards();
  CheckEvaluationUsesChanceSamples();
  CheckBestResponseActionSelectsBestLegalAction();
  CheckRangeBestResponseDoesNotKnowOpponentHand();
  CheckRangeBestResponseChanceUsesSampledOpponent();
  CheckPlayerBRegretsUsePlayerBUtility();
  CheckCfrPlusClipsNegativeRegrets();
  CheckCfrPlusWeightsLaterStrategies();
  CheckRegretOnlyTrainingSkipsAverageStrategyWrites();
  CheckMaxInfoSetsCapsTrainingAllocations();
  CheckMaxTreeNodesCapsTrainingAllocations();

  PokerConfig config;
  config.add_bet_sizes(1.0);
  config.set_starting_stack_size(10);
  config.set_max_depth(4);

  Hand player_a_hand = MakeHand(14, Suit::SPADES, 14, Suit::HEARTS);
  Hand player_b_hand = MakeHand(13, Suit::SPADES, 13, Suit::HEARTS);
  HandRange player_a_range;
  AddHand(player_a_range, player_a_hand, 1.0);
  HandRange player_b_range;
  AddHand(player_b_range, player_b_hand, 1.0);

  CFRSolver solver(TestSolverConfig(config));
  solver.run(1, player_a_range, player_b_range);

  if (solver.get_strategy_profile().empty()) {
    throw std::runtime_error("CFR did not visit any information sets");
  }

  return 0;
}
