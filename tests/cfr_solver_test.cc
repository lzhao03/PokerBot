#include "src/solver.h"
#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest/doctest.h"

namespace poker {

struct TabularCfrSolverTestAccess {
  static const CfrState& state(const TabularCfrSolver& solver) {
    return solver.state_;
  }
  static CfrState& state(TabularCfrSolver& solver) { return solver.state_; }
};

namespace {

using S = Suit;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(Card(static_cast<Rank>(r0 - 2), s0),
                        Card(static_cast<Rank>(r1 - 2), s1));
}

Card C(int rank, S suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
}

ComboRange R(ComboId hand) {
  ComboRange range;
  range.add(hand);
  return range;
}

BettingState Apply(const BettingState& state, GameAction action) {
  const auto* decision = std::get_if<DecisionState>(&state);
  if (decision == nullptr) {
    throw std::invalid_argument("expected decision state");
  }
  const auto child = ApplyAction(*decision, action);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

ExactPublicState DealChance(const ExactPublicState& state,
                            std::span<const Card> cards,
                            const BettingRules& rules) {
  const auto child = TryApplyChance(state, cards, rules);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

SolverConfig Config(bool accumulate_average = true,
                    int max_info_sets = 500000) {
  SolverConfig options;
  options.card_abstraction = {
      PublicCardMode::ExactCanonical,
      PrivateAbstractionKind::ExactCanonical,
  };
  for (auto& fractions : options.bet_abstraction.pot_fractions) {
    fractions = {0.5, 1.0};
  }
  options.chance_samples = 1;
  options.max_info_sets = max_info_sets;
  options.accumulate_average_strategy = accumulate_average;
  const auto config = SolverConfig::Create(std::move(options));
  if (!config.ok()) {
    throw std::invalid_argument(std::string(config.status().message()));
  }
  return *config;
}

ExactPublicState Root(const SolverConfig& config) {
  return MakeInitialState(config.betting_rules, {8, 8}, {1, 2});
}

ExactPublicState WinningRiverRoot() {
  BettingData data;
  data.stack = {4, 4};
  data.total_committed = {4, 4};
  data.street_committed = {0, 0};
  data.last_full_raise = 2;
  data.street = StreetKind::River;
  data.actions_remaining = 2;
  const std::array board = {
      C(2, S::Clubs), C(7, S::Diamonds), C(9, S::Hearts),
      C(11, S::Spades), C(12, S::Clubs)};
  return {DecisionState{data, Player::A}, *MakeBoard(board)};
}

std::unique_ptr<TabularCfrSolver> MakeSolver(
    const SolverConfig& config,
    const ComboRange& a,
    const ComboRange& b,
    std::optional<ExactPublicState> root = std::nullopt) {
  auto solver = TabularCfrSolver::Create(
      {config, root.value_or(Root(config)), {a, b}});
  if (!solver.ok()) {
    throw std::invalid_argument(std::string(solver.status().message()));
  }
  return std::make_unique<TabularCfrSolver>(std::move(*solver));
}

TEST_CASE("solver configuration rejects invalid boundary values") {
  SolverConfig options;
  options.max_info_sets = 0;
  CHECK_FALSE(SolverConfig::Create(options).ok());

  options.max_info_sets = 10;
  options.bet_abstraction.pot_fractions[0] = {-0.5};
  CHECK_FALSE(SolverConfig::Create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {0.0};
  CHECK_FALSE(SolverConfig::Create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {1.0, 0.25, 0.5, 0.5};
  const auto normalized = SolverConfig::Create(options);
  REQUIRE(normalized.ok());
  CHECK(normalized->bet_abstraction.pot_fractions[0] ==
        std::vector<double>{0.25, 0.5, 1.0});

  options.bet_abstraction.pot_fractions[0] = {0.1, 0.2, 0.3,
                                               0.4, 0.5, 0.6};
  CHECK_FALSE(SolverConfig::Create(options).ok());

  const SolverConfig defaults;
  CHECK(defaults.card_abstraction.public_mode == PublicCardMode::Texture);
  CHECK(defaults.card_abstraction.private_kind ==
        PrivateAbstractionKind::Handcrafted36);
  CHECK(defaults.card_abstraction.recall_mode ==
        RecallMode::BucketHistory);
}

const ComboId kA = H(14, S::Hearts, 14, S::Spades);
const ComboId kB = H(13, S::Clubs, 13, S::Diamonds);
const ComboId kC = H(12, S::Clubs, 12, S::Diamonds);

Policy PassiveCallingPolicy(const TabularCfrSolver& game, ComboId hand) {
  const CompiledGame& compiled = game.game();
  Policy policy;
  policy.model = compiled.model;
  const PublicPosition& position = compiled.root.public_state;
  const PrivateObservationId private_observation = ObservePrivate(
      hand, position);
  for (size_t history = 0; history < compiled.history.nodes.size();
       ++history) {
    const HistoryNode& node = compiled.history.nodes[history];
    const auto* decision = std::get_if<DecisionState>(&node.state);
    if (decision == nullptr || decision->actor != Player::B) continue;
    const AbstractActions actions = SelectAbstractActions(
        compiled.config.bet_abstraction, *decision);
    const size_t offset = policy.probabilities.size();
    policy.probabilities.resize(offset + node.child_count, 0.0f);
    bool selected = false;
    for (uint8_t action = 0; action < node.child_count; ++action) {
      const ActionKind kind = actions[action].kind;
      if (kind == ActionKind::Call || kind == ActionKind::Check) {
        policy.probabilities[offset + action] = 1.0f;
        selected = true;
      }
    }
    REQUIRE(selected);
    policy.rows.try_emplace(
        InfoSetKey{position.observation(),
                   HistoryId{static_cast<uint32_t>(history)},
                   private_observation},
        offset);
  }
  return policy;
}

TEST_CASE("small exact solver baseline is deterministic") {
  auto solver = MakeSolver(
      Config(), UniformComboRange(), UniformComboRange());
  solver->run(10);

  CHECK(solver->history_count() == 417);
  CHECK(solver->info_set_count() == 720);
  CHECK(solver->stats().decision_visits == 1440);
  CHECK(solver->expected_value(Player::A) ==
        doctest::Approx(-0.737666));
}

TEST_CASE("external sampling visits only traverser action branches") {
  auto full = MakeSolver(Config(), R(kA), R(kB));
  SolverConfig config = Config();
  config.external_sampling = true;
  auto sampled = MakeSolver(config, R(kA), R(kB));
  full->run(2);
  sampled->run(2);

  CHECK(sampled->stats().decision_visits <
        full->stats().decision_visits);
  CHECK(sampled->game().model == full->game().model);
  CHECK(sampled->iterations() == 2);
  CHECK(sampled->extract_average_policy().ok());
}

TEST_CASE("model fingerprints are stable and cover solve ranges") {
  auto first = MakeSolver(Config(), R(kA), R(kB));
  auto second = MakeSolver(Config(), R(kA), R(kB));
  auto different_training = MakeSolver(Config(false, 10), R(kA), R(kB));
  auto changed = MakeSolver(Config(), R(kB), R(kA));
  CHECK(first->game().model == second->game().model);
  CHECK(first->game().model == different_training->game().model);
  CHECK(first->game().model != changed->game().model);
  CHECK(std::to_underlying(first->game().model) ==
        0x9ebae6e5a4064673ULL);
}

TEST_CASE("private abstraction cannot change terminal utility") {
  ExactPublicState terminal = WinningRiverRoot();
  BettingData data = Data(terminal.betting);
  data.actions_remaining = 0;
  terminal.betting = ShowdownState{data};
  auto exact = MakeSolver(Config(), R(kA), R(kB), terminal);
  auto handcrafted = MakeSolver(
      SolverConfig{}, R(kA), R(kB), terminal);
  const double expected = exact->evaluate_current(kA, kB);
  CHECK(expected == doctest::Approx(4.0));
  CHECK(handcrafted->evaluate_current(kA, kB) == expected);
}

TEST_CASE("history tree stores direct rule transitions") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const HistoryTree& tree = solver->game().history;
  REQUIRE_FALSE(tree.nodes.empty());

  for (size_t id = 0; id < tree.nodes.size(); ++id) {
    const HistoryNode& node = tree.nodes[id];
    if (const auto* decision = std::get_if<DecisionState>(&node.state)) {
      const AbstractActions actions = SelectAbstractActions(
          solver->game().config.bet_abstraction, *decision);
      REQUIRE(node.child_count == actions.size());
      for (uint8_t action = 0; action < node.child_count; ++action) {
        const HistoryId child = tree.children[node.children_begin + action];
        REQUIRE(Index(child) < tree.nodes.size());
        CHECK(tree.nodes[Index(child)].state ==
              Apply(node.state, actions[action]));
      }
    } else if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
      REQUIRE(node.child_count == 1);
      const HistoryId child = tree.children[node.children_begin];
      REQUIRE(Index(child) < tree.nodes.size());
      CHECK(tree.nodes[Index(child)].state == AdvanceBettingStreet(
                *chance, solver->game().config.betting_rules));
    } else {
      CHECK(node.child_count == 0);
    }
  }

  const HistoryNode& root = tree.nodes[0];
  REQUIRE(root.child_count >= 2);
  CHECK(tree.children[root.children_begin] !=
        tree.children[root.children_begin + 1]);
}

TEST_CASE("training mutates only CFR state") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const size_t history_count = solver->history_count();
  solver->run(4);
  CHECK(solver->iterations() == 4);
  CHECK(solver->info_set_count() > 0);
  CHECK(solver->stats().decision_visits > 0);
  CHECK(std::isfinite(solver->expected_value(Player::A)));
  CHECK(solver->history_count() == history_count);

  const CfrState before = TabularCfrSolverTestAccess::state(*solver);
  const uint64_t updates = solver->stats().decision_visits;
  const auto value = solver->evaluate_average(kA, kB);
  REQUIRE(value.ok());
  CHECK(std::isfinite(*value));
  CHECK(solver->history_count() == history_count);
  CHECK(solver->stats().decision_visits == updates);
  CHECK(TabularCfrSolverTestAccess::state(*solver).row_entries() ==
        before.row_entries());
  CHECK(TabularCfrSolverTestAccess::state(*solver).regret_sum ==
        before.regret_sum);
  CHECK(TabularCfrSolverTestAccess::state(*solver).strategy_sum ==
        before.strategy_sum);
}

TEST_CASE("training uses preallocated action arrays") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const CfrState& before = TabularCfrSolverTestAccess::state(*solver);
  const size_t regret_capacity = before.regret_sum.capacity();
  const size_t strategy_capacity = before.strategy_sum.capacity();

  solver->run(4);

  const CfrState& after = TabularCfrSolverTestAccess::state(*solver);
  CHECK(after.regret_sum.capacity() == regret_capacity);
  CHECK(after.strategy_sum.capacity() == strategy_capacity);
}

TEST_CASE("infoset action rows are contiguous") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const CfrState& state = TabularCfrSolverTestAccess::state(*solver);

  struct RowSize {
    size_t offset;
    uint8_t action_count;
  };
  std::vector<RowSize> rows;
  const auto entries = state.row_entries();
  rows.reserve(entries.size());
  for (const auto& entry : entries) {
    const HistoryNode& node =
        solver->game().history.nodes[Index(entry.first.history)];
    REQUIRE(std::holds_alternative<DecisionState>(node.state));
    rows.push_back({entry.second, node.child_count});
  }
  std::sort(rows.begin(), rows.end(), [](RowSize left, RowSize right) {
    return left.offset < right.offset;
  });

  size_t offset = 0;
  for (const RowSize& item : rows) {
    CHECK(item.offset == offset);
    offset += item.action_count;
  }
  CHECK(offset == state.regret_sum.size());
  CHECK(state.strategy_sum.size() == state.regret_sum.size());
}

TEST_CASE("postflop roots use full observation identity") {
  SolverConfig config = Config();
  const BettingRules& rules = config.betting_rules;
  ExactPublicState root = MakeInitialState(rules, {8, 8}, {1, 2});
  root.betting = Apply(root.betting, {ActionKind::Call, 2});
  root.betting = Apply(root.betting, {ActionKind::Check, 0});
  const std::array<Card, 3> flop = {
      Card(Rank::Two, S::Hearts), Card(Rank::Seven, S::Diamonds),
      Card(Rank::Queen, S::Clubs)};
  root = DealChance(root, flop, rules);

  auto solver = MakeSolver(config, R(kA), R(kB), root);
  solver->run(2);
  const HistoryTree& tree = solver->game().history;
  const Player player = std::get<DecisionState>(tree.nodes[0].state).actor;
  const ComboId hand = player == Player::A ? kA : kB;
  const CardAbstractionConfig& cards =
      solver->game().config.card_abstraction;
  const PublicPosition public_state(cards, root.board);
  const PrivateObservationId private_id = ObservePrivate(hand, public_state);
  CHECK(TabularCfrSolverTestAccess::state(*solver)
            .find({public_state.observation(), HistoryId{}, private_id})
            .has_value());
}

TEST_CASE("training continues after the infoset cap is reached") {
  auto solver = MakeSolver(Config(true, 1), R(kA), R(kB));
  solver->run(2);
  CHECK(solver->iterations() == 2);
  CHECK(solver->info_set_count() == 1);
}

TEST_CASE("parallel training updates a fixed-capacity table") {
  auto solver = MakeSolver(Config(true, 1), R(kA), R(kB));
  solver->run(20, 4);
  CHECK(solver->iterations() == 20);
  CHECK(solver->info_set_count() == 1);
  CHECK(solver->stats().decision_visits > 0);
  const CfrState& state = TabularCfrSolverTestAccess::state(*solver);
  CHECK(std::ranges::all_of(state.regret_sum, [](float value) {
    return std::isfinite(value) && value >= 0.0f;
  }));
  CHECK(std::ranges::all_of(state.strategy_sum, [](float value) {
    return std::isfinite(value) && value >= 0.0f;
  }));
}

TEST_CASE("average strategy storage is optional") {
  const SolverConfig config = Config(false);
  auto solver = MakeSolver(config, R(kA), R(kB));
  solver->run(2);

  CHECK(TabularCfrSolverTestAccess::state(*solver).strategy_sum.empty());
  CHECK(std::isfinite(solver->evaluate_current(kA, kB)));
  CHECK_FALSE(solver->evaluate_average(kA, kB).ok());
}

TEST_CASE("average policies are normalized and evaluate reproducibly") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const auto extracted = solver->extract_average_policy();
  REQUIRE(extracted.ok());
  const Policy policy = *extracted;
  REQUIRE_FALSE(policy.rows.empty());
  for (const auto& [key, offset] : policy.rows) {
    (void)offset;
    const HistoryNode& node =
        solver->game().history.nodes[Index(key.history)];
    std::vector<float> probabilities(node.child_count);
    CHECK(policy.strategy(key, absl::MakeSpan(probabilities)));
    double sum = 0.0;
    for (float probability : probabilities) sum += probability;
    CHECK(sum == doctest::Approx(1.0));
  }

  std::array<float, 3> missing = {};
  CHECK_FALSE(policy.strategy(
      {PublicObservationId(),
       HistoryId{std::numeric_limits<uint32_t>::max()},
       PrivateObservationId()},
      absl::MakeSpan(missing)));
  for (float probability : missing) {
    CHECK(probability == doctest::Approx(1.0 / missing.size()));
  }

  const auto evaluated =
      EstimateExpectedValue(solver->game(), policy, policy, 4, 17, true);
  const auto repeated =
      EstimateExpectedValue(solver->game(), policy, policy, 4, 17);
  REQUIRE(evaluated.ok());
  REQUIRE(repeated.ok());
  CHECK(std::isfinite(evaluated->mean));
  CHECK(evaluated->policy_lookups > 0);
  CHECK(evaluated->weighted_policy_lookups > 0.0);
  CHECK(evaluated->observed_info_sets > 0);
  CHECK(evaluated->info_sets_for_99_percent_reach <=
        evaluated->observed_info_sets);
  CHECK(evaluated->mean == repeated->mean);
  CHECK(evaluated->standard_error == repeated->standard_error);
  CHECK(evaluated->policy_lookups == repeated->policy_lookups);
  CHECK(evaluated->missing_policy_lookups ==
        repeated->missing_policy_lookups);
  CHECK(evaluated->weighted_policy_lookups ==
        repeated->weighted_policy_lookups);
  CHECK(evaluated->weighted_missing_policy_lookups ==
        repeated->weighted_missing_policy_lookups);

  Policy empty;
  empty.model = policy.model;
  const auto fallback =
      EstimateExpectedValue(solver->game(), empty, empty, 2, 17);
  REQUIRE(fallback.ok());
  CHECK(fallback->missing_policy_lookups > 0);
  CHECK(fallback->weighted_missing_policy_lookups > 0.0);

  auto different = MakeSolver(Config(), R(kB), R(kA));
  CHECK_FALSE(EstimateExpectedValue(
      different->game(), policy, policy, 1, 17).ok());
}

TEST_CASE("zero average mass extracts as uniform policy") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(1);
  CfrState& state = TabularCfrSolverTestAccess::state(*solver);
  std::fill(state.strategy_sum.begin(), state.strategy_sum.end(), 0.0f);
  const auto extracted = solver->extract_average_policy();
  REQUIRE(extracted.ok());
  for (const auto& [key, offset] : extracted->rows) {
    (void)offset;
    const HistoryNode& node =
        solver->game().history.nodes[Index(key.history)];
    std::vector<float> probabilities(node.child_count);
    REQUIRE(extracted->strategy(key, absl::MakeSpan(probabilities)));
    for (float probability : probabilities) {
      CHECK(probability == doctest::Approx(1.0 / node.child_count));
    }
  }
}

TEST_CASE("approximate responses are reproducible and respect infosets") {
  ComboRange opponent_range;
  opponent_range.add(kB);
  opponent_range.add(kC);
  auto game = MakeSolver(Config(), R(kA), opponent_range);
  game->run(20);
  const auto opponent = game->extract_average_policy();
  REQUIRE(opponent.ok());

  const BestResponseConfig config{30, 20, 91};
  const auto first = TrainApproximateBestResponse(
      game->game(), Player::A, *opponent, config);
  const auto second = TrainApproximateBestResponse(
      game->game(), Player::A, *opponent, config);
  const StrategyLookup lookup = [&opponent](
      InfoSetKey key, std::span<float> output) {
    return opponent->strategy(key, output);
  };
  const auto generic = TrainApproximateBestResponse(
      game->game(), Player::A, lookup, config);
  REQUIRE(first.ok());
  REQUIRE(second.ok());
  REQUIRE(generic.ok());
  CHECK(first->response_policy.rows == second->response_policy.rows);
  CHECK(first->response_policy.probabilities ==
        second->response_policy.probabilities);
  CHECK(first->response_policy.rows == generic->response_policy.rows);
  CHECK(first->response_policy.probabilities ==
        generic->response_policy.probabilities);
  CHECK(first->value == second->value);
  CHECK(first->value == generic->value);
  CHECK(first->standard_error == second->standard_error);
  CHECK(first->opponent_policy_lookups > 0);
  CHECK(first->response_policy_lookups > 0);

  size_t root_rows = 0;
  for (const auto& [key, row] : first->response_policy.rows) {
    (void)row;
    const HistoryNode& node =
        game->game().history.nodes[Index(key.history)];
    REQUIRE(std::holds_alternative<DecisionState>(node.state));
    CHECK(std::get<DecisionState>(node.state).actor == Player::A);
    root_rows += key.history == HistoryId{} ? 1 : 0;
  }
  CHECK(root_rows == 1);
}

TEST_CASE("approximate response continues after infoset capacity") {
  auto game = MakeSolver(Config(true, 1), R(kA), R(kB));
  game->run(1);
  const auto opponent = game->extract_average_policy();
  REQUIRE(opponent.ok());
  const auto response = TrainApproximateBestResponse(
      game->game(), Player::A, *opponent, BestResponseConfig{10, 2, 7});
  REQUIRE(response.ok());
  CHECK(response->response_policy.rows.size() == 1);
}

TEST_CASE("approximate response learns a profitable shared action") {
  auto game = MakeSolver(Config(), R(kA), R(kB), WinningRiverRoot());
  const Policy opponent = PassiveCallingPolicy(*game, kB);
  Policy uniform;
  uniform.model = game->game().model;
  const auto baseline = EstimateExpectedValue(
      game->game(), uniform, opponent, 1, 11);
  BestResponseConfig response_config{100, 1, 11};
  response_config.external_sampling = true;
  const auto response = TrainApproximateBestResponse(
      game->game(), Player::A, opponent, response_config);
  REQUIRE(baseline.ok());
  REQUIRE(response.ok());
  CHECK(response->value >= baseline->mean);
  CHECK(response->value > 7.5);

  const CompiledGame& compiled = game->game();
  const PublicPosition& position = compiled.root.public_state;
  const InfoSetKey root_key{
      position.observation(), compiled.root.history,
      ObservePrivate(kA, position)};
  const size_t offset = response->response_policy.rows.at(root_key);
  const HistoryNode& root = compiled.history.nodes[0];
  const AbstractActions actions = SelectAbstractActions(
      compiled.config.bet_abstraction,
      std::get<DecisionState>(root.state));
  for (uint8_t action = 0; action < root.child_count; ++action) {
    const GameAction game_action = actions[action];
    if (game_action.kind == ActionKind::AllIn) {
      CHECK(response->response_policy.probabilities[
                offset + action] > 0.95f);
    }
  }
}

TEST_CASE("exploitability reports both responder perspectives") {
  auto game = MakeSolver(Config(), R(kA), R(kB), WinningRiverRoot());
  const auto initial_policy = game->extract_average_policy();
  REQUIRE(initial_policy.ok());
  const auto initial = EstimateExploitability(
      game->game(), *initial_policy, BestResponseConfig{200, 2, 23});
  REQUIRE(initial.ok());

  game->run(200);
  const auto policy = game->extract_average_policy();
  REQUIRE(policy.ok());
  const auto estimate = EstimateExploitability(
      game->game(), *policy, BestResponseConfig{200, 2, 23});
  REQUIRE(estimate.ok());
  CHECK(estimate->nash_conv == doctest::Approx(
      estimate->player_a_response.value +
      estimate->player_b_response.value));
  CHECK(estimate->exploitability ==
        doctest::Approx(0.5 * estimate->nash_conv));
  const double sampling_tolerance = 3.0 * (
      estimate->player_a_response.standard_error +
      estimate->player_b_response.standard_error);
  CHECK(estimate->nash_conv >= -sampling_tolerance);
  CHECK(estimate->exploitability < 0.1);
  CHECK(estimate->exploitability < initial->exploitability);
}

}  // namespace
}  // namespace poker
