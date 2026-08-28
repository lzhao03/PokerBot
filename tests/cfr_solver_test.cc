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
  static const InfoSetTable& table(const TabularCfrSolver& solver) {
    return solver.info_sets_;
  }
  static InfoSetTable& table(TabularCfrSolver& s) { return s.info_sets_; }
};

namespace {

using S = Suit;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(Card(static_cast<Rank>(r0 - 2), s0),
                        Card(static_cast<Rank>(r1 - 2), s1));
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

SolverConfig Config(int max_info_sets = 500000) {
  SolverConfig options;
  options.card_abstraction = {
      PublicCardMode::ExactCanonical,
      PrivateAbstractionKind::ExactCanonical,
  };
  for (auto& fractions : options.bet_abstraction.pot_fractions) {
    fractions = {0.5, 1.0};
  }
  options.starting_stacks = {8, 8};
  options.chance_samples = 1;
  options.max_info_sets = max_info_sets;
  return options;
}

std::unique_ptr<TabularCfrSolver> MakeSolver(
    const SolverConfig& config,
    const ComboRange& a,
    const ComboRange& b) {
  auto solver = TabularCfrSolver::Create(config, {a, b});
  if (!solver.ok()) {
    throw std::invalid_argument(std::string(solver.status().message()));
  }
  return std::make_unique<TabularCfrSolver>(std::move(*solver));
}

TEST_CASE("solver configuration rejects invalid boundary values") {
  const SolverConfig defaults;
  const ComboRange range = UniformComboRange();
  const auto create = [&](SolverConfig config) {
    return TabularCfrSolver::Create(std::move(config), {range, range});
  };
  SolverConfig options;
  options.starting_stacks[0] = options.small_blind;
  CHECK_FALSE(create(options).ok());

  options = SolverConfig{};
  options.small_blind = options.betting_rules.minimum_bet + 1;
  CHECK_FALSE(create(options).ok());

  options = SolverConfig{};
  options.max_info_sets = 0;
  CHECK_FALSE(create(options).ok());

  options.max_info_sets = 10;
  options.bet_abstraction.pot_fractions[0] = {-0.5};
  CHECK_FALSE(create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {0.0};
  CHECK_FALSE(create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {1.0, 0.25, 0.5};
  CHECK_FALSE(create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {0.25, 0.5, 0.5, 1.0};
  CHECK_FALSE(create(options).ok());

  options.bet_abstraction.pot_fractions[0] = {0.25, 0.5, 1.0};
  const auto valid = create(options);
  REQUIRE(valid.ok());
  CHECK(valid->config().bet_abstraction.pot_fractions[0] ==
        std::vector<double>{0.25, 0.5, 1.0});

  options.bet_abstraction.pot_fractions[0] = {0.1, 0.2, 0.3,
                                               0.4, 0.5, 0.6};
  CHECK_FALSE(create(options).ok());

  CHECK(defaults.card_abstraction.public_mode == PublicCardMode::Texture);
  CHECK(defaults.card_abstraction.private_kind ==
        PrivateAbstractionKind::Handcrafted36);
  CHECK(defaults.card_abstraction.recall_mode == RecallMode::BucketHistory);
}

const ComboId kA = H(14, S::Hearts, 14, S::Spades);
const ComboId kB = H(13, S::Clubs, 13, S::Diamonds);
const ComboId kC = H(12, S::Clubs, 12, S::Diamonds);

TEST_CASE("small exact solver preserves structural baseline") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(10);

  CHECK(solver->history_count() == 417);
  CHECK(solver->info_set_count() == 672);
  CHECK(solver->stats().decision_visits == 1440);
  CHECK(std::isfinite(solver->expected_value(Player::A)));
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
  CHECK(sampled->model() == full->model());
  CHECK(sampled->iterations() == 2);
  CHECK_FALSE(sampled->extract_average_policy().rows.empty());
}

TEST_CASE("external sampling linearly weights average strategies") {
  SolverConfig config = Config();
  config.external_sampling = true;
  auto solver = MakeSolver(config, R(kA), R(kB));
  solver->run(2);

  const InfoSetTable& table = TabularCfrSolverTestAccess::table(*solver);
  const InfoSetKey root_key{
      solver->initial_public().observation(), HistoryId{},
      ObservePrivate(kA, solver->initial_public())};
  const std::optional<uint32_t> offset = table.find(root_key);
  REQUIRE(offset.has_value());
  const uint8_t action_count = solver->history().nodes[0].child_count;
  CHECK(std::accumulate(table.strategy_sum.begin() + *offset,
                        table.strategy_sum.begin() + *offset + action_count,
                        0.0f) == doctest::Approx(2.0f));
}

TEST_CASE("model fingerprints are stable and cover the game") {
  auto first = MakeSolver(Config(), R(kA), R(kB));
  auto second = MakeSolver(Config(), R(kA), R(kB));
  auto different_training = MakeSolver(Config(10), R(kA), R(kB));
  auto changed = MakeSolver(Config(), R(kB), R(kA));
  SolverConfig blind_config = Config();
  blind_config.small_blind = 2;
  auto changed_blind = MakeSolver(blind_config, R(kA), R(kB));
  CHECK(first->model() == second->model());
  CHECK(first->model() == different_training->model());
  CHECK(first->model() != changed->model());
  CHECK(first->model() != changed_blind->model());
  CHECK(std::to_underlying(first->model()) == 0x9ebae6e5a4064673ULL);
}

TEST_CASE("history tree stores direct rule transitions") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const HistoryTree& tree = solver->history();
  REQUIRE_FALSE(tree.nodes.empty());

  for (size_t id = 0; id < tree.nodes.size(); ++id) {
    const HistoryNode& node = tree.nodes[id];
    if (const auto* decision = std::get_if<DecisionState>(&node.state)) {
      const AbstractActions actions = SelectAbstractActions(
          solver->config().bet_abstraction, *decision);
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
                *chance, solver->config().betting_rules));
    } else {
      CHECK(node.child_count == 0);
    }
  }

  const HistoryNode& root = tree.nodes[0];
  REQUIRE(root.child_count >= 2);
  CHECK(tree.children[root.children_begin] !=
        tree.children[root.children_begin + 1]);
}

TEST_CASE("evaluation does not mutate the infoset table") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const size_t history_count = solver->history_count();
  solver->run(4);
  CHECK(solver->iterations() == 4);
  CHECK(solver->info_set_count() > 0);
  CHECK(solver->stats().decision_visits > 0);
  CHECK(std::isfinite(solver->expected_value(Player::A)));
  CHECK(solver->history_count() == history_count);

  const InfoSetTable before = TabularCfrSolverTestAccess::table(*solver);
  const uint64_t updates = solver->stats().decision_visits;
  const StrategyLookup average = solver->average_strategy();
  const auto value = EstimateExpectedValue(
      solver->config(), solver->deals(), solver->history(),
      solver->initial_public(), average, average, 1, 17);
  REQUIRE(value.ok());
  CHECK(std::isfinite(value->mean));
  CHECK(solver->history_count() == history_count);
  CHECK(solver->stats().decision_visits == updates);
  CHECK(TabularCfrSolverTestAccess::table(*solver).row_entries() ==
        before.row_entries());
  CHECK(TabularCfrSolverTestAccess::table(*solver).regret_sum ==
        before.regret_sum);
  CHECK(TabularCfrSolverTestAccess::table(*solver).strategy_sum ==
        before.strategy_sum);
}

TEST_CASE("training uses preallocated action arrays") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const InfoSetTable& before = TabularCfrSolverTestAccess::table(*solver);
  const size_t regret_capacity = before.regret_sum.capacity();
  const size_t strategy_capacity = before.strategy_sum.capacity();

  solver->run(4);

  const InfoSetTable& after = TabularCfrSolverTestAccess::table(*solver);
  CHECK(after.regret_sum.capacity() == regret_capacity);
  CHECK(after.strategy_sum.capacity() == strategy_capacity);
}

TEST_CASE("infoset action rows are contiguous") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const InfoSetTable& table = TabularCfrSolverTestAccess::table(*solver);

  struct RowSize {
    size_t offset;
    uint8_t action_count;
  };
  std::vector<RowSize> rows;
  const auto entries = table.row_entries();
  rows.reserve(entries.size());
  for (const auto& entry : entries) {
    const HistoryNode& node =
        solver->history().nodes[Index(entry.first.history)];
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
  CHECK(offset == table.regret_sum.size());
  CHECK(table.strategy_sum.size() == table.regret_sum.size());
}

TEST_CASE("training continues after the infoset cap is reached") {
  auto solver = MakeSolver(Config(1), R(kA), R(kB));
  solver->run(2);
  CHECK(solver->iterations() == 2);
  CHECK(solver->info_set_count() == 1);
}

TEST_CASE("parallel training updates a fixed-capacity table") {
  auto solver = MakeSolver(Config(1), R(kA), R(kB));
  solver->run(20, 4);
  CHECK(solver->iterations() == 20);
  CHECK(solver->info_set_count() == 1);
  CHECK(solver->stats().decision_visits > 0);
  const InfoSetTable& table = TabularCfrSolverTestAccess::table(*solver);
  CHECK(std::ranges::all_of(table.regret_sum, [](float value) {
    return std::isfinite(value) && value >= 0.0f;
  }));
  CHECK(std::ranges::all_of(table.strategy_sum, [](float value) {
    return std::isfinite(value) && value >= 0.0f;
  }));
}

TEST_CASE("average policies are normalized and evaluate reproducibly") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const Policy policy = solver->extract_average_policy();
  REQUIRE_FALSE(policy.rows.empty());
  for (const auto& [key, offset] : policy.rows) {
    (void)offset;
    const HistoryNode& node = solver->history().nodes[Index(key.history)];
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
      EstimateExpectedValue(solver->config(), solver->deals(),
                            solver->history(), solver->initial_public(),
                            solver->model(), policy, policy, 4, 17, true);
  const auto repeated =
      EstimateExpectedValue(solver->config(), solver->deals(),
                            solver->history(), solver->initial_public(),
                            solver->model(), policy, policy, 4, 17);
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
  CHECK(evaluated->missing_policy_lookups == repeated->missing_policy_lookups);
  CHECK(evaluated->weighted_policy_lookups ==
        repeated->weighted_policy_lookups);
  CHECK(evaluated->weighted_missing_policy_lookups ==
        repeated->weighted_missing_policy_lookups);

  Policy empty;
  empty.model = policy.model;
  const auto fallback =
      EstimateExpectedValue(solver->config(), solver->deals(),
                            solver->history(), solver->initial_public(),
                            solver->model(), empty, empty, 2, 17);
  REQUIRE(fallback.ok());
  CHECK(fallback->missing_policy_lookups > 0);
  CHECK(fallback->weighted_missing_policy_lookups > 0.0);

  const StrategyLookup unavailable = [](InfoSetKey, std::span<float> output) {
    std::ranges::fill(output, std::numeric_limits<float>::quiet_NaN());
    return false;
  };
  const auto generic_fallback = EstimateExpectedValue(
      solver->config(), solver->deals(), solver->history(),
      solver->initial_public(), unavailable, unavailable, 2, 17);
  REQUIRE(generic_fallback.ok());
  CHECK(generic_fallback->mean == fallback->mean);
  CHECK(generic_fallback->missing_policy_lookups ==
        fallback->missing_policy_lookups);

  auto different = MakeSolver(Config(), R(kB), R(kA));
  CHECK_FALSE(EstimateExpectedValue(
      different->config(), different->deals(), different->history(),
      different->initial_public(), different->model(), policy, policy, 1,
      17).ok());
}

TEST_CASE("zero average mass extracts as uniform policy") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(1);
  InfoSetTable& table = TabularCfrSolverTestAccess::table(*solver);
  std::fill(table.strategy_sum.begin(), table.strategy_sum.end(), 0.0f);
  const Policy extracted = solver->extract_average_policy();
  for (const auto& [key, offset] : extracted.rows) {
    (void)offset;
    const HistoryNode& node = solver->history().nodes[Index(key.history)];
    std::vector<float> probabilities(node.child_count);
    REQUIRE(extracted.strategy(key, absl::MakeSpan(probabilities)));
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
  const Policy opponent = game->extract_average_policy();

  const BestResponseConfig config{30, 20, 91};
  const auto first = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, opponent, config);
  const auto second = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, opponent, config);
  const StrategyLookup lookup = [&opponent](
      InfoSetKey key, std::span<float> output) {
    return opponent.strategy(key, output);
  };
  const auto generic = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, lookup, config);
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
    const HistoryNode& node = game->history().nodes[Index(key.history)];
    REQUIRE(std::holds_alternative<DecisionState>(node.state));
    CHECK(std::get<DecisionState>(node.state).actor == Player::A);
    root_rows += key.history == HistoryId{} ? 1 : 0;
  }
  CHECK(root_rows == 1);
}

TEST_CASE("approximate response continues after infoset capacity") {
  auto game = MakeSolver(Config(1), R(kA), R(kB));
  game->run(1);
  const Policy opponent = game->extract_average_policy();
  const auto response = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, opponent, BestResponseConfig{10, 2, 7});
  REQUIRE(response.ok());
  CHECK(response->response_policy.rows.size() == 1);
}

TEST_CASE("approximate response learns a profitable initial action") {
  SolverConfig config = Config();
  config.starting_stacks = {3, 3};
  auto game = MakeSolver(config, R(kA), R(kB));
  Policy uniform;
  uniform.model = game->model();
  const auto baseline = EstimateExpectedValue(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), uniform, uniform, 1000, 11);
  BestResponseConfig response_config{1000, 1000, 11};
  response_config.external_sampling = true;
  const auto sampled = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, uniform, response_config);
  response_config.external_sampling = false;
  const auto full = TrainApproximateBestResponse(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), Player::A, uniform, response_config);
  REQUIRE(baseline.ok());
  REQUIRE(sampled.ok());
  REQUIRE(full.ok());
  CHECK(sampled->value > baseline->mean);
  CHECK(full->value > baseline->mean);
}

TEST_CASE("parallel exploitability matches serial") {
  SolverConfig solver_config = Config();
  solver_config.starting_stacks = {2, 2};
  auto game = MakeSolver(solver_config, R(kA), R(kB));
  const BestResponseConfig response_config{200, 100, 23};
  const Policy initial_policy = game->extract_average_policy();
  const auto initial = EstimateExploitability(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), initial_policy, response_config);
  game->run(200);
  const Policy policy = game->extract_average_policy();
  const auto estimate = EstimateExploitability(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), policy, response_config);
  const StrategyLookup lookup = [&policy](
      InfoSetKey key, std::span<float> output) {
    return policy.strategy(key, output);
  };
  size_t factory_calls = 0;
  const auto parallel = EstimateExploitabilityParallel(
      game->config(), game->deals(), game->history(), game->initial_public(),
      game->model(), [&] {
        ++factory_calls;
        return lookup;
      },
      response_config);
  REQUIRE(initial.ok());
  REQUIRE(estimate.ok());
  REQUIRE(parallel.ok());
  CHECK(factory_calls == kPlayerCount);
  CHECK(parallel->player_a_response.value == estimate->player_a_response.value);
  CHECK(parallel->player_b_response.value == estimate->player_b_response.value);
  CHECK(estimate->nash_conv == doctest::Approx(
      estimate->player_a_response.value +
      estimate->player_b_response.value));
  CHECK(estimate->exploitability == doctest::Approx(0.5 * estimate->nash_conv));
  CHECK(estimate->exploitability < initial->exploitability);
}

}  // namespace
}  // namespace poker
