#include "src/solver.h"
#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
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
  static CfrState& state(CFRSolver& solver) { return solver.state_; }
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
  return SingleComboRange(hand);
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
                            absl::Span<const Card> cards,
                            const BettingRules& rules) {
  const auto child = TryApplyChance(state, cards, rules);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

BettingState NodeState(const HistoryNode& node) {
  return std::visit([](const auto& value) -> BettingState {
    return value.state;
  }, node);
}

SolverConfig Config(bool accumulate_average = true,
                    int max_info_sets = 500000) {
  SolverConfigOptions options;
  options.starting_stack = 8;
  options.small_blind = 1;
  options.big_blind = 2;
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
  const Chips stack = config.starting_stack();
  return MakeInitialState(BettingRules{config.big_blind()}, {stack, stack},
                          {config.small_blind(), config.big_blind()});
}

ExactPublicState WinningRiverRoot() {
  BettingData data;
  data.stack = {4, 4};
  data.total_committed = {4, 4};
  data.street_committed = {0, 0};
  data.last_full_raise = 2;
  data.street = StreetKind::River;
  data.pending_action_mask = kAllPlayersMask;
  const FlopBoard flop = DealFlop(
      PreflopBoard{}, {C(2, S::Clubs), C(7, S::Diamonds),
                      C(9, S::Hearts)});
  return {DecisionState{data, Player::A},
          DealRiver(DealTurn(flop, C(11, S::Spades)),
                    C(12, S::Clubs))};
}

std::unique_ptr<CFRSolver> MakeSolver(
    const SolverConfig& config,
    const ComboRange& a,
    const ComboRange& b,
    std::optional<ExactPublicState> root = std::nullopt) {
  auto solver = CFRSolver::Create(
      {config, root.value_or(Root(config)), {a, b}});
  if (!solver.ok()) {
    throw std::invalid_argument(std::string(solver.status().message()));
  }
  return std::move(*solver);
}

std::string Hex(ModelFingerprint fingerprint) {
  constexpr char kDigits[] = "0123456789abcdef";
  std::string text;
  text.reserve(fingerprint.bytes.size() * 2);
  for (std::byte byte : fingerprint.bytes) {
    const uint8_t value = std::to_integer<uint8_t>(byte);
    text.push_back(kDigits[value >> 4]);
    text.push_back(kDigits[value & 0x0F]);
  }
  return text;
}

void CheckSameState(const CfrState& left, const CfrState& right) {
  CHECK(left.rows == right.rows);
  CHECK(left.regret_sum == right.regret_sum);
  CHECK(left.strategy_sum == right.strategy_sum);
  CHECK(left.iterations == right.iterations);
  CHECK(left.cumulative_root_utility == right.cumulative_root_utility);
}

TEST_CASE("solver configuration rejects invalid boundary values") {
  SolverConfigOptions options;
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
  CHECK(normalized->bet_abstraction().pot_fractions[0] ==
        std::vector<double>{0.25, 0.5, 1.0});

  const SolverConfig defaults = SolverConfig::Default();
  CHECK(defaults.card_abstraction().public_mode == PublicCardMode::Texture);
  CHECK(defaults.card_abstraction().private_kind ==
        PrivateAbstractionKind::Handcrafted36);
  CHECK(defaults.card_abstraction().recall_mode ==
        RecallMode::BucketHistory);
}

const ComboId kA = H(14, S::Hearts, 14, S::Spades);
const ComboId kB = H(13, S::Clubs, 13, S::Diamonds);
const ComboId kC = H(12, S::Clubs, 12, S::Diamonds);

Policy PassiveCallingPolicy(const CFRSolver& game, ComboId hand) {
  Policy policy;
  policy.model = game.model_fingerprint();
  const PublicPosition position = PublicPosition::Root(
      game.card_abstraction(), StreetKind::River,
      game.solve_spec().root.board);
  const PrivateObservationId private_observation = ObservePrivate(
      game.card_abstraction(), hand, position);
  for (size_t history = 0; history < game.history_tree().nodes.size();
       ++history) {
    const auto* node =
        std::get_if<DecisionNode>(&game.history_tree().nodes[history]);
    if (node == nullptr || node->state.actor != Player::B) continue;
    const size_t offset = policy.probabilities.size();
    policy.probabilities.resize(offset + node->edges.count, 0.0f);
    bool selected = false;
    for (uint8_t action = 0; action < node->edges.count; ++action) {
      const ActionKind kind = game.history_tree()
                                  .edges[node->edges.begin + action]
                                  .action.kind;
      if (kind == ActionKind::Call || kind == ActionKind::Check) {
        policy.probabilities[offset + action] = 1.0f;
        selected = true;
      }
    }
    REQUIRE(selected);
    policy.rows.emplace(
        InfoSetKey{HistoryId(static_cast<uint32_t>(history)),
                   position.observation(), private_observation},
        PolicyRow{offset, node->edges.count});
  }
  return policy;
}

TEST_CASE("small exact solver baseline is deterministic") {
  auto solver = MakeSolver(
      Config(), UniformComboRange(), UniformComboRange());
  solver->run(10);

  CHECK(solver->get_history_count() == 417);
  CHECK(solver->get_info_set_count() == 720);
  CHECK(solver->get_cfr_update_count() == 1440);
  CHECK(solver->get_expected_value(Player::A) ==
        doctest::Approx(-1.01572));
}

TEST_CASE("model fingerprints are stable and cover solve ranges") {
  auto first = MakeSolver(Config(), R(kA), R(kB));
  auto second = MakeSolver(Config(), R(kA), R(kB));
  auto changed = MakeSolver(Config(), R(kB), R(kA));
  CHECK(first->model_fingerprint() == second->model_fingerprint());
  CHECK(first->model_fingerprint() != changed->model_fingerprint());
  CHECK(Hex(first->model_fingerprint()) ==
        "d888124d143d268d139afce227d10e26"
        "624b3c05912c12885f0d8b7696b14041");
}

TEST_CASE("private abstraction cannot change terminal utility") {
  ExactPublicState terminal = WinningRiverRoot();
  BettingData data = Data(terminal.betting);
  data.pending_action_mask = 0;
  terminal.betting = ShowdownState{data};
  auto exact = MakeSolver(Config(), R(kA), R(kB), terminal);
  auto handcrafted = MakeSolver(
      SolverConfig::Default(), R(kA), R(kB), terminal);
  const double expected = exact->evaluate_current(
      HoleCards(kA), HoleCards(kB));
  CHECK(expected == doctest::Approx(4.0));
  CHECK(handcrafted->evaluate_current(HoleCards(kA), HoleCards(kB)) ==
        expected);
}

TEST_CASE("history tree stores direct rule transitions") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const HistoryTree& tree = CFRSolverTestAccess::history(*solver);
  REQUIRE(tree.root.index() < tree.nodes.size());

  for (size_t id = 0; id < tree.nodes.size(); ++id) {
    const HistoryNode& node = tree.nodes[id];
    if (const auto* decision = std::get_if<DecisionNode>(&node)) {
      REQUIRE(decision->edges.count > 0);
      for (uint8_t action = 0; action < decision->edges.count; ++action) {
        const HistoryEdge& edge =
            tree.edges[decision->edges.begin + action];
        REQUIRE(edge.child.index() < tree.nodes.size());
        CHECK(NodeState(tree.nodes[edge.child.index()]) ==
              Apply(decision->state, edge.action));
      }
    } else if (const auto* chance = std::get_if<ChanceNode>(&node)) {
      REQUIRE(chance->child.index() < tree.nodes.size());
      CHECK(NodeState(tree.nodes[chance->child.index()]) ==
            AdvanceBettingStreet(chance->state, BettingRules{2}));
    }
  }

  const DecisionNode& root =
      std::get<DecisionNode>(tree.nodes[tree.root.index()]);
  REQUIRE(root.edges.count >= 2);
  CHECK(tree.edges[root.edges.begin].child !=
        tree.edges[root.edges.begin + 1].child);
}

TEST_CASE("training mutates only CFR state") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const size_t history_count = solver->get_history_count();
  solver->run(4);
  CHECK(solver->get_iterations_run() == 4);
  CHECK(solver->get_info_set_count() > 0);
  CHECK(solver->get_cfr_update_count() > 0);
  CHECK(std::isfinite(solver->get_expected_value(Player::A)));
  CHECK(solver->get_history_count() == history_count);

  const CfrState before = CFRSolverTestAccess::state(*solver);
  const int64_t updates = solver->get_cfr_update_count();
  const auto value =
      solver->evaluate_average(HoleCards(kA), HoleCards(kB));
  REQUIRE(value.ok());
  CHECK(std::isfinite(*value));
  CHECK(solver->get_history_count() == history_count);
  CHECK(solver->get_cfr_update_count() == updates);
  CHECK(CFRSolverTestAccess::state(*solver).rows == before.rows);
  CHECK(CFRSolverTestAccess::state(*solver).regret_sum == before.regret_sum);
  CHECK(CFRSolverTestAccess::state(*solver).strategy_sum ==
        before.strategy_sum);
}

TEST_CASE("training uses preallocated action arrays") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  const CfrState& before = CFRSolverTestAccess::state(*solver);
  const size_t regret_capacity = before.regret_sum.capacity();
  const size_t strategy_capacity = before.strategy_sum.capacity();

  solver->run(4);

  const CfrState& after = CFRSolverTestAccess::state(*solver);
  CHECK(after.regret_sum.capacity() == regret_capacity);
  CHECK(after.strategy_sum.capacity() == strategy_capacity);
}

TEST_CASE("infoset action rows are contiguous") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const CfrState& state = CFRSolverTestAccess::state(*solver);

  struct RowSize {
    InfoSetRow row;
    uint8_t action_count;
  };
  std::vector<RowSize> rows;
  rows.reserve(state.rows.size());
  for (const auto& entry : state.rows) {
    const HistoryNode& node = CFRSolverTestAccess::history(*solver)
                                  .nodes[entry.first.history.index()];
    rows.push_back({entry.second,
                    std::get<DecisionNode>(node).edges.count});
  }
  std::sort(rows.begin(), rows.end(), [](RowSize left, RowSize right) {
    return left.row.action_offset < right.row.action_offset;
  });

  size_t offset = 0;
  for (const RowSize& item : rows) {
    CHECK(item.row.action_offset == offset);
    offset += item.action_count;
  }
  CHECK(offset == state.regret_sum.size());
  CHECK(state.strategy_sum.size() == state.regret_sum.size());
}

TEST_CASE("postflop roots use full observation identity") {
  SolverConfig config = Config();
  const BettingRules rules{config.big_blind()};
  ExactPublicState root = MakeInitialState(rules, {8, 8}, {1, 2});
  root.betting = Apply(root.betting, {ActionKind::Call, 2});
  root.betting = Apply(root.betting, {ActionKind::Check, 0});
  const std::array<Card, 3> flop = {
      Card(Rank::Two, S::Hearts), Card(Rank::Seven, S::Diamonds),
      Card(Rank::Queen, S::Clubs)};
  root = DealChance(root, flop, rules);

  auto solver = MakeSolver(config, R(kA), R(kB), root);
  solver->run(2);
  const HistoryTree& tree = CFRSolverTestAccess::history(*solver);
  const Player player =
      std::get<DecisionNode>(tree.nodes[tree.root.index()]).state.actor;
  const ComboId hand = player == Player::A ? kA : kB;
  const PublicPosition public_state = PublicPosition::Root(
      solver->card_abstraction(), Data(root.betting).street, root.board);
  const PrivateObservationId private_id = ObservePrivate(
      solver->card_abstraction(), hand, public_state);
  CHECK(CFRSolverTestAccess::state(*solver).rows.contains(
      {tree.root, public_state.observation(), private_id}));
}

TEST_CASE("training continues after the infoset cap is reached") {
  auto solver = MakeSolver(Config(true, 1), R(kA), R(kB));
  const TrainingResult result = solver->run(2);
  CHECK(result.iterations_completed == 2);
  CHECK(solver->get_iterations_run() == 2);
  CHECK(solver->get_info_set_count() == 1);
}

TEST_CASE("average strategy storage is optional") {
  const SolverConfig config = Config(false);
  auto solver = MakeSolver(config, R(kA), R(kB));
  solver->run(2);

  CHECK(CFRSolverTestAccess::state(*solver).strategy_sum.empty());
  CHECK(std::isfinite(
      solver->evaluate_current(HoleCards(kA), HoleCards(kB))));
  CHECK_FALSE(
      solver->evaluate_average(HoleCards(kA), HoleCards(kB)).ok());
}

TEST_CASE("average policies are normalized and persist exactly") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  const auto extracted = solver->extract_average_policy();
  REQUIRE(extracted.ok());
  const Policy policy = *extracted;
  REQUIRE_FALSE(policy.rows.empty());
  for (const auto& [key, row] : policy.rows) {
    std::vector<float> probabilities(row.action_count);
    CHECK(policy.strategy(key, absl::MakeSpan(probabilities)));
    double sum = 0.0;
    for (float probability : probabilities) sum += probability;
    CHECK(sum == doctest::Approx(1.0));
  }

  std::array<float, 3> missing = {};
  CHECK_FALSE(policy.strategy(
      {HistoryId(std::numeric_limits<uint32_t>::max()),
       PublicObservationId(), PrivateObservationId()},
      absl::MakeSpan(missing)));
  for (float probability : missing) {
    CHECK(probability == doctest::Approx(1.0 / missing.size()));
  }

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  REQUIRE(test_tmpdir != nullptr);
  const std::filesystem::path path =
      std::filesystem::path(test_tmpdir) / "average.policy";
  REQUIRE(SavePolicy(policy, path).ok());
  const auto loaded = LoadPolicy(path);
  REQUIRE(loaded.ok());
  CHECK(loaded->model == policy.model);
  CHECK(loaded->rows == policy.rows);
  CHECK(loaded->probabilities == policy.probabilities);

  const SerializedRngState rng_before = solver->checkpoint().rng;
  const auto evaluated = EstimateExpectedValue(*solver, policy, policy, 4, 17);
  const auto repeated = EstimateExpectedValue(*solver, policy, policy, 4, 17);
  REQUIRE(evaluated.ok());
  REQUIRE(repeated.ok());
  CHECK(std::isfinite(evaluated->mean));
  CHECK(evaluated->policy_lookups > 0);
  CHECK(evaluated->mean == repeated->mean);
  CHECK(evaluated->standard_error == repeated->standard_error);
  CHECK(evaluated->policy_lookups == repeated->policy_lookups);
  CHECK(evaluated->missing_policy_lookups ==
        repeated->missing_policy_lookups);
  CHECK(solver->checkpoint().rng == rng_before);

  Policy empty;
  empty.model = policy.model;
  const auto fallback = EstimateExpectedValue(*solver, empty, empty, 2, 17);
  REQUIRE(fallback.ok());
  CHECK(fallback->missing_policy_lookups > 0);

  auto different = MakeSolver(Config(), R(kB), R(kA));
  CHECK_FALSE(EstimateExpectedValue(*different, policy, policy, 1, 17).ok());
}

TEST_CASE("zero average mass extracts as uniform policy") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(1);
  CfrState& state = CFRSolverTestAccess::state(*solver);
  std::fill(state.strategy_sum.begin(), state.strategy_sum.end(), 0.0f);
  const auto extracted = solver->extract_average_policy();
  REQUIRE(extracted.ok());
  for (const auto& [key, row] : extracted->rows) {
    std::vector<float> probabilities(row.action_count);
    REQUIRE(extracted->strategy(key, absl::MakeSpan(probabilities)));
    for (float probability : probabilities) {
      CHECK(probability == doctest::Approx(1.0 / row.action_count));
    }
  }
}

TEST_CASE("lossy average policy respects its serialized byte budget") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(4);
  constexpr size_t kBudget = 256;
  const auto extracted = solver->extract_average_policy(kBudget);
  REQUIRE(extracted.ok());
  CHECK(extracted->rows.size() < solver->get_info_set_count());

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  REQUIRE(test_tmpdir != nullptr);
  const std::filesystem::path path =
      std::filesystem::path(test_tmpdir) / "lossy.policy";
  REQUIRE(SavePolicy(*extracted, path).ok());
  CHECK(std::filesystem::file_size(path) <= kBudget);

  const auto repeated = solver->extract_average_policy(kBudget);
  REQUIRE(repeated.ok());
  CHECK(repeated->rows == extracted->rows);
  CHECK(repeated->probabilities == extracted->probabilities);
  CHECK_FALSE(solver->extract_average_policy(59).ok());
}

TEST_CASE("checkpoints resume training bit for bit") {
  const SolverConfig config = Config();
  auto uninterrupted = MakeSolver(config, R(kA), R(kB));
  uninterrupted->run(6);

  auto split = MakeSolver(config, R(kA), R(kB));
  split->run(3);
  const SolverCheckpoint saved = split->checkpoint();

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  REQUIRE(test_tmpdir != nullptr);
  const std::filesystem::path path =
      std::filesystem::path(test_tmpdir) / "resume.checkpoint";
  REQUIRE(SaveCheckpoint(saved, path).ok());
  const auto loaded = LoadCheckpoint(path);
  REQUIRE(loaded.ok());
  CHECK(loaded->format_version == saved.format_version);
  CHECK(loaded->model == saved.model);
  CHECK(loaded->rng == saved.rng);
  CheckSameState(loaded->state, saved.state);

  auto resumed = MakeSolver(config, R(kA), R(kB));
  REQUIRE(resumed->restore(*loaded).ok());
  resumed->run(3);
  CheckSameState(CFRSolverTestAccess::state(*uninterrupted),
                 CFRSolverTestAccess::state(*resumed));
  CHECK(uninterrupted->checkpoint().rng == resumed->checkpoint().rng);
}

TEST_CASE("checkpoint restore rejects mismatched and corrupt state") {
  auto solver = MakeSolver(Config(), R(kA), R(kB));
  solver->run(2);
  const SolverCheckpoint valid = solver->checkpoint();

  auto different = MakeSolver(Config(), R(kB), R(kA));
  CHECK_FALSE(different->restore(valid).ok());

  SolverCheckpoint bad_offset = valid;
  REQUIRE_FALSE(bad_offset.state.rows.empty());
  ++bad_offset.state.rows.begin()->second.action_offset;
  CHECK_FALSE(solver->restore(std::move(bad_offset)).ok());

  SolverCheckpoint bad_size = valid;
  REQUIRE_FALSE(bad_size.state.regret_sum.empty());
  bad_size.state.regret_sum.pop_back();
  CHECK_FALSE(solver->restore(std::move(bad_size)).ok());

  SolverCheckpoint bad_value = valid;
  bad_value.state.regret_sum[0] =
      std::numeric_limits<float>::infinity();
  CHECK_FALSE(SaveCheckpoint(bad_value, "invalid.checkpoint").ok());
}

TEST_CASE("approximate responses are reproducible and respect infosets") {
  ComboRange opponent_range;
  opponent_range.add(kB);
  opponent_range.add(kC);
  auto game = MakeSolver(Config(), R(kA), opponent_range);
  game->run(20);
  const auto opponent = game->extract_average_policy();
  REQUIRE(opponent.ok());

  const SerializedRngState rng_before = game->checkpoint().rng;
  const BestResponseConfig config{30, 20, 91};
  const auto first = TrainApproximateBestResponse(
      *game, Player::A, *opponent, config);
  const auto second = TrainApproximateBestResponse(
      *game, Player::A, *opponent, config);
  REQUIRE(first.ok());
  REQUIRE(second.ok());
  CHECK(first->response_policy.rows == second->response_policy.rows);
  CHECK(first->response_policy.probabilities ==
        second->response_policy.probabilities);
  CHECK(first->value == second->value);
  CHECK(first->standard_error == second->standard_error);
  CHECK(first->opponent_policy_lookups > 0);
  CHECK(game->checkpoint().rng == rng_before);

  size_t root_rows = 0;
  for (const auto& [key, row] : first->response_policy.rows) {
    (void)row;
    const auto* node =
        std::get_if<DecisionNode>(&game->history_tree().nodes[
            key.history.index()]);
    REQUIRE(node != nullptr);
    CHECK(node->state.actor == Player::A);
    root_rows += key.history == game->history_tree().root ? 1 : 0;
  }
  CHECK(root_rows == 1);
}

TEST_CASE("approximate response continues after infoset capacity") {
  auto game = MakeSolver(Config(true, 1), R(kA), R(kB));
  game->run(1);
  const auto opponent = game->extract_average_policy();
  REQUIRE(opponent.ok());
  const auto response = TrainApproximateBestResponse(
      *game, Player::A, *opponent, BestResponseConfig{10, 2, 7});
  REQUIRE(response.ok());
  CHECK(response->training_iterations_completed == 10);
  CHECK(response->response_policy.rows.size() == 1);
}

TEST_CASE("approximate response learns a profitable shared action") {
  auto game = MakeSolver(Config(), R(kA), R(kB), WinningRiverRoot());
  const Policy opponent = PassiveCallingPolicy(*game, kB);
  Policy uniform;
  uniform.model = game->model_fingerprint();
  const auto baseline = EstimateExpectedValue(
      *game, uniform, opponent, 1, 11);
  const auto response = TrainApproximateBestResponse(
      *game, Player::A, opponent, BestResponseConfig{100, 1, 11});
  REQUIRE(baseline.ok());
  REQUIRE(response.ok());
  CHECK(response->value >= baseline->mean);
  CHECK(response->value > 7.5);

  const PublicPosition position = PublicPosition::Root(
      game->card_abstraction(), StreetKind::River,
      game->solve_spec().root.board);
  const InfoSetKey root_key{
      game->history_tree().root, position.observation(),
      ObservePrivate(game->card_abstraction(), kA, position)};
  const PolicyRow root_row = response->response_policy.rows.at(root_key);
  for (uint8_t action = 0; action < root_row.action_count; ++action) {
    const GameAction game_action = game->history_tree()
        .edges[std::get<DecisionNode>(game->history_tree().nodes[0])
                   .edges.begin + action]
        .action;
    if (game_action.kind == ActionKind::AllIn) {
      CHECK(response->response_policy.probabilities[
                root_row.action_offset + action] > 0.95f);
    }
  }
}

TEST_CASE("exploitability reports both responder perspectives") {
  auto game = MakeSolver(Config(), R(kA), R(kB), WinningRiverRoot());
  const auto initial_policy = game->extract_average_policy();
  REQUIRE(initial_policy.ok());
  const auto initial = EstimateExploitability(
      *game, *initial_policy, BestResponseConfig{200, 2, 23});
  REQUIRE(initial.ok());

  game->run(200);
  const auto policy = game->extract_average_policy();
  REQUIRE(policy.ok());
  const auto estimate = EstimateExploitability(
      *game, *policy, BestResponseConfig{200, 2, 23});
  REQUIRE(estimate.ok());
  CHECK(estimate->player_a_response.responder == Player::A);
  CHECK(estimate->player_b_response.responder == Player::B);
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
