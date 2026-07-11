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
      PublicCardMode::kExactCanonical,
      PrivateCardMode::kExactCanonical,
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

  const SolverConfig defaults = SolverConfig::Default();
  CHECK(defaults.card_abstraction().public_mode == PublicCardMode::kTexture);
  CHECK(defaults.card_abstraction().private_mode == PrivateCardMode::kCoarse);
}

const ComboId kA = H(14, S::kHearts, 14, S::kSpades);
const ComboId kB = H(13, S::kClubs, 13, S::kDiamonds);

TEST_CASE("small exact solver baseline is deterministic") {
  auto solver = MakeSolver(
      Config(), UniformComboRange(), UniformComboRange());
  solver->run(10);

  CHECK(solver->get_history_count() == 417);
  CHECK(solver->get_info_set_count() == 720);
  CHECK(solver->get_cfr_update_count() == 1440);
  CHECK(solver->get_expected_value(Player::kA) ==
        doctest::Approx(-1.01572));
}

TEST_CASE("model fingerprints are stable and cover solve ranges") {
  auto first = MakeSolver(Config(), R(kA), R(kB));
  auto second = MakeSolver(Config(), R(kA), R(kB));
  auto changed = MakeSolver(Config(), R(kB), R(kA));
  CHECK(first->model_fingerprint() == second->model_fingerprint());
  CHECK(first->model_fingerprint() != changed->model_fingerprint());
  CHECK(Hex(first->model_fingerprint()) ==
        "dbe9ae604cee3b303179e731f35dfbae"
        "478988c4fa4633fa5ba3b92c256af99c");
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
  CHECK(std::isfinite(solver->get_expected_value(Player::kA)));
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
  root.betting = Apply(root.betting, {ActionKind::kCall, 2});
  root.betting = Apply(root.betting, {ActionKind::kCheck, 0});
  const std::array<Card, 3> flop = {
      Card(Rank::kTwo, S::kHearts), Card(Rank::kSeven, S::kDiamonds),
      Card(Rank::kQueen, S::kClubs)};
  root = DealChance(root, flop, rules);

  auto solver = MakeSolver(config, R(kA), R(kB), root);
  solver->run(2);
  const HistoryTree& tree = CFRSolverTestAccess::history(*solver);
  const Player player =
      std::get<DecisionNode>(tree.nodes[tree.root.index()]).state.actor;
  const ComboId hand = player == Player::kA ? kA : kB;
  const PublicPosition public_state = PublicPosition::Root(
      config.card_abstraction(), Data(root.betting).street, root.board);
  const PrivateObservationId private_id = ObservePrivate(
      config.card_abstraction(), hand, public_state);
  CHECK(CFRSolverTestAccess::state(*solver).rows.contains(
      {tree.root, public_state.observation(), private_id}));
}

TEST_CASE("infoset caps stop after completing the current iteration") {
  auto solver = MakeSolver(Config(true, 1), R(kA), R(kB));
  const TrainingResult result = solver->run(2);
  CHECK(result.iterations_completed == 1);
  CHECK(result.stop_reason == TrainingStopReason::kInfoSetLimit);
  CHECK(solver->get_iterations_run() == 1);
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

}  // namespace
}  // namespace poker
