#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace poker {
namespace {

struct EvaluationFrame {
  std::array<double, kPlayerCount> reach = {1.0, 1.0};
  std::array<PrivateObservationId, kPlayerCount> private_observations = {};
};

struct EvaluationCounters {
  std::array<uint64_t, kPlayerCount> lookups = {};
  std::array<uint64_t, kPlayerCount> missing = {};
};

struct ProfileEstimate {
  ValueEstimate value;
  EvaluationCounters counters;
};

std::mt19937 MakeEvaluationRng(uint64_t seed) {
  const std::array<uint32_t, 2> words = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  std::seed_seq sequence(words.begin(), words.end());
  return std::mt19937(sequence);
}

Position RootPosition(const CFRSolver& game) {
  const SolveSpec& spec = game.solve_spec();
  return {game.history_tree().root,
          PublicPosition::Root(game.card_abstraction(),
                               Data(spec.root.betting).street,
                               spec.root.board)};
}

Position ActionChild(const CFRSolver& game,
                     Position position,
                     uint8_t action) {
  const HistoryTree& history = game.history_tree();
  const auto* node =
      std::get_if<DecisionNode>(&history.nodes[position.history.index()]);
  assert(node != nullptr && action < node->edges.count);
  position.history = history.edges[node->edges.begin + action].child;
  return position;
}

Position ChanceChild(const CFRSolver& game,
                     Position position,
                     const Deal& deal,
                     std::mt19937& rng) {
  const HistoryTree& history = game.history_tree();
  const auto* node =
      std::get_if<ChanceNode>(&history.nodes[position.history.index()]);
  assert(node != nullptr);
  const auto cards = SampleStreetCards(
      node->state.data.street, position.public_state.board(),
      deal.blocked_mask, rng);
  assert(cards.ok());
  const SolverConfig& config = game.solve_spec().config;
  const ExactPublicState child = AdvanceChance(
      node->state, position.public_state.board(), *cards,
      BettingRules{config.big_blind()});
  position.history = node->child;
  position.public_state = position.public_state.after_chance(
      game.card_abstraction(), Data(child.betting).street, child.board);
  return position;
}

EvaluationFrame InitialFrame(const CFRSolver& game,
                             const Position& position,
                             const Deal& deal) {
  EvaluationFrame frame;
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] =
        ObservePrivate(game.card_abstraction(), hand, position.public_state);
  }
  return frame;
}

void AdvancePrivateObservations(const CFRSolver& game,
                                EvaluationFrame& frame,
                                const Position& child,
                                const Deal& deal) {
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] = ObservePrivate(
        game.card_abstraction(), hand, child.public_state);
  }
}

double TraverseProfile(const CFRSolver& game,
                       const std::array<const Policy*, kPlayerCount>& policies,
                       Position position,
                       EvaluationFrame frame,
                       const Deal& deal,
                       std::mt19937& rng,
                       EvaluationCounters& counters) {
  const HistoryTree& history = game.history_tree();
  return std::visit([&](const auto& node) -> double {
    using Node = std::decay_t<decltype(node)>;
    if constexpr (std::is_same_v<Node, FoldTerminalNode>) {
      return TerminalUtility(node.state, Player::A);
    } else if constexpr (std::is_same_v<Node, ShowdownNode>) {
      const auto* board =
          std::get_if<RiverBoard>(&position.public_state.board());
      assert(board != nullptr);
      return TerminalUtility(node.state, *board, deal.hand(Player::A),
                             deal.hand(Player::B));
    } else if constexpr (std::is_same_v<Node, ChanceNode>) {
      double value = 0.0;
      const int samples = game.solve_spec().config.chance_samples();
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = ChanceChild(game, position, deal, rng);
        EvaluationFrame child_frame = frame;
        AdvancePrivateObservations(game, child_frame, child, deal);
        value += TraverseProfile(game, policies, child, child_frame, deal,
                                 rng, counters);
      }
      return value / samples;
    } else {
      const size_t player = Index(node.state.actor);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player]};
      absl::InlinedVector<float, 8> probabilities(node.edges.count, 0.0f);
      ++counters.lookups[player];
      if (!policies[player]->strategy(key, absl::MakeSpan(probabilities))) {
        ++counters.missing[player];
      }
      double value = 0.0;
      for (uint8_t action = 0; action < node.edges.count; ++action) {
        value += probabilities[action] *
                 TraverseProfile(game, policies,
                                 ActionChild(game, position, action), frame,
                                 deal, rng, counters);
      }
      return value;
    }
  }, history.nodes[position.history.index()]);
}

}  // namespace

namespace {

ProfileEstimate EstimateProfile(
    const CFRSolver& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const Position root = RootPosition(game);
  const std::array<const Policy*, kPlayerCount> policies = {
      &player_a, &player_b};
  EvaluationCounters counters;
  double mean = 0.0;
  double squared_error = 0.0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Deal deal = game.deal_distribution().sample(rng);
    const EvaluationFrame frame = InitialFrame(game, root, deal);
    const double value = TraverseProfile(
        game, policies, root, frame, deal, rng, counters);
    const double delta = value - mean;
    mean += delta / static_cast<double>(sample + 1);
    squared_error += delta * (value - mean);
  }
  const double standard_error = samples > 1
      ? std::sqrt(squared_error /
                  static_cast<double>(samples - 1) /
                  static_cast<double>(samples))
      : 0.0;
  const uint64_t lookups = counters.lookups[0] + counters.lookups[1];
  const uint64_t missing = counters.missing[0] + counters.missing[1];
  return {{mean, standard_error, samples, lookups, missing}, counters};
}

void FillUniform(absl::Span<double> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0 / probabilities.size());
  }
}

void RegretMatch(const CfrState& state,
                 const InfoSetRow* row,
                 absl::Span<double> probabilities) {
  if (row == nullptr) {
    FillUniform(probabilities);
    return;
  }
  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const float regret = state.regret_sum[row->action_offset + action];
    probabilities[action] = std::max(0.0, static_cast<double>(regret));
    sum += probabilities[action];
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
    return;
  }
  for (double& probability : probabilities) probability /= sum;
}

std::optional<InfoSetRow> FindOrCreateResponseRow(
    CfrState& state,
    InfoSetKey key,
    uint8_t action_count,
    size_t max_info_sets) {
  const auto found = state.rows.find(key);
  if (found != state.rows.end()) return found->second;
  if (state.rows.size() >= max_info_sets) return std::nullopt;
  const size_t offset = state.regret_sum.size();
  state.regret_sum.resize(offset + action_count, 0.0f);
  state.strategy_sum.resize(offset + action_count, 0.0f);
  const InfoSetRow row{offset};
  state.rows.emplace(key, row);
  return row;
}

struct ResponseTrainingContext {
  Player responder = Player::A;
  const Policy& opponent;
  CfrState& state;
  uint64_t iteration = 0;
  size_t max_info_sets = 0;
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
};

double TraverseResponse(const CFRSolver& game,
                        Position position,
                        EvaluationFrame frame,
                        const Deal& deal,
                        std::mt19937& rng,
                        ResponseTrainingContext& context) {
  const HistoryTree& history = game.history_tree();
  return std::visit([&](const auto& node) -> double {
    using Node = std::decay_t<decltype(node)>;
    if constexpr (std::is_same_v<Node, FoldTerminalNode>) {
      return TerminalUtility(node.state, Player::A);
    } else if constexpr (std::is_same_v<Node, ShowdownNode>) {
      const auto* board =
          std::get_if<RiverBoard>(&position.public_state.board());
      assert(board != nullptr);
      return TerminalUtility(node.state, *board, deal.hand(Player::A),
                             deal.hand(Player::B));
    } else if constexpr (std::is_same_v<Node, ChanceNode>) {
      double value = 0.0;
      const int samples = game.solve_spec().config.chance_samples();
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = ChanceChild(game, position, deal, rng);
        EvaluationFrame child_frame = frame;
        AdvancePrivateObservations(game, child_frame, child, deal);
        value += TraverseResponse(game, child, child_frame, deal, rng,
                                  context);
      }
      return value / samples;
    } else {
      const Player actor = node.state.actor;
      const size_t player = Index(actor);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player]};
      const bool responds = actor == context.responder;
      std::optional<InfoSetRow> row;
      absl::InlinedVector<double, 8> probabilities(node.edges.count, 0.0);
      if (responds) {
        row = FindOrCreateResponseRow(context.state, key, node.edges.count,
                                      context.max_info_sets);
        const InfoSetRow* strategy_row = row ? &*row : nullptr;
        RegretMatch(context.state, strategy_row,
                    absl::MakeSpan(probabilities));
      } else {
        absl::InlinedVector<float, 8> stored(node.edges.count, 0.0f);
        ++context.opponent_lookups;
        if (!context.opponent.strategy(key, absl::MakeSpan(stored))) {
          ++context.missing_opponent_lookups;
        }
        std::copy(stored.begin(), stored.end(), probabilities.begin());
      }

      absl::InlinedVector<double, 8> values(node.edges.count, 0.0);
      double node_value = 0.0;
      for (uint8_t action = 0; action < node.edges.count; ++action) {
        EvaluationFrame child_frame = frame;
        child_frame.reach[player] *= probabilities[action];
        values[action] = TraverseResponse(
            game, ActionChild(game, position, action), child_frame, deal, rng,
            context);
        node_value += probabilities[action] * values[action];
      }
      if (!responds || !row) return node_value;

      const double sign = actor == Player::A ? 1.0 : -1.0;
      const double opponent_reach = frame.reach[Index(Opponent(actor))];
      for (uint8_t action = 0; action < node.edges.count; ++action) {
        const size_t index = row->action_offset + action;
        const float delta = static_cast<float>(
            opponent_reach * sign * (values[action] - node_value));
        context.state.regret_sum[index] =
            std::max(0.0f, context.state.regret_sum[index] + delta);
        context.state.strategy_sum[index] += static_cast<float>(
            frame.reach[player] * static_cast<double>(context.iteration + 1) *
            probabilities[action]);
      }
      return node_value;
    }
  }, history.nodes[position.history.index()]);
}

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CFRSolver& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed) {
  if (samples == 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  if (player_a.model != game.model_fingerprint() ||
      player_b.model != game.model_fingerprint()) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateProfile(game, player_a, player_b, samples, seed).value;
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CFRSolver& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError(
        "best-response iteration counts must be positive");
  }
  if (opponent.model != game.model_fingerprint()) {
    return absl::FailedPreconditionError(
        "opponent policy model does not match game");
  }

  const SolverConfig& solver_config = game.solve_spec().config;
  const size_t max_info_sets =
      static_cast<size_t>(solver_config.max_info_sets());
  size_t max_actions = 3;
  for (const auto& fractions :
       solver_config.bet_abstraction().pot_fractions) {
    max_actions = std::max(max_actions, fractions.size() + 3);
  }

  CfrState response_state;
  response_state.rows.reserve(max_info_sets);
  response_state.regret_sum.reserve(max_info_sets * max_actions);
  response_state.strategy_sum.reserve(max_info_sets * max_actions);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  const Position root = RootPosition(game);
  ResponseTrainingContext context{
      responder, opponent, response_state, 0, max_info_sets};
  BestResponseResult result;
  result.responder = responder;
  for (uint64_t iteration = 0; iteration < config.training_iterations;
       ++iteration) {
    const Deal deal = game.deal_distribution().sample(rng);
    EvaluationFrame frame = InitialFrame(game, root, deal);
    context.iteration = iteration;
    const double value =
        TraverseResponse(game, root, frame, deal, rng, context);
    response_state.cumulative_root_utility += value;
    ++response_state.iterations;
    ++result.training_iterations_completed;
  }

  auto response = ExtractAveragePolicy(
      response_state, game.history_tree(), game.model_fingerprint());
  if (!response.ok()) return response.status();
  result.response_policy = std::move(*response);
  const uint64_t evaluation_seed = config.seed ^ 0x9e3779b97f4a7c15ULL;
  const Policy& player_a = responder == Player::A
                               ? result.response_policy
                               : opponent;
  const Policy& player_b = responder == Player::B
                               ? result.response_policy
                               : opponent;
  const ProfileEstimate estimate = EstimateProfile(
      game, player_a, player_b, config.evaluation_samples, evaluation_seed);
  result.value = responder == Player::A
                     ? estimate.value.mean
                     : -estimate.value.mean;
  result.standard_error = estimate.value.standard_error;
  const size_t opponent_index = Index(Opponent(responder));
  result.opponent_policy_lookups =
      context.opponent_lookups + estimate.counters.lookups[opponent_index];
  result.missing_opponent_lookups =
      context.missing_opponent_lookups +
      estimate.counters.missing[opponent_index];
  return result;
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const CFRSolver& game,
    const Policy& policy,
    const BestResponseConfig& config) {
  auto player_a = TrainApproximateBestResponse(
      game, Player::A, policy, config);
  if (!player_a.ok()) return player_a.status();

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainApproximateBestResponse(
      game, Player::B, policy, player_b_config);
  if (!player_b.ok()) return player_b.status();

  ExploitabilityEstimate result;
  result.player_a_response = std::move(*player_a);
  result.player_b_response = std::move(*player_b);
  result.nash_conv =
      result.player_a_response.value + result.player_b_response.value;
  result.exploitability = 0.5 * result.nash_conv;
  return result;
}

}  // namespace poker
