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
  return {HistoryId{},
          PublicPosition(game.card_abstraction(), spec.root.board)};
}

Position ActionChild(const HistoryTree& history,
                     const HistoryNode& node,
                     Position position,
                     uint8_t action) {
  position.history = history.children[node.children_begin + action];
  return position;
}

Position ChanceChild(const CFRSolver& game,
                     const HistoryNode& node,
                     Position position,
                     const Deal& deal,
                     std::mt19937& rng) {
  const ChanceState& chance = std::get<ChanceState>(node.state);
  const auto cards = SampleStreetCards(
      chance.data.street, position.public_state.board(),
      deal.blocked_mask(), rng);
  assert(cards.ok());
  position.history = game.history_tree().children[node.children_begin];
  position.public_state = PublicPosition(
      game.card_abstraction(),
      DealCards(position.public_state.board(), *cards));
  return position;
}

EvaluationFrame InitialFrame(const CFRSolver& game,
                             const Position& position,
                             const Deal& deal) {
  EvaluationFrame frame;
  for (Player player : {Player::A, Player::B}) {
    const ComboId hand = deal.hand(player);
    frame.private_observations[Index(player)] =
        ObservePrivate(game.card_abstraction(), hand,
                       position.public_state.board());
  }
  return frame;
}

void AdvancePrivateObservations(const CFRSolver& game,
                                EvaluationFrame& frame,
                                const Position& child,
                                const Deal& deal) {
  for (Player player : {Player::A, Player::B}) {
    const ComboId hand = deal.hand(player);
    frame.private_observations[Index(player)] = ObservePrivate(
        game.card_abstraction(), hand, child.public_state.board());
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
  const HistoryNode& node = history.nodes[position.history.index()];
  return std::visit([&](const auto& state) -> double {
    using State = std::decay_t<decltype(state)>;
    if constexpr (std::is_same_v<State, FoldTerminalState>) {
      return TerminalUtility(state, Player::A);
    } else if constexpr (std::is_same_v<State, ShowdownState>) {
      const Board& board = position.public_state.board();
      return TerminalUtility(state, board, deal.hand(Player::A),
                             deal.hand(Player::B));
    } else if constexpr (std::is_same_v<State, ChanceState>) {
      double value = 0.0;
      const int samples = game.solve_spec().config.chance_samples;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = ChanceChild(game, node, position, deal, rng);
        EvaluationFrame child_frame = frame;
        AdvancePrivateObservations(game, child_frame, child, deal);
        value += TraverseProfile(game, policies, child, child_frame, deal,
                                 rng, counters);
      }
      return value / samples;
    } else {
      const size_t player = Index(state.actor);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player]};
      absl::InlinedVector<float, 8> probabilities(node.child_count, 0.0f);
      ++counters.lookups[player];
      if (!policies[player]->strategy(key, absl::MakeSpan(probabilities))) {
        ++counters.missing[player];
      }
      double value = 0.0;
      for (uint8_t action = 0; action < node.child_count; ++action) {
        value += probabilities[action] *
                 TraverseProfile(game, policies,
                                 ActionChild(history, node, position, action),
                                 frame, deal, rng, counters);
      }
      return value;
    }
  }, node.state);
}

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
    mean += delta / (sample + 1);
    squared_error += delta * (value - mean);
  }
  const double standard_error = samples > 1
      ? std::sqrt(squared_error /
                  (samples - 1) / samples)
      : 0.0;
  const uint64_t lookups = counters.lookups[0] + counters.lookups[1];
  const uint64_t missing = counters.missing[0] + counters.missing[1];
  return {{mean, standard_error, lookups, missing}, counters};
}

struct ResponseTrainingContext {
  Player responder = Player::A;
  const Policy& opponent;
  CfrState& state;
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
  const HistoryNode& node = history.nodes[position.history.index()];
  return std::visit([&](const auto& state) -> double {
    using State = std::decay_t<decltype(state)>;
    if constexpr (std::is_same_v<State, FoldTerminalState>) {
      return TerminalUtility(state, Player::A);
    } else if constexpr (std::is_same_v<State, ShowdownState>) {
      const Board& board = position.public_state.board();
      return TerminalUtility(state, board, deal.hand(Player::A),
                             deal.hand(Player::B));
    } else if constexpr (std::is_same_v<State, ChanceState>) {
      double value = 0.0;
      const int samples = game.solve_spec().config.chance_samples;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = ChanceChild(game, node, position, deal, rng);
        EvaluationFrame child_frame = frame;
        AdvancePrivateObservations(game, child_frame, child, deal);
        value += TraverseResponse(game, child, child_frame, deal, rng,
                                  context);
      }
      return value / samples;
    } else {
      const Player actor = state.actor;
      const size_t player = Index(actor);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player]};
      const bool responds = actor == context.responder;
      std::optional<size_t> offset;
      absl::InlinedVector<double, 8> probabilities(node.child_count, 0.0);
      if (responds) {
        offset = context.state.find_or_create(
            key, node.child_count, context.max_info_sets, true);
        context.state.strategy(context.state.regret_sum, offset,
                               absl::MakeSpan(probabilities));
      } else {
        absl::InlinedVector<float, 8> stored(node.child_count, 0.0f);
        ++context.opponent_lookups;
        if (!context.opponent.strategy(key, absl::MakeSpan(stored))) {
          ++context.missing_opponent_lookups;
        }
        std::copy(stored.begin(), stored.end(), probabilities.begin());
      }

      absl::InlinedVector<double, 8> values(node.child_count, 0.0);
      double node_value = 0.0;
      for (uint8_t action = 0; action < node.child_count; ++action) {
        EvaluationFrame child_frame = frame;
        child_frame.reach[player] *= probabilities[action];
        values[action] = TraverseResponse(
            game, ActionChild(history, node, position, action), child_frame,
            deal, rng, context);
        node_value += probabilities[action] * values[action];
      }
      if (!responds || !offset) return node_value;

      const double sign = actor == Player::A ? 1.0 : -1.0;
      const double opponent_reach = frame.reach[Index(Opponent(actor))];
      for (uint8_t action = 0; action < node.child_count; ++action) {
        const float delta = static_cast<float>(
            opponent_reach * sign * (values[action] - node_value));
        context.state.add_regret(*offset, action, delta);
      }
      context.state.add_strategy(
          *offset, absl::MakeConstSpan(probabilities),
          frame.reach[player] * (context.state.iterations + 1));
      return node_value;
    }
  }, node.state);
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
      static_cast<size_t>(solver_config.max_info_sets);

  CfrState response_state;
  response_state.reserve(solver_config, true);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  const Position root = RootPosition(game);
  ResponseTrainingContext context{
      responder, opponent, response_state, max_info_sets};
  BestResponseResult result;
  result.responder = responder;
  while (response_state.iterations < config.training_iterations) {
    const Deal deal = game.deal_distribution().sample(rng);
    EvaluationFrame frame = InitialFrame(game, root, deal);
    const double value =
        TraverseResponse(game, root, frame, deal, rng, context);
    response_state.cumulative_root_utility += value;
    ++response_state.iterations;
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
