#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "src/cfr_traversal.h"

namespace poker {
namespace {

struct EvaluationCounters {
  std::array<uint64_t, kPlayerCount> lookups = {};
  std::array<uint64_t, kPlayerCount> missing = {};
  std::array<double, kPlayerCount> weighted_lookups = {};
  std::array<double, kPlayerCount> weighted_missing = {};
  absl::flat_hash_map<InfoSetKey, double> reach_by_info_set;
  bool measure_reach_coverage = false;
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

double TraverseProfile(const CompiledGame& game,
                       const std::array<const Policy*, kPlayerCount>& policies,
                       Position position,
                       const internal::TraversalFrame& frame,
                       const Deal& deal,
                       std::mt19937& rng,
                       EvaluationCounters& counters,
                       double reach) {
  const HistoryTree& history = game.history;
  const HistoryNode& node = history.nodes[Index(position.history)];
  if (const auto* state = std::get_if<FoldTerminalState>(&node.state)) {
    return TerminalUtility(*state, Player::A);
  }
  if (const auto* state = std::get_if<ShowdownState>(&node.state)) {
    const Board& board = position.public_state.board();
    return TerminalUtility(*state, board, deal.hand(Player::A),
                           deal.hand(Player::B));
  }
  if (std::holds_alternative<ChanceState>(node.state)) {
    double value = 0.0;
    const int samples = game.spec.config.chance_samples;
    for (int sample = 0; sample < samples; ++sample) {
      const Position child = internal::SampleChanceChild(
          game, node, position.public_state, deal, rng);
      internal::TraversalFrame child_frame = frame;
      internal::AdvancePrivateObservations(game, child_frame, deal, child);
      value += TraverseProfile(game, policies, child, child_frame, deal,
                               rng, counters, reach);
    }
    return value / samples;
  }

  const DecisionState& decision = std::get<DecisionState>(node.state);
  const size_t player = Index(decision.actor);
  const InfoSetKey key{
      position.public_state.observation(), position.history,
      frame.private_observations[player]};
  if (counters.measure_reach_coverage && reach > 0.0) {
    counters.reach_by_info_set[key] += reach;
  }
  absl::InlinedVector<float, 8> probabilities(node.child_count, 0.0f);
  ++counters.lookups[player];
  counters.weighted_lookups[player] += reach;
  if (!policies[player]->strategy(key, absl::MakeSpan(probabilities))) {
    ++counters.missing[player];
    counters.weighted_missing[player] += reach;
  }
  double value = 0.0;
  for (uint8_t action = 0; action < node.child_count; ++action) {
    Position child = position;
    child.history = history.children[node.children_begin + action];
    value += probabilities[action] * TraverseProfile(
        game, policies, child, frame, deal, rng, counters,
        reach * probabilities[action]);
  }
  return value;
}

ProfileEstimate EstimateProfile(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const Position root = internal::RootPosition(game);
  const std::array<const Policy*, kPlayerCount> policies = {
      &player_a, &player_b};
  EvaluationCounters counters;
  counters.measure_reach_coverage = measure_reach_coverage;
  double mean = 0.0;
  double squared_error = 0.0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Deal deal = game.deals.sample(rng);
    const internal::TraversalFrame frame =
        internal::InitialTraversalFrame(game, deal, root);
    const double value = TraverseProfile(
        game, policies, root, frame, deal, rng, counters, 1.0);
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
  const double weighted_lookups =
      counters.weighted_lookups[0] + counters.weighted_lookups[1];
  const double weighted_missing =
      counters.weighted_missing[0] + counters.weighted_missing[1];
  std::vector<double> reaches;
  reaches.reserve(counters.reach_by_info_set.size());
  for (const auto& [key, reach] : counters.reach_by_info_set) {
    (void)key;
    reaches.push_back(reach);
  }
  counters.reach_by_info_set = {};
  std::ranges::sort(reaches, std::greater<>());
  size_t rows_for_99_percent = 0;
  double covered_reach = 0.0;
  while (rows_for_99_percent < reaches.size() &&
         covered_reach < 0.99 * weighted_lookups) {
    covered_reach += reaches[rows_for_99_percent++];
  }
  return {{mean, standard_error, lookups, missing,
           weighted_lookups, weighted_missing,
           reaches.size(), rows_for_99_percent},
          counters};
}

struct ResponseTrainingContext {
  Player responder;
  const Policy& opponent;
  CfrState& response;
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
};

double TraverseResponse(const CompiledGame& game,
                        Position position,
                        const internal::TraversalFrame& frame,
                        const Deal& deal,
                        std::mt19937& rng,
                        ResponseTrainingContext& context) {
  const HistoryTree& history = game.history;
  const HistoryNode& node = history.nodes[Index(position.history)];
  if (const auto* state = std::get_if<FoldTerminalState>(&node.state)) {
    return TerminalUtility(*state, Player::A);
  }
  if (const auto* state = std::get_if<ShowdownState>(&node.state)) {
    const Board& board = position.public_state.board();
    return TerminalUtility(*state, board, deal.hand(Player::A),
                           deal.hand(Player::B));
  }
  if (std::holds_alternative<ChanceState>(node.state)) {
    double value = 0.0;
    const int samples = game.spec.config.chance_samples;
    for (int sample = 0; sample < samples; ++sample) {
      const Position child = internal::SampleChanceChild(
          game, node, position.public_state, deal, rng);
      internal::TraversalFrame child_frame = frame;
      internal::AdvancePrivateObservations(game, child_frame, deal, child);
      value += TraverseResponse(game, child, child_frame, deal, rng,
                                context);
    }
    return value / samples;
  }

  const DecisionState& decision = std::get<DecisionState>(node.state);
  const Player actor = decision.actor;
  const size_t player = Index(actor);
  const InfoSetKey key{
      position.public_state.observation(), position.history,
      frame.private_observations[player]};
  const bool responder_turn = actor == context.responder;
  std::optional<size_t> offset;
  absl::InlinedVector<float, 8> probabilities(node.child_count, 0.0f);
  if (responder_turn) {
    offset = context.response.find_or_create(key, node.child_count);
    context.response.strategy(context.response.regret_sum, offset,
                              absl::MakeSpan(probabilities));
  } else {
    ++context.opponent_lookups;
    if (!context.opponent.strategy(key, absl::MakeSpan(probabilities))) {
      ++context.missing_opponent_lookups;
    }
  }

  absl::InlinedVector<double, 8> action_values(node.child_count, 0.0);
  double node_value = 0.0;
  for (uint8_t action = 0; action < node.child_count; ++action) {
    internal::TraversalFrame child_frame = frame;
    child_frame.reach[player] *= probabilities[action];
    Position child = position;
    child.history = history.children[node.children_begin + action];
    action_values[action] = TraverseResponse(
        game, child, child_frame, deal, rng, context);
    node_value += probabilities[action] * action_values[action];
  }
  if (!responder_turn || !offset) return node_value;

  const double utility_sign = actor == Player::A ? 1.0 : -1.0;
  const double opponent_reach = frame.reach[Index(Opponent(actor))];
  for (uint8_t action = 0; action < node.child_count; ++action) {
    const float regret = static_cast<float>(
        opponent_reach * utility_sign *
        (action_values[action] - node_value));
    context.response.add_regret(*offset, action, regret);
  }
  context.response.add_strategy(
      *offset, absl::MakeConstSpan(probabilities),
      frame.reach[player] * (context.response.iterations + 1));
  return node_value;
}

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const CompiledGame& game,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage) {
  if (samples == 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  if (player_a.model != game.model || player_b.model != game.model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateProfile(game, player_a, player_b, samples, seed,
                         measure_reach_coverage).value;
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const CompiledGame& game,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError(
        "best-response iteration counts must be positive");
  }
  if (opponent.model != game.model) {
    return absl::FailedPreconditionError(
        "opponent policy model does not match game");
  }

  CfrState response_state(game.spec.config, true);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  const Position root = internal::RootPosition(game);
  ResponseTrainingContext context{responder, opponent, response_state};
  BestResponseResult result;
  while (response_state.iterations < config.training_iterations) {
    const Deal deal = game.deals.sample(rng);
    internal::TraversalFrame frame =
        internal::InitialTraversalFrame(game, deal, root);
    const double value =
        TraverseResponse(game, root, frame, deal, rng, context);
    response_state.cumulative_root_utility += value;
    ++response_state.iterations;
  }

  auto response = ExtractAveragePolicy(
      response_state, game.history, game.model);
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
    const CompiledGame& game,
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

  const double nash_conv = player_a->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(*player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

}  // namespace poker
