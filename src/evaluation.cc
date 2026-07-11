#include "src/evaluation.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace poker {
namespace {

struct EvaluationFrame {
  std::array<PrivateObservationId, kPlayerCount> private_observations = {};
};

struct EvaluationCounters {
  uint64_t lookups = 0;
  uint64_t missing = 0;
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
          PublicPosition::Root(spec.config.card_abstraction(),
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
      config.card_abstraction(), Data(child.betting).street, child.board);
  return position;
}

EvaluationFrame InitialFrame(const CFRSolver& game,
                             const Position& position,
                             const Deal& deal) {
  EvaluationFrame frame;
  const CardAbstractionConfig& config =
      game.solve_spec().config.card_abstraction();
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] =
        ObservePrivate(config, hand, position.public_state);
  }
  return frame;
}

void AdvancePrivateObservations(const CFRSolver& game,
                                EvaluationFrame& frame,
                                const Position& child,
                                const Deal& deal) {
  const CardAbstractionConfig& config =
      game.solve_spec().config.card_abstraction();
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] = AdvancePrivateObservation(
        config, frame.private_observations[player], hand,
        child.public_state);
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
      return TerminalUtility(node.state, Player::kA);
    } else if constexpr (std::is_same_v<Node, ShowdownNode>) {
      const auto* board =
          std::get_if<RiverBoard>(&position.public_state.board());
      assert(board != nullptr);
      return TerminalUtility(node.state, *board, deal.hand(Player::kA),
                             deal.hand(Player::kB));
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
      ++counters.lookups;
      if (!policies[player]->strategy(key, absl::MakeSpan(probabilities))) {
        ++counters.missing;
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
  return ValueEstimate{mean, standard_error, samples, counters.lookups,
                       counters.missing};
}

}  // namespace poker
