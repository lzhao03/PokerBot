#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

#include "absl/types/span.h"
#include "src/hand_evaluator.h"
#include "src/solver.h"

namespace poker::internal {

enum class TraversalMode : uint8_t {
  Train,
  EvaluateCurrent,
  EvaluateAverage,
};

enum class StrategyAccess : uint8_t {
  ReadOnly,
  Writable,
};

struct TraversalFrame {
  std::array<double, kPlayerCount> reach = {1.0, 1.0};
  std::array<PrivateObservationId, kPlayerCount> private_observations = {};
  std::optional<int8_t> showdown_comparison;
};

struct TraversalContext {
  const Deal& deal;
  TraversalMode mode;
  Player update_player;
  uint64_t iteration;
  std::mt19937& rng;
  SolverStats& stats;
};

struct DecisionView {
  InfoSetKey key;
  const DecisionState& state;
  uint8_t action_count;
  uint64_t iteration;
};

template <typename Backend>
concept CfrBackend = requires(Backend& backend,
                              const DecisionView& decision,
                              StrategyAccess access,
                              absl::Span<float> probabilities,
                              typename Backend::UpdateHandle handle,
                              absl::Span<const float> regrets,
                              double weight) {
  { backend.current_strategy(decision, access, probabilities) }
      -> std::same_as<std::optional<typename Backend::UpdateHandle>>;
  { backend.average_strategy(decision, probabilities) } -> std::same_as<void>;
  { backend.record_regrets(decision, handle, regrets) } -> std::same_as<void>;
  { backend.record_strategy(decision, handle, probabilities, weight) }
      -> std::same_as<void>;
};

Position RootPosition(const SolveSpec& spec);
TraversalFrame InitialTraversalFrame(const SolveSpec& spec,
                                     const Deal& deal,
                                     const Position& position);
Position SampleChanceChild(const SolveSpec& spec,
                           const HistoryTree& history,
                           const HistoryNode& node,
                           const PublicPosition& public_state,
                           const Deal& deal,
                           std::mt19937& rng);
void AdvancePrivateObservations(const SolveSpec& spec,
                                const HistoryTree& history,
                                TraversalFrame& frame,
                                const Deal& deal,
                                const Position& child);

template <CfrBackend Backend>
double Traverse(const SolveSpec& spec,
                const HistoryTree& tree,
                HistoryId history,
                const PublicPosition& public_state,
                const TraversalFrame& frame,
                TraversalContext& context,
                Backend& backend) {
  while (true) {
    const HistoryNode& history_node = tree.nodes[Index(history)];
    const BettingState& betting_state = history_node.state;
    if (const auto* fold = std::get_if<FoldTerminalState>(&betting_state)) {
      ++context.stats.terminal_visits;
      return TerminalUtility(*fold, Player::A);
    }
    if (const auto* showdown = std::get_if<ShowdownState>(&betting_state)) {
      ++context.stats.terminal_visits;
      assert(frame.showdown_comparison.has_value());
      return TerminalUtilityFromComparison(
          *showdown, *frame.showdown_comparison, Player::A);
    }
    if (std::holds_alternative<ChanceState>(betting_state)) {
      const int samples = spec.config.chance_samples;
      context.stats.chance_samples += static_cast<uint64_t>(samples);
      double value = 0.0;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = SampleChanceChild(
            spec, tree, history_node, public_state, context.deal, context.rng);
        TraversalFrame child_frame = frame;
        if (child.public_state.board().count() == kMaxBoardCards) {
          child_frame.showdown_comparison = static_cast<int8_t>(CompareHands(
              context.deal.hand(Player::A), context.deal.hand(Player::B),
              child.public_state.board()));
        }
        AdvancePrivateObservations(spec, tree, child_frame, context.deal,
                                   child);
        value += Traverse(spec, tree, child.history, child.public_state,
                          child_frame, context, backend);
      }
      return value / samples;
    }

    const DecisionState& decision = std::get<DecisionState>(betting_state);
    const Player player = decision.actor;
    const size_t player_index = Index(player);
    const uint8_t action_count = history_node.child_count;
    const bool training = context.mode == TraversalMode::Train;
    const bool external_sampling = training && spec.config.external_sampling;
    const bool updates_regrets = training && context.update_player == player;
    const DecisionView view{
        {public_state.observation(), history,
         frame.private_observations[player_index]},
        decision,
        action_count,
        context.iteration,
    };
    std::array<float, kMaxActionsPerNode> probabilities;
    std::array<double, kMaxActionsPerNode> action_values;
    const absl::Span<float> probability_span(probabilities.data(),
                                             action_count);
    const StrategyAccess access =
        training && (updates_regrets || external_sampling)
            ? StrategyAccess::Writable
            : StrategyAccess::ReadOnly;
    std::optional<typename Backend::UpdateHandle> handle;
    if (context.mode == TraversalMode::EvaluateAverage) {
      backend.average_strategy(view, probability_span);
    } else {
      handle = backend.current_strategy(view, access, probability_span);
    }
    if (training) ++context.stats.decision_visits;

    if (external_sampling && !updates_regrets) {
      if (handle) {
        backend.record_strategy(view, *handle, probability_span, 1.0);
      }
      float sample = std::uniform_real_distribution<float>{}(context.rng);
      uint8_t sampled_action = 0;
      while (sampled_action + 1 < action_count &&
             sample >= probabilities[sampled_action]) {
        sample -= probabilities[sampled_action];
        ++sampled_action;
      }
      history = tree.children[history_node.children_begin + sampled_action];
      continue;
    }

    double node_value = 0.0;
    TraversalFrame child_frame = frame;
    for (uint8_t action = 0; action < action_count; ++action) {
      if (training && !external_sampling) {
        child_frame.reach[player_index] =
            frame.reach[player_index] * probabilities[action];
      }
      const HistoryId child =
          tree.children[history_node.children_begin + action];
      action_values[action] = Traverse(spec, tree, child, public_state,
                                       child_frame, context, backend);
      node_value += probabilities[action] * action_values[action];
    }
    if (!training || !updates_regrets || !handle) {
      return node_value;
    }

    const double utility_sign = player == Player::A ? 1.0 : -1.0;
    const double opponent_reach =
        external_sampling ? 1.0 : frame.reach[Index(Opponent(player))];
    std::array<float, kMaxActionsPerNode> regrets;
    for (uint8_t action = 0; action < action_count; ++action) {
      regrets[action] = static_cast<float>(
          opponent_reach * utility_sign *
          (action_values[action] - node_value));
    }
    backend.record_regrets(
        view, *handle, absl::Span<const float>(regrets.data(), action_count));
    if (!external_sampling) {
      const double weight = frame.reach[player_index] * (context.iteration + 1);
      backend.record_strategy(view, *handle, probability_span, weight);
    }
    return node_value;
  }
}

}  // namespace poker::internal
