#include "src/evaluation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "src/hand_evaluator.h"

namespace poker {
StrategyLookup MakeStrategyLookup(const Policy& policy) {
  return [&policy](InfoSetKey key, std::span<float> output) {
    return policy.strategy(key, output);
  };
}

namespace {

bool LookupOrUniform(const StrategyLookup& lookup,
                     InfoSetKey key,
                     std::span<float> probabilities) {
  if (lookup(key, probabilities)) return true;
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0f / static_cast<float>(probabilities.size()));
  }
  return false;
}

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

ProfileEstimate EstimateProfile(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage = false,
    bool sample_actions = false) {
  std::mt19937 rng = MakeEvaluationRng(seed);
  const std::array<const StrategyLookup*, kPlayerCount> policies = {
      &player_a, &player_b};
  EvaluationCounters counters;
  counters.measure_reach_coverage = measure_reach_coverage;
  double mean = 0.0;
  double squared_error = 0.0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Deal deal = deals.sample(rng);
    std::array<PrivateObservationId, kPlayerCount> initial_observations;
    for (Player player : {Player::A, Player::B}) {
      initial_observations[Index(player)] =
          ObservePrivate(deal.hand(player), initial_public);
    }
    std::optional<int8_t> initial_showdown;
    if (initial_public.board().count() == kMaxBoardCards) {
      initial_showdown = static_cast<int8_t>(CompareHands(
          deal.hand(Player::A), deal.hand(Player::B),
          initial_public.board()));
    }

    auto evaluate = [&](auto&& self,
                        HistoryId history_id,
                        const PublicPosition& public_state,
                        std::array<double, kPlayerCount> reach,
                        std::array<PrivateObservationId, kPlayerCount>
                            private_observations,
                        std::optional<int8_t> showdown_comparison) -> double {
      while (true) {
        const HistoryNode& node = history.nodes[Index(history_id)];
        if (const auto* fold = std::get_if<FoldTerminalState>(&node.state)) {
          return TerminalUtility(*fold, Player::A);
        }
        if (const auto* showdown = std::get_if<ShowdownState>(&node.state)) {
          assert(showdown_comparison.has_value());
          return TerminalUtilityFromComparison(
              *showdown, *showdown_comparison, Player::A);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          double value = 0.0;
          for (int chance_sample = 0;
               chance_sample < solver_config.chance_samples;
               ++chance_sample) {
            const auto cards = SampleStreetCards(
                chance->data.street, public_state.board(),
                deal.blocked_mask(), rng);
            assert(cards.ok());
            const HistoryId child_history =
                history.children[node.children_begin];
            const PublicPosition child_public(
                solver_config.card_abstraction,
                DealCards(public_state.board(), *cards));
            auto child_observations = private_observations;
            auto child_showdown = showdown_comparison;
            if (child_public.board().count() == kMaxBoardCards) {
              child_showdown = static_cast<int8_t>(CompareHands(
                  deal.hand(Player::A), deal.hand(Player::B),
                  child_public.board()));
            }
            const HistoryNode& child_node = history.nodes[Index(child_history)];
            if (std::holds_alternative<DecisionState>(child_node.state)) {
              for (Player player : {Player::A, Player::B}) {
                auto& observation = child_observations[Index(player)];
                observation = ObservePrivate(
                    deal.hand(player), child_public, observation);
              }
            }
            value += self(self, child_history, child_public, reach,
                          child_observations, child_showdown);
          }
          return value / solver_config.chance_samples;
        }

        const DecisionState& decision = std::get<DecisionState>(node.state);
        const size_t player = Index(decision.actor);
        const uint8_t action_count = node.child_count;
        const InfoSetKey key{
            public_state.observation(), history_id,
            private_observations[player]};
        const double decision_reach = reach[0] * reach[1];
        if (counters.measure_reach_coverage && decision_reach > 0.0) {
          counters.reach_by_info_set[key] += decision_reach;
        }
        ++counters.lookups[player];
        counters.weighted_lookups[player] += decision_reach;
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        if (!LookupOrUniform(*policies[player], key, strategy)) {
          ++counters.missing[player];
          counters.weighted_missing[player] += decision_reach;
        }

        if (sample_actions) {
          float action_sample = std::uniform_real_distribution<float>{}(rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count &&
                 action_sample >= probabilities[sampled_action]) {
            action_sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history_id = history.children[node.children_begin + sampled_action];
          continue;
        }

        double value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          auto child_reach = reach;
          child_reach[player] *= probabilities[action];
          const HistoryId child =
              history.children[node.children_begin + action];
          value += probabilities[action] * self(
              self, child, public_state, child_reach,
              private_observations, showdown_comparison);
        }
        return value;
      }
    };
    const double value = evaluate(
        evaluate, HistoryId{}, initial_public, {1.0, 1.0},
        initial_observations, initial_showdown);
    const double delta = value - mean;
    mean += delta / static_cast<double>(sample + 1);
    squared_error += delta * (value - mean);
  }
  const double sample_count = static_cast<double>(samples);
  const double standard_error = samples > 1
      ? std::sqrt(squared_error / (sample_count - 1.0) / sample_count)
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

}  // namespace

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    const StrategyLookup& player_a,
    const StrategyLookup& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (samples == 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  return EstimateProfile(
      config, deals, history, initial_public, player_a, player_b, samples, seed,
      measure_reach_coverage, sample_actions).value;
}

absl::StatusOr<ValueEstimate> EstimateExpectedValue(
    const SolverConfig& config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const Policy& player_a,
    const Policy& player_b,
    uint64_t samples,
    uint64_t seed,
    bool measure_reach_coverage,
    bool sample_actions) {
  if (player_a.model != model || player_b.model != model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateExpectedValue(
      config, deals, history, initial_public, MakeStrategyLookup(player_a),
      MakeStrategyLookup(player_b), samples, seed, measure_reach_coverage,
      sample_actions);
}

namespace {

absl::StatusOr<BestResponseResult> TrainResponse(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    Player responder,
    const StrategyLookup& opponent,
    const StrategyLookup* responder_fallback,
    const BestResponseConfig& config) {
  if (config.training_iterations == 0 || config.evaluation_samples == 0) {
    return absl::InvalidArgumentError(
        "best-response iteration counts must be positive");
  }
  CfrState response_state(solver_config, true);
  std::mt19937 rng = MakeEvaluationRng(config.seed);
  uint64_t opponent_lookups = 0;
  uint64_t missing_opponent_lookups = 0;
  BestResponseResult result;
  while (response_state.iterations < config.training_iterations) {
    const Deal deal = deals.sample(rng);
    std::array<PrivateObservationId, kPlayerCount> initial_observations;
    for (Player player : {Player::A, Player::B}) {
      initial_observations[Index(player)] =
          ObservePrivate(deal.hand(player), initial_public);
    }
    std::optional<int8_t> initial_showdown;
    if (initial_public.board().count() == kMaxBoardCards) {
      initial_showdown = static_cast<int8_t>(CompareHands(
          deal.hand(Player::A), deal.hand(Player::B),
          initial_public.board()));
    }

    auto cfr = [&](auto&& self,
                   HistoryId history_id,
                   const PublicPosition& public_state,
                   std::array<double, kPlayerCount> reach,
                   std::array<PrivateObservationId, kPlayerCount>
                       private_observations,
                   std::optional<int8_t> showdown_comparison) -> double {
      while (true) {
        const HistoryNode& node = history.nodes[Index(history_id)];
        if (const auto* fold = std::get_if<FoldTerminalState>(&node.state)) {
          return TerminalUtility(*fold, Player::A);
        }
        if (const auto* showdown = std::get_if<ShowdownState>(&node.state)) {
          assert(showdown_comparison.has_value());
          return TerminalUtilityFromComparison(
              *showdown, *showdown_comparison, Player::A);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          double value = 0.0;
          for (int chance_sample = 0;
               chance_sample < solver_config.chance_samples;
               ++chance_sample) {
            const auto cards = SampleStreetCards(
                chance->data.street, public_state.board(),
                deal.blocked_mask(), rng);
            assert(cards.ok());
            const HistoryId child_history =
                history.children[node.children_begin];
            const PublicPosition child_public(
                solver_config.card_abstraction,
                DealCards(public_state.board(), *cards));
            auto child_observations = private_observations;
            auto child_showdown = showdown_comparison;
            if (child_public.board().count() == kMaxBoardCards) {
              child_showdown = static_cast<int8_t>(CompareHands(
                  deal.hand(Player::A), deal.hand(Player::B),
                  child_public.board()));
            }
            const HistoryNode& child_node = history.nodes[Index(child_history)];
            if (std::holds_alternative<DecisionState>(child_node.state)) {
              for (Player player : {Player::A, Player::B}) {
                auto& observation = child_observations[Index(player)];
                observation = ObservePrivate(
                    deal.hand(player), child_public, observation);
              }
            }
            value += self(self, child_history, child_public, reach,
                          child_observations, child_showdown);
          }
          return value / solver_config.chance_samples;
        }

        const DecisionState& decision = std::get<DecisionState>(node.state);
        const Player player = decision.actor;
        const size_t player_index = Index(player);
        const uint8_t action_count = node.child_count;
        const InfoSetKey key{
            public_state.observation(), history_id,
            private_observations[player_index]};
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        std::optional<uint32_t> offset;
        if (player == responder) {
          offset = response_state.find_or_create(key, action_count);
          if (offset || responder_fallback == nullptr) {
            response_state.strategy(
                response_state.regret_sum, offset, strategy);
          } else {
            LookupOrUniform(*responder_fallback, key, strategy);
          }
        } else {
          ++opponent_lookups;
          if (!LookupOrUniform(opponent, key, strategy)) {
            ++missing_opponent_lookups;
          }
        }

        if (config.external_sampling && player != responder) {
          float action_sample = std::uniform_real_distribution<float>{}(rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count &&
                 action_sample >= probabilities[sampled_action]) {
            action_sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history_id = history.children[node.children_begin + sampled_action];
          continue;
        }

        std::array<double, kMaxActionsPerNode> action_values;
        double node_value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          auto child_reach = reach;
          child_reach[player_index] *= probabilities[action];
          const HistoryId child =
              history.children[node.children_begin + action];
          action_values[action] = self(
              self, child, public_state, child_reach,
              private_observations, showdown_comparison);
          node_value += probabilities[action] * action_values[action];
        }
        if (player != responder || !offset) return node_value;

        const double utility_sign = player == Player::A ? 1.0 : -1.0;
        const double opponent_reach =
            config.external_sampling
                ? 1.0
                : reach[Index(Opponent(player))];
        for (uint8_t action = 0; action < action_count; ++action) {
          response_state.add_regret(
              *offset, action,
              static_cast<float>(
                  opponent_reach * utility_sign *
                  (action_values[action] - node_value)));
        }
        response_state.add_strategy(
            *offset, strategy,
            reach[player_index] *
                static_cast<double>(response_state.iterations + 1));
        return node_value;
      }
    };
    const double value = cfr(
        cfr, HistoryId{}, initial_public, {1.0, 1.0},
        initial_observations, initial_showdown);
    response_state.cumulative_root_utility += value;
    ++response_state.iterations;
  }

  auto response = ExtractAveragePolicy(response_state, history, model);
  if (!response.ok()) return response.status();
  result.response_policy = std::move(*response);
  const uint64_t evaluation_seed = config.seed ^ 0x9e3779b97f4a7c15ULL;
  const StrategyLookup response_lookup = [&result, responder_fallback](
      InfoSetKey key, std::span<float> output) {
    if (result.response_policy.strategy(key, output)) return true;
    return responder_fallback != nullptr &&
           (*responder_fallback)(key, output);
  };
  const StrategyLookup& player_a =
      responder == Player::A ? response_lookup : opponent;
  const StrategyLookup& player_b =
      responder == Player::B ? response_lookup : opponent;
  const ProfileEstimate estimate = EstimateProfile(
      solver_config, deals, history, initial_public, player_a, player_b,
      config.evaluation_samples, evaluation_seed, false,
      config.external_sampling);
  result.value = responder == Player::A
                     ? estimate.value.mean
                     : -estimate.value.mean;
  result.standard_error = estimate.value.standard_error;
  const size_t opponent_index = Index(Opponent(responder));
  result.opponent_policy_lookups =
      opponent_lookups + estimate.counters.lookups[opponent_index];
  result.missing_opponent_lookups =
      missing_opponent_lookups +
      estimate.counters.missing[opponent_index];
  const size_t responder_index = Index(responder);
  result.response_policy_lookups = estimate.counters.lookups[responder_index];
  result.missing_response_lookups = estimate.counters.missing[responder_index];
  return result;
}

}  // namespace

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    Player responder,
    const StrategyLookup& opponent,
    const BestResponseConfig& config) {
  return TrainResponse(
      solver_config, deals, history, initial_public, model, responder,
      opponent, nullptr, config);
}

absl::StatusOr<BestResponseResult> TrainApproximateBestResponse(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    Player responder,
    const Policy& opponent,
    const BestResponseConfig& config) {
  if (opponent.model != model) {
    return absl::FailedPreconditionError(
        "opponent policy model does not match game");
  }
  return TrainApproximateBestResponse(
      solver_config, deals, history, initial_public, model, responder,
      MakeStrategyLookup(opponent), config);
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const StrategyLookup& policy,
    const BestResponseConfig& config) {
  auto player_a = TrainResponse(
      solver_config, deals, history, initial_public, model, Player::A,
      policy, &policy, config);
  if (!player_a.ok()) return player_a.status();

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(
      solver_config, deals, history, initial_public, model, Player::B,
      policy, &policy, player_b_config);
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = player_a->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(*player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitabilityParallel(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const StrategyLookupFactory& policy_factory,
    const BestResponseConfig& config) {
  if (!policy_factory) {
    return absl::InvalidArgumentError("policy factory is empty");
  }
  const std::array<StrategyLookup, kPlayerCount> policies = {
      policy_factory(), policy_factory()};
  if (!policies[0] || !policies[1]) {
    return absl::InvalidArgumentError("policy factory returned an empty lookup");
  }
  std::optional<absl::StatusOr<BestResponseResult>> player_a;
  std::thread player_a_thread([&] {
    player_a.emplace(TrainResponse(
        solver_config, deals, history, initial_public, model, Player::A,
        policies[Index(Player::A)], &policies[Index(Player::A)], config));
  });

  BestResponseConfig player_b_config = config;
  player_b_config.seed ^= 0xd1b54a32d192ed03ULL;
  auto player_b = TrainResponse(
      solver_config, deals, history, initial_public, model, Player::B,
      policies[Index(Player::B)], &policies[Index(Player::B)],
      player_b_config);
  player_a_thread.join();
  assert(player_a.has_value());
  if (!player_a->ok()) return player_a->status();
  if (!player_b.ok()) return player_b.status();

  const double nash_conv = (*player_a)->value + player_b->value;
  return ExploitabilityEstimate{
      std::move(**player_a), std::move(*player_b),
      nash_conv, 0.5 * nash_conv};
}

absl::StatusOr<ExploitabilityEstimate> EstimateExploitability(
    const SolverConfig& solver_config,
    const DealDistribution& deals,
    const HistoryTree& history,
    const PublicPosition& initial_public,
    ModelFingerprint model,
    const Policy& policy,
    const BestResponseConfig& config) {
  if (policy.model != model) {
    return absl::FailedPreconditionError(
        "policy model does not match game");
  }
  return EstimateExploitability(
      solver_config, deals, history, initial_public, model,
      MakeStrategyLookup(policy), config);
}

}  // namespace poker
