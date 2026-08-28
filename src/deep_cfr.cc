#include "src/deep_cfr.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "src/neural_policy.h"

namespace poker {
namespace {

class Reservoir {
 public:
  explicit Reservoir(size_t capacity) : capacity_(capacity) { samples_.reserve(capacity); }

  void add(NeuralSample sample, std::mt19937& rng) {
    const uint64_t index = seen_++;
    if (samples_.size() < capacity_) {
      samples_.push_back(std::move(sample));
      return;
    }
    const uint64_t replacement = std::uniform_int_distribution<uint64_t>(0, index)(rng);
    if (replacement < capacity_) {
      samples_[static_cast<size_t>(replacement)] = std::move(sample);
    }
  }

  size_t size() const noexcept { return samples_.size(); }
  std::span<const NeuralSample> samples() const noexcept { return samples_; }

 private:
  size_t capacity_;
  uint64_t seen_ = 0;
  std::vector<NeuralSample> samples_;
};

uint64_t NetworkSeed(uint64_t base, uint64_t iteration, uint64_t network) {
  return base + iteration * 0x9e3779b97f4a7c15ULL + network;
}

std::mt19937 MakeRng(uint64_t seed) {
  const std::array<uint32_t, 2> words = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  std::seed_seq sequence(words.begin(), words.end());
  return std::mt19937(sequence);
}

absl::Status ValidateConfig(const DeepCfrConfig& config) {
  if (config.advantage_memory_capacity == 0 ||
      config.strategy_memory_capacity == 0) {
    return absl::InvalidArgumentError("Deep CFR memories must be nonempty");
  }
  if (config.traversals_per_player <= 0 || config.training_steps <= 0 ||
      config.policy_training_steps <= 0 || config.batch_size <= 0 ||
      config.hidden_size <= 0) {
    return absl::InvalidArgumentError("Deep CFR sizes must be positive");
  }
  if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
    return absl::InvalidArgumentError("Deep CFR learning rate must be finite and positive");
  }
  return absl::OkStatus();
}

absl::Status TorchError(const std::exception& error) { return absl::InternalError(error.what()); }

}  // namespace

struct DeepCfrSolver::Impl {
  Impl(SolverConfig solver_options,
       DealDistribution deal_distribution,
       HistoryTree history_tree,
       PublicPosition root_public,
       ModelFingerprint fingerprint,
       DeepCfrConfig deep_config)
      : solver_config(std::move(solver_options)),
        deals(std::move(deal_distribution)),
        history(std::move(history_tree)),
        initial_public(std::move(root_public)),
        model(fingerprint),
        config(deep_config),
        advantage_memory{Reservoir(config.advantage_memory_capacity),
                         Reservoir(config.advantage_memory_capacity)},
        strategy_memory(config.strategy_memory_capacity),
        advantage_network{NeuralNetwork(config.hidden_size), NeuralNetwork(config.hidden_size)},
        inference_hidden{
            std::vector<float>(static_cast<size_t>(config.hidden_size)),
            std::vector<float>(static_cast<size_t>(config.hidden_size))},
        game_rng(MakeRng(config.seed)),
        reservoir_rng(MakeRng(config.seed + 1)) {
    for (auto& cache : advantage_cache) cache.reserve(config.inference_cache_capacity);
  }

  void fill_current_strategy(Player actor, InfoSetKey key, std::span<float> probabilities) {
    const size_t player = Index(actor);
    if (!advantage_trained[player]) {
      FillUniform(probabilities);
    } else {
      const NeuralActionVector values = cached_prediction(
          advantage_network[player], advantage_cache[player], key,
          inference_hidden, config.inference_cache_capacity,
          stats.network_evaluations, stats.cache_hits);
      RegretMatch(values, probabilities);
    }
  }

  bool policy_strategy(InfoSetKey key, std::span<float> probabilities) {
    return policy_strategy(key, probabilities, policy_cache, stats.network_evaluations, stats.cache_hits);
  }

  bool policy_strategy(
      InfoSetKey key,
      std::span<float> probabilities,
      absl::flat_hash_map<InfoSetKey, NeuralActionVector>& cache,
      uint64_t& network_evaluations,
      uint64_t& cache_hits) {
    if (!policy) {
      FillUniform(probabilities);
      return false;
    }
    const auto found = cache.find(key);
    if (found != cache.end()) {
      ++cache_hits;
      std::copy_n(found->second.begin(), probabilities.size(), probabilities.begin());
      return true;
    }
    ++network_evaluations;
    NeuralActionVector values = {};
    const bool available = policy->strategy(
        history, solver_config.card_abstraction, model, key,
        std::span<float>(values.data(), probabilities.size()));
    std::copy_n(values.begin(), probabilities.size(), probabilities.begin());
    if (available && cache.size() < config.policy_cache_capacity) cache.emplace(key, values);
    return available;
  }

  StrategyLookup strategy_lookup(DeepCfrStrategy strategy) {
    return [this, strategy](InfoSetKey key, std::span<float> probabilities) {
      if (strategy == DeepCfrStrategy::Average) return policy_strategy(key, probabilities);
      const HistoryNode& node = history.nodes[Index(key.history)];
      fill_current_strategy(std::get<DecisionState>(node.state).actor, key, probabilities);
      return true;
    };
  }

  absl::StatusOr<ValueEstimate> evaluate(
      const StrategyLookup& player_a,
      const StrategyLookup& player_b,
      int samples) {
    if (samples <= 0) return absl::InvalidArgumentError("evaluation samples must be positive");
    return EstimateExpectedValue(solver_config, deals, history, initial_public, player_a, player_b,
                                 static_cast<uint64_t>(samples), config.seed + 3, false, true);
  }

  NeuralActionVector cached_prediction(
      NeuralNetwork& network,
      absl::flat_hash_map<InfoSetKey, NeuralActionVector>& cache,
      InfoSetKey key,
      std::array<std::vector<float>, 2>& hidden,
      size_t capacity,
      uint64_t& network_evaluations,
      uint64_t& cache_hits) {
    const auto found = cache.find(key);
    if (found != cache.end()) {
      ++cache_hits;
      return found->second;
    }
    ++network_evaluations;
    const NeuralActionVector values =
        PredictNeuralNetwork(network, history, solver_config.card_abstraction, key, hidden);
    if (cache.size() < capacity) cache.emplace(key, values);
    return values;
  }

  float train_network(NeuralNetwork& network, const Reservoir& memory, uint64_t seed,
                      NeuralTarget target_kind, int training_steps) {
    return FitNeuralNetwork(
        network, history, solver_config.card_abstraction, memory.samples(),
        {.seed = seed,
         .steps = training_steps,
         .batch_size = config.batch_size,
         .hidden_size = config.hidden_size,
         .learning_rate = config.learning_rate},
        target_kind);
  }

  double traverse(const Deal& deal, Player update_player, uint64_t iteration) {
    TerminalEvaluator terminal_utility(deal.hands);
    const ObservedPosition initial_position = ObservedPosition::Observe(initial_public, deal);
    ChanceSampler sample_chance{solver_config.card_abstraction, deal, game_rng};
    auto cfr = [&](auto&& self, HistoryId history_id,
                   const ObservedPosition& position) -> double {
      while (true) {
        const HistoryNode& node = history.nodes[Index(history_id)];
        if (IsTerminal(node.state)) {
          ++stats.traversal.terminal_visits;
          return terminal_utility(node.state, position.board(), update_player);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          stats.traversal.chance_samples += static_cast<uint64_t>(solver_config.chance_samples);
          const HistoryId child = history.children[node.children_begin];
          const auto& child_state = history.nodes[Index(child)].state;
          double value = 0.0;
          for (int draw = 0; draw < solver_config.chance_samples; ++draw) {
            const auto next = sample_chance(position, *chance, child_state);
            value += self(self, child, next);
          }
          return value / solver_config.chance_samples;
        }

        const DecisionState& decision = std::get<DecisionState>(node.state);
        const Player player = decision.actor;
        const uint8_t action_count = node.child_count;
        const InfoSetKey key = position.info_set_key(history_id, player);
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        fill_current_strategy(player, key, strategy);
        ++stats.traversal.decision_visits;

        if (player != update_player) {
          NeuralSample strategy_sample{key};
          std::copy(strategy.begin(), strategy.end(), strategy_sample.target.begin());
          strategy_sample.weight = static_cast<float>(iteration + 1);
          strategy_memory.add(std::move(strategy_sample), reservoir_rng);

          float sample = std::uniform_real_distribution<float>{}(game_rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count && sample >= probabilities[sampled_action]) {
            sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history_id = history.children[node.children_begin + sampled_action];
          continue;
        }

        std::array<double, kMaxActionsPerNode> action_values;
        double node_value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          const HistoryId child = history.children[node.children_begin + action];
          action_values[action] = self(self, child, position);
          node_value += probabilities[action] * action_values[action];
        }

        NeuralSample advantage_sample{key};
        advantage_sample.weight = static_cast<float>(iteration + 1);
        const BettingData& betting = decision.data;
        const float scale = 1.0f / static_cast<float>(Pot(betting) + betting.stack[0] + betting.stack[1]);
        for (uint8_t action = 0; action < action_count; ++action) {
          advantage_sample.target[action] = static_cast<float>(action_values[action] - node_value) * scale;
        }
        advantage_memory[Index(player)].add(std::move(advantage_sample), reservoir_rng);
        return node_value;
      }
    };

    return cfr(cfr, HistoryId{}, initial_position);
  }

  void run(uint64_t iterations) {
    // Brown et al. (2019), Algorithms 1-2: collect external-sampling
    // traversals, then retrain each traverser's advantage network from scratch.
    for (uint64_t outer = 0; outer < iterations; ++outer) {
      const uint64_t iteration = stats.iterations;
      for (Player player : {Player::A, Player::B}) {
        for (int traversal = 0; traversal < config.traversals_per_player; ++traversal) {
          traverse(deals.sample(game_rng), player, iteration);
          ++stats.traversals;
        }
        const size_t index = Index(player);
        stats.advantage_loss[index] = train_network(
            advantage_network[index], advantage_memory[index],
            NetworkSeed(config.seed, iteration, index),
            NeuralTarget::Advantage, config.training_steps);
        advantage_trained[index] = true;
        advantage_cache[index].clear();
      }
      ++stats.iterations;
    }

    NeuralNetwork trained_policy(config.hidden_size);
    stats.strategy_loss = train_network(
        trained_policy, strategy_memory, NetworkSeed(config.seed, stats.iterations, kPlayerCount),
        NeuralTarget::AveragePolicy, config.policy_training_steps);
    if (strategy_memory.size() > 0) {
      policy.emplace(std::move(trained_policy), model);
      stats.policy_parameter_bytes = policy->parameter_bytes();
    }
    policy_cache.clear();
    for (size_t player = 0; player < kPlayerCount; ++player) {
      stats.advantage_samples[player] = advantage_memory[player].size();
    }
    stats.strategy_samples = strategy_memory.size();
  }

  SolverConfig solver_config;
  DealDistribution deals;
  HistoryTree history;
  PublicPosition initial_public;
  ModelFingerprint model;
  DeepCfrConfig config;
  std::array<Reservoir, kPlayerCount> advantage_memory;
  Reservoir strategy_memory;
  std::array<NeuralNetwork, kPlayerCount> advantage_network;
  std::optional<NeuralPolicy> policy;
  std::array<std::vector<float>, 2> inference_hidden;
  std::array<bool, kPlayerCount> advantage_trained = {};
  std::array<absl::flat_hash_map<InfoSetKey, NeuralActionVector>, kPlayerCount> advantage_cache;
  absl::flat_hash_map<InfoSetKey, NeuralActionVector> policy_cache;
  std::mt19937 game_rng;
  std::mt19937 reservoir_rng;
  DeepCfrStats stats;
};

DeepCfrSolver::DeepCfrSolver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeepCfrSolver::~DeepCfrSolver() = default;
DeepCfrSolver::DeepCfrSolver(DeepCfrSolver&&) noexcept = default;
DeepCfrSolver& DeepCfrSolver::operator=(DeepCfrSolver&&) noexcept = default;

absl::StatusOr<DeepCfrSolver> DeepCfrSolver::Create(SolveSpec spec, DeepCfrConfig config) {
  const absl::Status deep_status = ValidateConfig(config);
  if (!deep_status.ok()) return deep_status;
  const absl::Status solver_status = ValidateSolverConfig(spec.config);
  if (!solver_status.ok()) return solver_status;
  if (!IsValidBettingData(Data(spec.root.betting))) {
    return absl::InvalidArgumentError("invalid root betting state");
  }
  if (spec.config.card_abstraction.private_kind ==
          PrivateAbstractionKind::Handcrafted36 &&
      spec.config.card_abstraction.recall_mode !=
          RecallMode::BucketHistory) {
    return absl::InvalidArgumentError("Deep CFR requires private bucket history recall");
  }
  auto deals = DealDistribution::Create(spec.ranges[Index(Player::A)],
                                        spec.ranges[Index(Player::B)]);
  if (!deals.ok()) return deals.status();
  PublicPosition initial_public(spec.config.card_abstraction, spec.root.board);
  const ModelFingerprint model = ModelFingerprintFor(spec.config, spec.root, spec.ranges);
  HistoryTree history = BuildHistoryTree(
      spec.root.betting, spec.config.betting_rules, spec.config.bet_abstraction);
  try {
    UseSingleThreadedNeuralRuntime();
    SetNeuralSeed(config.seed);
    return DeepCfrSolver(
        std::make_unique<Impl>(
            std::move(spec.config), std::move(*deals), std::move(history),
            std::move(initial_public), model, config));
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::Status DeepCfrSolver::run(uint64_t iterations) {
  if (iterations == 0) return absl::OkStatus();
  try {
    impl_->run(iterations);
    return absl::OkStatus();
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_current(int samples) {
  try {
    const StrategyLookup lookup = impl_->strategy_lookup(DeepCfrStrategy::Current);
    auto result = impl_->evaluate(lookup, lookup, samples);
    if (!result.ok()) return result.status();
    return result->mean;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_average(int samples) {
  try {
    const StrategyLookup lookup = impl_->strategy_lookup(DeepCfrStrategy::Average);
    auto result = impl_->evaluate(lookup, lookup, samples);
    if (!result.ok()) return result.status();
    return result->mean;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_average_against_uniform(Player policy_player, int samples) {
  try {
    const StrategyLookup policy = impl_->strategy_lookup(DeepCfrStrategy::Average);
    const StrategyLookup uniform = [](InfoSetKey, std::span<float>) { return false; };
    auto result = policy_player == Player::A
                      ? impl_->evaluate(policy, uniform, samples)
                      : impl_->evaluate(uniform, policy, samples);
    if (!result.ok()) return result.status();
    return policy_player == Player::A ? result->mean : -result->mean;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<DeepCfrMatchResult> DeepCfrSolver::evaluate_against_policy(
    Player policy_player, const Policy& opponent, DeepCfrStrategy strategy, int samples) {
  if (opponent.model != impl_->model) {
    return absl::FailedPreconditionError("policy model does not match game");
  }
  try {
    const StrategyLookup policy = impl_->strategy_lookup(strategy);
    uint64_t opponent_lookups = 0;
    uint64_t missing_opponent_lookups = 0;
    const StrategyLookup opponent_lookup =
        [&opponent, &opponent_lookups, &missing_opponent_lookups](
            InfoSetKey key, std::span<float> probabilities) {
          ++opponent_lookups;
          if (opponent.strategy(key, probabilities)) return true;
          ++missing_opponent_lookups;
          return false;
        };
    auto result = policy_player == Player::A
                      ? impl_->evaluate(policy, opponent_lookup, samples)
                      : impl_->evaluate(opponent_lookup, policy, samples);
    if (!result.ok()) return result.status();
    const double sign = policy_player == Player::A ? 1.0 : -1.0;
    return DeepCfrMatchResult{sign * result->mean, result->standard_error, opponent_lookups,
                              missing_opponent_lookups};
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<ExploitabilityEstimate> DeepCfrSolver::estimate_exploitability(
    const BestResponseConfig& config) {
  if (!impl_->policy) {
    return absl::FailedPreconditionError("average policy has not been trained");
  }
  try {
    std::array<absl::flat_hash_map<InfoSetKey, NeuralActionVector>,
               kPlayerCount> caches;
    std::array<uint64_t, kPlayerCount> network_evaluations = {};
    std::array<uint64_t, kPlayerCount> cache_hits = {};
    size_t next_lookup = 0;
    auto result = EstimateExploitabilityParallel(
        impl_->solver_config, impl_->deals, impl_->history,
        impl_->initial_public, impl_->model,
        [&] {
          const size_t index = next_lookup++;
          caches[index].reserve(impl_->config.policy_cache_capacity);
          return [this, &cache = caches[index],
                  &evaluations = network_evaluations[index],
                  &hits = cache_hits[index]](
                     InfoSetKey key, std::span<float> probabilities) {
            return impl_->policy_strategy(
                key, probabilities, cache, evaluations, hits);
          };
        },
        config);
    impl_->stats.network_evaluations += network_evaluations[0] + network_evaluations[1];
    impl_->stats.cache_hits += cache_hits[0] + cache_hits[1];
    return result;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::Status DeepCfrSolver::load_average_model(const std::filesystem::path& path) {
  auto loaded = LoadNeuralPolicy(path, impl_->model);
  if (!loaded.ok()) return loaded.status();
  impl_->policy.emplace(std::move(*loaded));
  impl_->stats.policy_parameter_bytes = impl_->policy->parameter_bytes();
  impl_->policy_cache.clear();
  return absl::OkStatus();
}

absl::Status DeepCfrSolver::save_average_model(const std::filesystem::path& path) const {
  if (!impl_->policy) {
    return absl::FailedPreconditionError("average policy has not been trained");
  }
  return SaveNeuralPolicy(*impl_->policy, path);
}

const DeepCfrStats& DeepCfrSolver::stats() const noexcept { return impl_->stats; }

const SolverConfig& DeepCfrSolver::solver_config() const noexcept { return impl_->solver_config; }

const DealDistribution& DeepCfrSolver::deals() const noexcept { return impl_->deals; }

const HistoryTree& DeepCfrSolver::history() const noexcept { return impl_->history; }

const PublicPosition& DeepCfrSolver::initial_public() const noexcept { return impl_->initial_public; }

ModelFingerprint DeepCfrSolver::model() const noexcept { return impl_->model; }

const NeuralPolicy* DeepCfrSolver::average_policy() const noexcept {
  return impl_->policy ? &*impl_->policy : nullptr;
}

}  // namespace poker
