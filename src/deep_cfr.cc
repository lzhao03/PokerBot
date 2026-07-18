#include "src/deep_cfr.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "src/cfr_traversal.h"
#include "src/neural_policy.h"

namespace poker {
namespace {

class Reservoir {
 public:
  explicit Reservoir(size_t capacity) : capacity_(capacity) {
    samples_.reserve(capacity);
  }

  void add(NeuralSample sample, std::mt19937& rng) {
    const uint64_t index = seen_++;
    if (samples_.size() < capacity_) {
      samples_.push_back(std::move(sample));
      return;
    }
    const uint64_t replacement =
        std::uniform_int_distribution<uint64_t>(0, index)(rng);
    if (replacement < capacity_) {
      samples_[static_cast<size_t>(replacement)] = std::move(sample);
    }
  }

  size_t size() const noexcept { return samples_.size(); }
  std::span<const NeuralSample> samples() const noexcept {
    return samples_;
  }

 private:
  size_t capacity_;
  uint64_t seen_ = 0;
  std::vector<NeuralSample> samples_;
};

uint64_t NetworkSeed(uint64_t base,
                     uint64_t iteration,
                     uint64_t network) {
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
    return absl::InvalidArgumentError(
        "Deep CFR learning rate must be finite and positive");
  }
  return absl::OkStatus();
}

absl::Status TorchError(const std::exception& error) {
  return absl::InternalError(error.what());
}

}  // namespace

struct DeepCfrSolver::Impl {
  struct UpdateHandle {};

  struct EvaluationResult {
    double mean = 0.0;
    double standard_error = 0.0;
    uint64_t opponent_policy_lookups = 0;
    uint64_t missing_opponent_lookups = 0;
  };

  Impl(CompiledGame compiled_game, DeepCfrConfig deep_config)
      : game(std::move(compiled_game)),
        config(deep_config),
        advantage_memory{
            Reservoir(config.advantage_memory_capacity),
            Reservoir(config.advantage_memory_capacity)},
        strategy_memory(config.strategy_memory_capacity),
        advantage_network{
            NeuralNetwork(config.hidden_size),
            NeuralNetwork(config.hidden_size)},
        policy_network(config.hidden_size),
        inference_hidden{
            std::vector<float>(static_cast<size_t>(config.hidden_size)),
            std::vector<float>(static_cast<size_t>(config.hidden_size))},
        game_rng(MakeRng(config.seed)),
        reservoir_rng(MakeRng(config.seed + 1)) {
    for (auto& cache : advantage_cache) {
      cache.reserve(config.inference_cache_capacity);
    }
    stats.policy_parameter_bytes = NeuralParameterBytes(policy_network);
  }

  std::optional<UpdateHandle> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess access,
      std::span<float> probabilities) {
    const size_t player = Index(decision.state.actor);
    if (!advantage_trained[player]) {
      FillUniform(probabilities);
    } else {
      const NeuralActionVector values = cached_prediction(
          advantage_network[player], advantage_cache[player], decision.key,
          inference_hidden, config.inference_cache_capacity,
          stats.network_evaluations, stats.cache_hits);
      RegretMatch(values, probabilities);
    }
    return access == internal::StrategyAccess::ReadOnly
               ? std::nullopt
               : std::optional<UpdateHandle>{std::in_place};
  }

  void average_strategy(const internal::DecisionView& decision,
                        std::span<float> probabilities) {
    policy_strategy(decision.key, probabilities);
  }

  bool policy_strategy(InfoSetKey key,
                       std::span<float> probabilities) {
    return policy_strategy(
        key, probabilities, policy_cache, inference_hidden,
        stats.network_evaluations, stats.cache_hits);
  }

  bool policy_strategy(
      InfoSetKey key,
      std::span<float> probabilities,
      absl::flat_hash_map<InfoSetKey, NeuralActionVector>& cache,
      std::array<std::vector<float>, 2>& hidden,
      uint64_t& network_evaluations,
      uint64_t& cache_hits) {
    if (!policy_trained) {
      FillUniform(probabilities);
      return false;
    }
    const NeuralActionVector logits =
        cached_prediction(policy_network, cache, key, hidden,
                          config.policy_cache_capacity,
                          network_evaluations, cache_hits);
    Softmax(logits, probabilities);
    return true;
  }

  struct EvaluationBackend {
    using UpdateHandle = Impl::UpdateHandle;

    std::optional<UpdateHandle> current_strategy(
        const internal::DecisionView& decision,
        internal::StrategyAccess access,
        std::span<float> probabilities) {
      if (fixed_strategy(decision, probabilities)) return std::nullopt;
      return model.current_strategy(decision, access, probabilities);
    }

    void average_strategy(const internal::DecisionView& decision,
                          std::span<float> probabilities) {
      if (!fixed_strategy(decision, probabilities)) {
        model.average_strategy(decision, probabilities);
      }
    }

    void record_regrets(const internal::DecisionView&,
                        UpdateHandle,
                        std::span<const float>) {}
    void record_strategy(const internal::DecisionView&,
                         UpdateHandle,
                         std::span<const float>,
                         double) {}

    bool fixed_strategy(const internal::DecisionView& decision,
                        std::span<float> probabilities) {
      if (uniform_player == decision.state.actor) {
        FillUniform(probabilities);
        return true;
      }
      if (opponent == nullptr || opponent_player != decision.state.actor) {
        return false;
      }
      ++opponent_policy_lookups;
      if (!opponent->strategy(decision.key, probabilities)) {
        ++missing_opponent_lookups;
      }
      return true;
    }

    Impl& model;
    std::optional<Player> uniform_player;
    const Policy* opponent = nullptr;
    Player opponent_player = Player::A;
    uint64_t opponent_policy_lookups = 0;
    uint64_t missing_opponent_lookups = 0;
  };

  void record_regrets(const internal::DecisionView& decision,
                      UpdateHandle,
                      std::span<const float> regrets) {
    NeuralSample sample{decision.key};
    const BettingData& betting = decision.state.data;
    const float scale = 1.0f / static_cast<float>(
        Pot(betting) + betting.stack[0] + betting.stack[1]);
    std::ranges::transform(regrets, sample.target.begin(),
                           [scale](float regret) { return regret * scale; });
    advantage_memory[Index(decision.state.actor)].add(
        std::move(sample), reservoir_rng);
  }

  void record_strategy(const internal::DecisionView& decision,
                       UpdateHandle,
                       std::span<const float> probabilities,
                       double weight) {
    NeuralSample sample{decision.key};
    std::copy(probabilities.begin(), probabilities.end(),
              sample.target.begin());
    sample.weight = static_cast<float>(
        weight * static_cast<double>(decision.iteration + 1));
    strategy_memory.add(std::move(sample), reservoir_rng);
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
        PredictNeuralNetwork(network, game, key, hidden);
    if (cache.size() < capacity) {
      cache.emplace(key, values);
    }
    return values;
  }

  float train_network(NeuralNetwork& network,
                      const Reservoir& memory,
                      uint64_t seed,
                      NeuralTarget target_kind,
                      int training_steps) {
    return FitNeuralNetwork(
        network, game, memory.samples(),
        {.seed = seed,
         .steps = training_steps,
         .batch_size = config.batch_size,
         .hidden_size = config.hidden_size,
         .learning_rate = config.learning_rate},
        target_kind,
        {&advantage_network[0], &advantage_network[1]});
  }

  double traverse(const Deal& deal,
                  Player update_player,
                  uint64_t iteration) {
    internal::TraversalContext context{
        .deal = deal,
        .mode = internal::TraversalMode::Train,
        .update_player = update_player,
        .iteration = iteration,
        .external_sampling = true,
        .rng = game_rng,
        .stats = stats.traversal,
    };
    return internal::Traverse(game, context, *this);
  }

  void run(uint64_t iterations) {
    // Brown et al. (2019), Algorithms 1-2: collect external-sampling
    // traversals, then retrain each traverser's advantage network from scratch.
    for (uint64_t outer = 0; outer < iterations; ++outer) {
      const uint64_t iteration = stats.iterations;
      for (Player player : {Player::A, Player::B}) {
        for (int traversal = 0; traversal < config.traversals_per_player;
             ++traversal) {
          traverse(game.deals.sample(game_rng), player, iteration);
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

    stats.strategy_loss = train_network(
        policy_network, strategy_memory,
        NetworkSeed(config.seed, stats.iterations, kPlayerCount),
        config.distill_current_policy ? NeuralTarget::CurrentPolicy
                                      : NeuralTarget::AveragePolicy,
        config.policy_training_steps);
    policy_trained = strategy_memory.size() > 0;
    policy_cache.clear();
    for (size_t player = 0; player < kPlayerCount; ++player) {
      stats.advantage_samples[player] = advantage_memory[player].size();
    }
    stats.strategy_samples = strategy_memory.size();
  }

  EvaluationResult evaluate(
      int samples,
      internal::TraversalMode mode,
      std::optional<Player> uniform_player = std::nullopt,
      const Policy* opponent = nullptr,
      Player opponent_player = Player::A) {
    SolverStats evaluation_stats;
    EvaluationBackend backend{
        *this, uniform_player, opponent, opponent_player};
    std::mt19937 rng = MakeRng(config.seed + 3);
    double mean = 0.0;
    double squared_error = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
      const Deal deal = game.deals.sample(rng);
      internal::TraversalContext context{
          .deal = deal,
          .mode = mode,
          .update_player = Player::A,
          .iteration = stats.iterations,
          .external_sampling = true,
          .rng = rng,
          .stats = evaluation_stats,
      };
      const double value = internal::Traverse(game, context, backend);
      const double delta = value - mean;
      mean += delta / (sample + 1);
      squared_error += delta * (value - mean);
    }
    const double standard_error = samples > 1
        ? std::sqrt(squared_error / (samples - 1) / samples)
        : 0.0;
    return {mean, standard_error, backend.opponent_policy_lookups,
            backend.missing_opponent_lookups};
  }

  CompiledGame game;
  DeepCfrConfig config;
  std::array<Reservoir, kPlayerCount> advantage_memory;
  Reservoir strategy_memory;
  std::array<NeuralNetwork, kPlayerCount> advantage_network;
  NeuralNetwork policy_network;
  std::array<std::vector<float>, 2> inference_hidden;
  std::array<bool, kPlayerCount> advantage_trained = {};
  bool policy_trained = false;
  std::array<absl::flat_hash_map<InfoSetKey, NeuralActionVector>, kPlayerCount>
      advantage_cache;
  absl::flat_hash_map<InfoSetKey, NeuralActionVector> policy_cache;
  std::mt19937 game_rng;
  std::mt19937 reservoir_rng;
  DeepCfrStats stats;
};

DeepCfrSolver::DeepCfrSolver(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DeepCfrSolver::~DeepCfrSolver() = default;
DeepCfrSolver::DeepCfrSolver(DeepCfrSolver&&) noexcept = default;
DeepCfrSolver& DeepCfrSolver::operator=(DeepCfrSolver&&) noexcept = default;

absl::StatusOr<DeepCfrSolver> DeepCfrSolver::Create(
    SolveSpec spec,
    DeepCfrConfig config) {
  const absl::Status config_status = ValidateConfig(config);
  if (!config_status.ok()) return config_status;
  auto game = CompileGame(std::move(spec));
  if (!game.ok()) return game.status();
  try {
    SetNeuralSeed(config.seed);
    return DeepCfrSolver(
        std::make_unique<Impl>(std::move(*game), config));
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
  if (samples <= 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  try {
    return impl_->evaluate(samples, internal::TraversalMode::EvaluateCurrent)
        .mean;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_average(int samples) {
  if (samples <= 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  try {
    return impl_->evaluate(samples, internal::TraversalMode::EvaluateAverage)
        .mean;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_average_against_uniform(
    Player policy_player,
    int samples) {
  if (samples <= 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  try {
    const double player_a_value = impl_->evaluate(
        samples, internal::TraversalMode::EvaluateAverage,
        Opponent(policy_player)).mean;
    return policy_player == Player::A ? player_a_value : -player_a_value;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<DeepCfrMatchResult> DeepCfrSolver::evaluate_against_policy(
    Player policy_player,
    const Policy& opponent,
    DeepCfrStrategy strategy,
    int samples) {
  if (samples <= 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  if (opponent.model != impl_->game.model) {
    return absl::FailedPreconditionError("policy model does not match game");
  }
  try {
    const internal::TraversalMode mode =
        strategy == DeepCfrStrategy::Average
            ? internal::TraversalMode::EvaluateAverage
            : internal::TraversalMode::EvaluateCurrent;
    const Impl::EvaluationResult result = impl_->evaluate(
        samples, mode, std::nullopt, &opponent, Opponent(policy_player));
    const double sign = policy_player == Player::A ? 1.0 : -1.0;
    return DeepCfrMatchResult{
        sign * result.mean, result.standard_error,
        result.opponent_policy_lookups,
        result.missing_opponent_lookups};
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<ExploitabilityEstimate>
DeepCfrSolver::estimate_exploitability(
    const BestResponseConfig& config) {
  if (!impl_->policy_trained) {
    return absl::FailedPreconditionError(
        "average policy has not been trained");
  }
  try {
    impl_->policy_cache.reserve(impl_->config.policy_cache_capacity);
    const StrategyLookup player_a_lookup = [this](
        InfoSetKey key, std::span<float> probabilities) {
      return impl_->policy_strategy(key, probabilities);
    };
    absl::flat_hash_map<InfoSetKey, NeuralActionVector> player_b_cache;
    player_b_cache.reserve(impl_->config.policy_cache_capacity);
    std::array<std::vector<float>, 2> player_b_hidden = {
        std::vector<float>(static_cast<size_t>(impl_->config.hidden_size)),
        std::vector<float>(static_cast<size_t>(impl_->config.hidden_size))};
    uint64_t network_evaluations = 0;
    uint64_t cache_hits = 0;
    const StrategyLookup player_b_lookup = [this, &player_b_cache,
                                             &player_b_hidden,
                                             &network_evaluations,
                                             &cache_hits](
        InfoSetKey key, std::span<float> probabilities) {
      return impl_->policy_strategy(
          key, probabilities, player_b_cache, player_b_hidden,
          network_evaluations, cache_hits);
    };
    const std::array<StrategyLookup, kPlayerCount> lookups = {
        player_a_lookup, player_b_lookup};
    auto result = EstimateExploitabilityParallel(
        impl_->game, lookups, config);
    impl_->stats.network_evaluations += network_evaluations;
    impl_->stats.cache_hits += cache_hits;
    return result;
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::Status DeepCfrSolver::load_average_model(
    const std::filesystem::path& path) {
  try {
    LoadNeuralNetwork(impl_->policy_network, path, impl_->game.model);
    impl_->policy_trained = true;
    impl_->policy_cache.clear();
    return absl::OkStatus();
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::Status DeepCfrSolver::save_average_model(
    const std::filesystem::path& path) const {
  if (!impl_->policy_trained) {
    return absl::FailedPreconditionError("average policy has not been trained");
  }
  try {
    SaveNeuralNetwork(impl_->policy_network, path, impl_->game.model);
    return absl::OkStatus();
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

const DeepCfrStats& DeepCfrSolver::stats() const noexcept {
  return impl_->stats;
}

const CompiledGame& DeepCfrSolver::game() const noexcept {
  return impl_->game;
}

}  // namespace poker
