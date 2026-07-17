#include "src/deep_cfr.h"

#include <algorithm>
#include <array>
#include <cassert>
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

#include <torch/torch.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "src/cfr_traversal.h"

namespace poker {
namespace {

constexpr size_t kIdentityFeatureCount = 32 + 64 + 32;
constexpr size_t kFeatureCount = kIdentityFeatureCount + 15;

using FeatureVector = std::array<float, kFeatureCount>;
using ActionVector = std::array<float, kMaxActionsPerNode>;

enum class NetworkTarget : uint8_t {
  Advantage,
  Policy,
};

struct NetworkSample {
  InfoSetKey key;
  ActionVector target = {};
  float weight = 1.0f;
};

FeatureVector Features(InfoSetKey key, const HistoryNode& node) {
  const DecisionState& decision = std::get<DecisionState>(node.state);
  const BettingData& betting = decision.data;
  FeatureVector features = {};
  size_t output = 0;
  const auto append_bits = [&](auto value) {
    for (size_t bit = 0; bit < sizeof(value) * 8; ++bit) {
      features[output++] = static_cast<float>((value >> bit) & 1);
    }
  };
  append_bits(std::to_underlying(key.history));
  append_bits(std::to_underlying(key.public_observation));
  append_bits(std::to_underlying(key.private_observation));

  features[output++] = decision.actor == Player::B ? 1.0f : 0.0f;
  features[output++] =
      static_cast<float>(node.child_count) / kMaxActionsPerNode;
  for (StreetKind street : {StreetKind::Preflop, StreetKind::Flop,
                            StreetKind::Turn, StreetKind::River}) {
    features[output++] = betting.street == street ? 1.0f : 0.0f;
  }

  const Chips total_chips =
      Pot(betting) + betting.stack[0] + betting.stack[1];
  const float chip_scale = 1.0f / std::max(Chips{1}, total_chips);
  for (Chips value : betting.stack) {
    features[output++] = value * chip_scale;
  }
  for (Chips value : betting.total_committed) {
    features[output++] = value * chip_scale;
  }
  for (Chips value : betting.street_committed) {
    features[output++] = value * chip_scale;
  }
  features[output++] = betting.last_full_raise * chip_scale;
  features[output++] = betting.actions_remaining / 2.0f;
  features[output++] = Pot(betting) * chip_scale;
  assert(output == features.size());
  return features;
}

class Reservoir {
 public:
  explicit Reservoir(size_t capacity) : capacity_(capacity) {
    samples_.reserve(capacity);
  }

  void add(NetworkSample sample, std::mt19937& rng) {
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
  const NetworkSample& operator[](size_t index) const {
    return samples_[index];
  }

 private:
  size_t capacity_;
  uint64_t seen_ = 0;
  std::vector<NetworkSample> samples_;
};

struct CfrNetImpl : torch::nn::Module {
  explicit CfrNetImpl(int hidden_size)
      : hidden1(register_module(
            "hidden1", torch::nn::Linear(kFeatureCount, hidden_size))),
        hidden2(register_module(
            "hidden2", torch::nn::Linear(hidden_size, hidden_size))),
        output(register_module(
            "output", torch::nn::Linear(hidden_size, kMaxActionsPerNode))) {
    torch::nn::init::zeros_(output->weight);
    torch::nn::init::zeros_(output->bias);
  }

  torch::Tensor forward(torch::Tensor input) {
    input = torch::relu(hidden1->forward(std::move(input)));
    input = torch::relu(hidden2->forward(std::move(input)));
    return output->forward(std::move(input));
  }

  torch::nn::Linear hidden1;
  torch::nn::Linear hidden2;
  torch::nn::Linear output;
};
TORCH_MODULE(CfrNet);

size_t ParameterBytes(const CfrNet& network) {
  size_t bytes = 0;
  for (const torch::Tensor& parameter : network->parameters()) {
    bytes += static_cast<size_t>(parameter.numel()) *
             static_cast<size_t>(parameter.element_size());
  }
  return bytes;
}

void FillUniform(std::span<float> probabilities) {
  std::fill(probabilities.begin(), probabilities.end(),
            1.0f / static_cast<float>(probabilities.size()));
}

void RegretMatch(std::span<const float> advantages,
                 std::span<float> probabilities) {
  float sum = 0.0f;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    probabilities[action] = std::max(0.0f, advantages[action]);
    sum += probabilities[action];
  }
  if (!std::isfinite(sum) || sum <= 0.0f) {
    FillUniform(probabilities);
    return;
  }
  for (float& probability : probabilities) probability /= sum;
}

void Softmax(std::span<const float> logits,
             std::span<float> probabilities) {
  const float maximum =
      std::ranges::max(logits.first(probabilities.size()));
  float sum = 0.0f;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    probabilities[action] = std::exp(logits[action] - maximum);
    sum += probabilities[action];
  }
  if (!std::isfinite(sum) || sum <= 0.0f) {
    FillUniform(probabilities);
    return;
  }
  for (float& probability : probabilities) probability /= sum;
}

ActionVector Predict(CfrNet& network,
                     const CompiledGame& game,
                     InfoSetKey key) {
  const FeatureVector features =
      Features(key, game.history.nodes[Index(key.history)]);
  torch::NoGradGuard no_grad;
  const torch::Tensor tensor = torch::from_blob(
      const_cast<float*>(features.data()), {1, kFeatureCount}, torch::kFloat32);
  const torch::Tensor output = network->forward(tensor).contiguous();
  ActionVector values = {};
  std::copy_n(output.data_ptr<float>(), values.size(), values.begin());
  return values;
}

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
      config.batch_size <= 0 || config.hidden_size <= 0) {
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

  Impl(CompiledGame compiled_game, DeepCfrConfig deep_config)
      : game(std::move(compiled_game)),
        config(deep_config),
        advantage_memory{
            Reservoir(config.advantage_memory_capacity),
            Reservoir(config.advantage_memory_capacity)},
        strategy_memory(config.strategy_memory_capacity),
        advantage_network{
            CfrNet(config.hidden_size),
            CfrNet(config.hidden_size)},
        policy_network(config.hidden_size),
        game_rng(MakeRng(config.seed)),
        reservoir_rng(MakeRng(config.seed + 1)),
        evaluation_rng(MakeRng(config.seed + 3)) {
    for (auto& cache : advantage_cache) {
      cache.reserve(config.inference_cache_capacity);
    }
    policy_cache.reserve(config.inference_cache_capacity);
    stats.policy_parameter_bytes = ParameterBytes(policy_network);
  }

  std::optional<UpdateHandle> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess access,
      std::span<float> probabilities) {
    const size_t player = Index(decision.state.actor);
    if (!advantage_trained[player]) {
      FillUniform(probabilities);
    } else {
      const ActionVector values = cached_prediction(
          advantage_network[player], advantage_cache[player], decision);
      RegretMatch(values, probabilities);
    }
    return access == internal::StrategyAccess::ReadOnly
               ? std::nullopt
               : std::optional<UpdateHandle>{std::in_place};
  }

  void average_strategy(const internal::DecisionView& decision,
                        std::span<float> probabilities) {
    if (!policy_trained) {
      FillUniform(probabilities);
      return;
    }
    const ActionVector logits =
        cached_prediction(policy_network, policy_cache, decision);
    Softmax(logits, probabilities);
  }

  struct EvaluationBackend {
    using UpdateHandle = Impl::UpdateHandle;

    std::optional<UpdateHandle> current_strategy(
        const internal::DecisionView& decision,
        internal::StrategyAccess access,
        std::span<float> probabilities) {
      if (uniform_player == decision.state.actor) {
        FillUniform(probabilities);
        return std::nullopt;
      }
      return model.current_strategy(decision, access, probabilities);
    }

    void average_strategy(const internal::DecisionView& decision,
                          std::span<float> probabilities) {
      if (uniform_player == decision.state.actor) {
        FillUniform(probabilities);
      } else {
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

    Impl& model;
    std::optional<Player> uniform_player;
  };

  void record_regrets(const internal::DecisionView& decision,
                      UpdateHandle,
                      std::span<const float> regrets) {
    NetworkSample sample{decision.key};
    std::copy(regrets.begin(), regrets.end(), sample.target.begin());
    sample.weight = static_cast<float>(decision.iteration + 1);
    advantage_memory[Index(decision.state.actor)].add(
        std::move(sample), reservoir_rng);
  }

  void record_strategy(const internal::DecisionView& decision,
                       UpdateHandle,
                       std::span<const float> probabilities,
                       double weight) {
    NetworkSample sample{decision.key};
    std::copy(probabilities.begin(), probabilities.end(),
              sample.target.begin());
    sample.weight = static_cast<float>(
        weight * static_cast<double>(decision.iteration + 1));
    strategy_memory.add(std::move(sample), reservoir_rng);
  }

  ActionVector cached_prediction(
      CfrNet& network,
      absl::flat_hash_map<InfoSetKey, ActionVector>& cache,
      const internal::DecisionView& decision) {
    const auto found = cache.find(decision.key);
    if (found != cache.end()) {
      ++stats.cache_hits;
      return found->second;
    }
    ++stats.network_evaluations;
    const ActionVector values = Predict(network, game, decision.key);
    if (cache.size() < config.inference_cache_capacity) {
      cache.emplace(decision.key, values);
    }
    return values;
  }

  float train_network(CfrNet& network,
                      const Reservoir& memory,
                      uint64_t seed,
                      NetworkTarget target_kind) {
    if (memory.size() == 0) return 0.0f;

    torch::manual_seed(seed);
    network = CfrNet(config.hidden_size);
    network->train();
    torch::optim::Adam optimizer(
        network->parameters(),
        torch::optim::AdamOptions(config.learning_rate));

    const size_t batch_size = std::min(
        static_cast<size_t>(config.batch_size), memory.size());
    std::vector<float> inputs(batch_size * kFeatureCount);
    std::vector<float> targets(batch_size * kMaxActionsPerNode);
    std::vector<float> masks(batch_size * kMaxActionsPerNode);
    std::vector<float> weights(batch_size);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    std::uniform_int_distribution<size_t> sample_index(0, memory.size() - 1);
    std::mt19937 batch_rng = MakeRng(seed);
    float final_loss = 0.0f;

    for (int step = 0; step < config.training_steps; ++step) {
      std::fill(masks.begin(), masks.end(), 0.0f);
      for (size_t row = 0; row < batch_size; ++row) {
        const NetworkSample& sample = memory[sample_index(batch_rng)];
        const HistoryNode& node =
            game.history.nodes[Index(sample.key.history)];
        const FeatureVector features = Features(sample.key, node);
        std::copy(features.begin(), features.end(),
                  inputs.data() + row * kFeatureCount);
        std::copy(sample.target.begin(), sample.target.end(),
                  targets.data() + row * kMaxActionsPerNode);
        std::fill_n(masks.data() + row * kMaxActionsPerNode,
                    node.child_count, 1.0f);
        weights[row] = sample.weight;
      }

      const torch::Tensor input = torch::from_blob(
          inputs.data(), {static_cast<int64_t>(batch_size), kFeatureCount},
          options);
      const torch::Tensor target = torch::from_blob(
          targets.data(),
          {static_cast<int64_t>(batch_size), kMaxActionsPerNode}, options);
      const torch::Tensor mask = torch::from_blob(
          masks.data(),
          {static_cast<int64_t>(batch_size), kMaxActionsPerNode}, options);
      const torch::Tensor weight = torch::from_blob(
          weights.data(), {static_cast<int64_t>(batch_size)}, options);

      optimizer.zero_grad();
      torch::Tensor prediction = network->forward(input);
      if (target_kind == NetworkTarget::Policy) {
        prediction = torch::softmax(
            prediction + (mask - 1.0f) * 1e9f, 1);
      }
      const torch::Tensor squared_error =
          (prediction - target).square() * mask;
      const torch::Tensor loss =
          (squared_error.sum(1) * weight).sum() / weight.sum();
      loss.backward();
      optimizer.step();
      final_loss = loss.item<float>();
    }
    network->eval();
    return final_loss;
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
            NetworkTarget::Advantage);
        advantage_trained[index] = true;
        advantage_cache[index].clear();
      }
      ++stats.iterations;
    }

    stats.strategy_loss = train_network(
        policy_network, strategy_memory,
        NetworkSeed(config.seed, stats.iterations, kPlayerCount),
        NetworkTarget::Policy);
    policy_trained = strategy_memory.size() > 0;
    policy_cache.clear();
    for (size_t player = 0; player < kPlayerCount; ++player) {
      stats.advantage_samples[player] = advantage_memory[player].size();
    }
    stats.strategy_samples = strategy_memory.size();
  }

  double evaluate(int samples,
                  internal::TraversalMode mode,
                  std::optional<Player> uniform_player = std::nullopt) {
    SolverStats evaluation_stats;
    EvaluationBackend backend{*this, uniform_player};
    double value = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
      const Deal deal = game.deals.sample(evaluation_rng);
      internal::TraversalContext context{
          .deal = deal,
          .mode = mode,
          .update_player = Player::A,
          .iteration = stats.iterations,
          .external_sampling = true,
          .rng = evaluation_rng,
          .stats = evaluation_stats,
      };
      value += internal::Traverse(game, context, backend);
    }
    return value / samples;
  }

  CompiledGame game;
  DeepCfrConfig config;
  std::array<Reservoir, kPlayerCount> advantage_memory;
  Reservoir strategy_memory;
  std::array<CfrNet, kPlayerCount> advantage_network;
  CfrNet policy_network;
  std::array<bool, kPlayerCount> advantage_trained = {};
  bool policy_trained = false;
  std::array<absl::flat_hash_map<InfoSetKey, ActionVector>, kPlayerCount>
      advantage_cache;
  absl::flat_hash_map<InfoSetKey, ActionVector> policy_cache;
  std::mt19937 game_rng;
  std::mt19937 reservoir_rng;
  std::mt19937 evaluation_rng;
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
    torch::manual_seed(config.seed);
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
    return impl_->evaluate(samples, internal::TraversalMode::EvaluateCurrent);
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<double> DeepCfrSolver::evaluate_average(int samples) {
  if (samples <= 0) {
    return absl::InvalidArgumentError("evaluation samples must be positive");
  }
  try {
    return impl_->evaluate(samples, internal::TraversalMode::EvaluateAverage);
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
        Opponent(policy_player));
    return policy_player == Player::A ? player_a_value : -player_a_value;
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
    torch::save(impl_->policy_network, path.string());
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
