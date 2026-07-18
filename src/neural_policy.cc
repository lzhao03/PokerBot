#include "src/neural_policy.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <Accelerate/Accelerate.h>
#include <torch/torch.h>

#include "absl/status/status.h"

namespace poker {
namespace {

constexpr std::array<size_t, 3> kCompactPublicBuckets = {16, 16, 64};
constexpr size_t kPublicFeatureCount = 16 + 16 + 64;
constexpr size_t kPrivateBucketCount = 36;
constexpr size_t kPrivateFeatureCount = 4 * kPrivateBucketCount;
constexpr std::array<uint32_t, 4> kPrivateObservationPlaces = {
    1, 37, 37 * 37, 37 * 37 * 37};

struct CfrNetImpl : torch::nn::Module {
  explicit CfrNetImpl(int hidden_size)
      : hidden1(register_module(
            "hidden1", torch::nn::Linear(kNeuralFeatureCount, hidden_size))),
        hidden2(register_module(
            "hidden2", torch::nn::Linear(hidden_size, hidden_size))),
        hidden3(register_module(
            "hidden3", torch::nn::Linear(hidden_size, hidden_size))),
        output(register_module(
            "output", torch::nn::Linear(hidden_size, kMaxActionsPerNode))) {
    torch::nn::init::zeros_(output->weight);
    torch::nn::init::zeros_(output->bias);
  }

  torch::Tensor forward(torch::Tensor input) {
    input = torch::relu(hidden1->forward(std::move(input)));
    input = torch::relu(hidden2->forward(std::move(input)));
    input = torch::relu(hidden3->forward(std::move(input)));
    return output->forward(std::move(input));
  }

  torch::nn::Linear hidden1;
  torch::nn::Linear hidden2;
  torch::nn::Linear hidden3;
  torch::nn::Linear output;
};
TORCH_MODULE(CfrNet);

void Linear(const torch::nn::Linear& layer,
            std::span<const float> input,
            std::span<float> output) {
  assert(layer->weight.size(0) == static_cast<int64_t>(output.size()));
  assert(layer->weight.size(1) == static_cast<int64_t>(input.size()));
  std::copy_n(layer->bias.data_ptr<float>(), output.size(), output.begin());
  cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(output.size()),
              static_cast<int>(input.size()), 1.0f,
              layer->weight.data_ptr<float>(), static_cast<int>(input.size()),
              input.data(), 1, 1.0f, output.data(), 1);
}

void Relu(std::span<float> values) {
  for (float& value : values) value = std::max(0.0f, value);
}

std::mt19937 MakeRng(uint64_t seed) {
  const std::array<uint32_t, 2> words = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  std::seed_seq sequence(words.begin(), words.end());
  return std::mt19937(sequence);
}

struct NeuralMetadata {
  uint32_t feature_schema;
  int hidden_size;
  ModelFingerprint model;
};

NeuralMetadata ReadMetadata(const std::filesystem::path& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string());
  torch::Tensor schema;
  torch::Tensor hidden_size;
  torch::Tensor model;
  archive.read("feature_schema", schema);
  archive.read("hidden_size", hidden_size);
  archive.read("model", model);
  return {
      static_cast<uint32_t>(schema.item<int64_t>()),
      static_cast<int>(hidden_size.item<int64_t>()),
      ModelFingerprint{std::bit_cast<uint64_t>(model.item<int64_t>())}};
}

absl::Status TorchError(const std::exception& error) {
  return absl::InternalError(error.what());
}

}  // namespace

struct NeuralNetwork::Impl {
  explicit Impl(int hidden) : hidden_size(hidden), network(hidden) {}

  int hidden_size;
  CfrNet network;
};

NeuralNetwork::NeuralNetwork(int hidden_size)
    : impl_(std::make_unique<Impl>(hidden_size)) {}

NeuralNetwork::~NeuralNetwork() = default;
NeuralNetwork::NeuralNetwork(NeuralNetwork&&) noexcept = default;
NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&&) noexcept = default;

int NeuralNetwork::hidden_size() const noexcept {
  return impl_->hidden_size;
}

void SetNeuralSeed(uint64_t seed) {
  torch::manual_seed(seed);
}

NeuralFeatureVector EncodeNeuralFeatures(InfoSetKey key,
                                         const HistoryNode& node,
                                         const SolverConfig& config) {
  const DecisionState& decision = std::get<DecisionState>(node.state);
  const BettingData& betting = decision.data;
  NeuralFeatureVector features = {};
  size_t output = 0;
  const auto append_bits = [&](auto value) {
    for (size_t bit = 0; bit < sizeof(value) * 8; ++bit) {
      features[output++] = static_cast<float>((value >> bit) & 1);
    }
  };
  append_bits(std::to_underlying(key.history));

  const size_t public_begin = output;
  if (config.card_abstraction.public_mode == PublicCardMode::CompactTexture) {
    const uint64_t observation = std::to_underlying(key.public_observation);
    size_t bucket_offset = 0;
    for (size_t street = 0; street < kCompactPublicBuckets.size(); ++street) {
      const uint64_t bucket = (observation >> (street * 7)) & 0x7f;
      if (bucket != 0) {
        features[output + bucket_offset + bucket - 1] = 1.0f;
      }
      bucket_offset += kCompactPublicBuckets[street];
    }
  } else {
    append_bits(std::to_underlying(key.public_observation));
  }
  output = public_begin + kPublicFeatureCount;

  const size_t private_begin = output;
  if (config.card_abstraction.private_kind ==
      PrivateAbstractionKind::Handcrafted36) {
    const uint32_t observation =
        std::to_underlying(key.private_observation);
    if (config.card_abstraction.recall_mode == RecallMode::BucketHistory) {
      for (size_t street = 0; street < kPrivateObservationPlaces.size();
           ++street) {
        const uint32_t bucket =
            (observation / kPrivateObservationPlaces[street]) % 37;
        if (bucket != 0) {
          features[output + street * kPrivateBucketCount + bucket - 1] = 1.0f;
        }
      }
    } else if (observation != 0) {
      features[output + observation - 1] = 1.0f;
    }
  } else {
    append_bits(std::to_underlying(key.private_observation));
  }
  output = private_begin + kPrivateFeatureCount;

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

size_t NeuralParameterBytes(const NeuralNetwork& network) {
  size_t bytes = 0;
  for (const torch::Tensor& parameter : network.impl_->network->parameters()) {
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

NeuralActionVector PredictNeuralNetwork(
    const NeuralNetwork& network,
    const CompiledGame& game,
    InfoSetKey key,
    std::array<std::vector<float>, 2>& hidden) {
  const NeuralFeatureVector features = EncodeNeuralFeatures(
      key, game.history.nodes[Index(key.history)], game.config);
  Linear(network.impl_->network->hidden1, features, hidden[0]);
  Relu(hidden[0]);
  Linear(network.impl_->network->hidden2, hidden[0], hidden[1]);
  Relu(hidden[1]);
  Linear(network.impl_->network->hidden3, hidden[1], hidden[0]);
  Relu(hidden[0]);
  NeuralActionVector values = {};
  Linear(network.impl_->network->output, hidden[0], values);
  return values;
}

float FitNeuralNetwork(
    NeuralNetwork& network,
    const CompiledGame& game,
    std::span<const NeuralSample> samples,
    const NeuralTrainingConfig& config,
    NeuralTarget target_kind,
    const std::array<NeuralNetwork*, kPlayerCount>& current_policy_sources) {
  if (samples.empty()) return 0.0f;

  torch::manual_seed(config.seed);
  network.impl_->network = CfrNet(network.impl_->hidden_size);
  network.impl_->network->train();
  torch::optim::Adam optimizer(
      network.impl_->network->parameters(),
      torch::optim::AdamOptions(config.learning_rate));

  const size_t batch_size =
      std::min(static_cast<size_t>(config.batch_size), samples.size());
  std::vector<float> inputs(batch_size * kNeuralFeatureCount);
  std::vector<float> targets(batch_size * kMaxActionsPerNode);
  std::vector<float> masks(batch_size * kMaxActionsPerNode);
  std::vector<float> weights(batch_size);
  std::vector<float> player_b(batch_size);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  std::uniform_int_distribution<size_t> sample_index(0, samples.size() - 1);
  std::mt19937 batch_rng = MakeRng(config.seed);
  float final_loss = 0.0f;

  for (int step = 0; step < config.steps; ++step) {
    std::fill(masks.begin(), masks.end(), 0.0f);
    for (size_t row = 0; row < batch_size; ++row) {
      const NeuralSample& sample = samples[sample_index(batch_rng)];
      const HistoryNode& node = game.history.nodes[Index(sample.key.history)];
      const NeuralFeatureVector features =
          EncodeNeuralFeatures(sample.key, node, game.config);
      std::copy(features.begin(), features.end(),
                inputs.data() + row * kNeuralFeatureCount);
      std::copy(sample.target.begin(), sample.target.end(),
                targets.data() + row * kMaxActionsPerNode);
      std::fill_n(masks.data() + row * kMaxActionsPerNode,
                  node.child_count, 1.0f);
      weights[row] = sample.weight;
      player_b[row] = std::get<DecisionState>(node.state).actor == Player::B;
    }

    const torch::Tensor input = torch::from_blob(
        inputs.data(),
        {static_cast<int64_t>(batch_size), kNeuralFeatureCount}, options);
    torch::Tensor target = torch::from_blob(
        targets.data(),
        {static_cast<int64_t>(batch_size), kMaxActionsPerNode}, options);
    const torch::Tensor mask = torch::from_blob(
        masks.data(),
        {static_cast<int64_t>(batch_size), kMaxActionsPerNode}, options);
    const torch::Tensor weight = torch::from_blob(
        weights.data(), {static_cast<int64_t>(batch_size)}, options);

    if (target_kind == NeuralTarget::CurrentPolicy) {
      assert(current_policy_sources[0] != nullptr &&
             current_policy_sources[1] != nullptr);
      const torch::Tensor use_player_b = torch::from_blob(
          player_b.data(), {static_cast<int64_t>(batch_size), 1}, options);
      torch::NoGradGuard no_grad;
      const torch::Tensor advantages =
          current_policy_sources[0]->impl_->network->forward(input) *
              (1.0f - use_player_b) +
          current_policy_sources[1]->impl_->network->forward(input) *
              use_player_b;
      const torch::Tensor positive = torch::relu(advantages) * mask;
      const torch::Tensor sum = positive.sum(1, true);
      const torch::Tensor uniform = mask / mask.sum(1, true);
      target = torch::where(sum > 0.0f, positive / sum.clamp_min(1e-12f),
                            uniform);
    }

    optimizer.zero_grad();
    torch::Tensor prediction = network.impl_->network->forward(input);
    if (target_kind != NeuralTarget::Advantage) {
      prediction = torch::softmax(prediction + (mask - 1.0f) * 1e9f, 1);
    }
    const torch::Tensor squared_error =
        (prediction - target).square() * mask;
    const torch::Tensor loss =
        (squared_error.sum(1) * weight).sum() / weight.sum();
    loss.backward();
    optimizer.step();
    final_loss = loss.item<float>();
  }
  network.impl_->network->eval();
  return final_loss;
}

void SaveNeuralNetwork(const NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint model) {
  torch::serialize::OutputArchive archive;
  archive.write("feature_schema", torch::tensor(
      static_cast<int64_t>(kNeuralFeatureSchemaVersion)));
  archive.write("hidden_size", torch::tensor(
      static_cast<int64_t>(network.impl_->hidden_size)));
  archive.write("model", torch::tensor(
      std::bit_cast<int64_t>(std::to_underlying(model))));
  network.impl_->network->save(archive);
  archive.save_to(path.string());
}

void LoadNeuralNetwork(NeuralNetwork& network,
                       const std::filesystem::path& path,
                       ModelFingerprint expected_model) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string());
  torch::Tensor schema;
  torch::Tensor hidden_size;
  torch::Tensor model;
  archive.read("feature_schema", schema);
  archive.read("hidden_size", hidden_size);
  archive.read("model", model);
  if (schema.item<int64_t>() != kNeuralFeatureSchemaVersion) {
    throw std::runtime_error("neural feature schema does not match");
  }
  if (hidden_size.item<int64_t>() != network.impl_->hidden_size) {
    throw std::runtime_error("neural hidden size does not match");
  }
  if (model.item<int64_t>() !=
      std::bit_cast<int64_t>(std::to_underlying(expected_model))) {
    throw std::runtime_error("neural model fingerprint does not match");
  }
  network.impl_->network->load(archive);
  network.impl_->network->eval();
}

NeuralPolicy::NeuralPolicy(NeuralNetwork network, ModelFingerprint model)
    : network_(std::move(network)), model_(model) {}

NeuralPolicy::NeuralPolicy(NeuralPolicy&&) noexcept = default;
NeuralPolicy& NeuralPolicy::operator=(NeuralPolicy&&) noexcept = default;

bool NeuralPolicy::strategy(const CompiledGame& game,
                            InfoSetKey key,
                            std::span<float> probabilities) const {
  if (game.model != model_) {
    FillUniform(probabilities);
    return false;
  }
  thread_local std::array<std::vector<float>, 2> hidden;
  for (auto& values : hidden) {
    values.resize(static_cast<size_t>(network_.hidden_size()));
  }
  const NeuralActionVector logits =
      PredictNeuralNetwork(network_, game, key, hidden);
  Softmax(logits, probabilities);
  return true;
}

size_t NeuralPolicy::parameter_bytes() const {
  return NeuralParameterBytes(network_);
}

absl::StatusOr<NeuralPolicyFitResult> FitNeuralPolicy(
    const CompiledGame& game,
    const Policy& teacher,
    const NeuralTrainingConfig& config) {
  if (teacher.model != game.model) {
    return absl::FailedPreconditionError(
        "policy model does not match compiled game");
  }
  if (teacher.rows.empty()) {
    return absl::InvalidArgumentError("policy has no explicit rows");
  }
  if (config.steps <= 0 || config.batch_size <= 0 ||
      config.hidden_size <= 0 || !std::isfinite(config.learning_rate) ||
      config.learning_rate <= 0.0) {
    return absl::InvalidArgumentError("invalid neural training config");
  }

  std::vector<std::pair<InfoSetKey, uint32_t>> rows(
      teacher.rows.begin(), teacher.rows.end());
  std::ranges::sort(rows, {}, &std::pair<InfoSetKey, uint32_t>::first);
  std::vector<NeuralSample> samples;
  samples.reserve(rows.size());
  for (const auto& [key, offset] : rows) {
    if (Index(key.history) >= game.history.nodes.size()) {
      return absl::InvalidArgumentError("policy history is out of range");
    }
    const HistoryNode& node = game.history.nodes[Index(key.history)];
    if (!std::holds_alternative<DecisionState>(node.state) ||
        node.child_count == 0 ||
        offset + node.child_count > teacher.probabilities.size()) {
      return absl::InvalidArgumentError("invalid policy row");
    }
    NeuralSample sample{key};
    const std::span<float> target(sample.target.data(), node.child_count);
    if (!teacher.strategy(key, target)) {
      return absl::InvalidArgumentError("invalid policy row offset");
    }
    samples.push_back(sample);
  }

  try {
    SetNeuralSeed(config.seed);
    NeuralNetwork network(config.hidden_size);
    const float loss = FitNeuralNetwork(
        network, game, samples, config, NeuralTarget::AveragePolicy);
    return NeuralPolicyFitResult{
        NeuralPolicy(std::move(network), game.model), loss, samples.size()};
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::Status SaveNeuralPolicy(const NeuralPolicy& policy,
                              const std::filesystem::path& path) {
  try {
    SaveNeuralNetwork(policy.network_, path, policy.model_);
    return absl::OkStatus();
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

absl::StatusOr<NeuralPolicy> LoadNeuralPolicy(
    const std::filesystem::path& path,
    ModelFingerprint expected_model) {
  try {
    const NeuralMetadata metadata = ReadMetadata(path);
    if (metadata.feature_schema != kNeuralFeatureSchemaVersion) {
      return absl::FailedPreconditionError(
          "neural feature schema does not match");
    }
    if (metadata.model != expected_model) {
      return absl::FailedPreconditionError(
          "neural model fingerprint does not match");
    }
    NeuralNetwork network(metadata.hidden_size);
    LoadNeuralNetwork(network, path, expected_model);
    return NeuralPolicy(std::move(network), expected_model);
  } catch (const std::exception& error) {
    return TorchError(error);
  }
}

}  // namespace poker
