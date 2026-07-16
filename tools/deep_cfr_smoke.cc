// Experimental: train one LibTorch advantage network on real CFR regret
// samples. This intentionally omits reservoirs and average-policy training.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "absl/types/span.h"
#include "src/cfr_traversal.h"
#include "src/solver.h"

namespace poker {
namespace {

constexpr size_t kFeatureCount = 32 + 64 + 32 + 2;

std::array<float, kFeatureCount> Features(
    const internal::DecisionView& decision) {
  std::array<float, kFeatureCount> features{};
  size_t output = 0;
  const auto append_bits = [&](auto value) {
    for (size_t bit = 0; bit < sizeof(value) * 8; ++bit) {
      features[output++] = static_cast<float>((value >> bit) & 1);
    }
  };
  append_bits(std::to_underlying(decision.key.history));
  append_bits(std::to_underlying(decision.key.public_observation));
  append_bits(std::to_underlying(decision.key.private_observation));
  features[output++] = decision.state.actor == Player::B ? 1.0f : 0.0f;
  features[output] =
      static_cast<float>(decision.action_count) / kMaxActionsPerNode;
  return features;
}

struct AdvantageNetImpl : torch::nn::Module {
  AdvantageNetImpl()
      : hidden(register_module(
            "hidden", torch::nn::Linear(kFeatureCount, 32))),
        output(register_module(
            "output", torch::nn::Linear(32, kMaxActionsPerNode))) {
    torch::nn::init::zeros_(output->weight);
    torch::nn::init::zeros_(output->bias);
  }

  torch::Tensor forward(torch::Tensor input) {
    return output->forward(torch::relu(hidden->forward(std::move(input))));
  }

  torch::nn::Linear hidden;
  torch::nn::Linear output;
};
TORCH_MODULE(AdvantageNet);

struct RegretSample {
  std::array<float, kFeatureCount> features;
  std::array<float, kMaxActionsPerNode> regrets{};
  std::array<float, kMaxActionsPerNode> mask{};
};

struct DeepCfrBackend {
  using DecisionToken = std::monostate;

  AdvantageNet network;
  std::vector<RegretSample> samples;

  std::optional<DecisionToken> strategy(const internal::DecisionView&,
                                        internal::DecisionRole role,
                                        absl::Span<float> probabilities) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0f / static_cast<float>(probabilities.size()));
    if (role != internal::DecisionRole::UpdatePlayer &&
        role != internal::DecisionRole::SampledOpponent) {
      return std::nullopt;
    }
    return DecisionToken{};
  }

  void observe_regrets(const internal::DecisionView& decision,
                       DecisionToken,
                       absl::Span<const float> regrets) {
    RegretSample sample{Features(decision)};
    std::copy(regrets.begin(), regrets.end(), sample.regrets.begin());
    std::fill_n(sample.mask.begin(), regrets.size(), 1.0f);
    samples.push_back(sample);
  }

  void observe_strategy(const internal::DecisionView&,
                        DecisionToken,
                        absl::Span<const float>,
                        double) {}

  std::pair<float, float> train(int steps) {
    std::vector<float> inputs;
    std::vector<float> targets;
    std::vector<float> masks;
    inputs.reserve(samples.size() * kFeatureCount);
    targets.reserve(samples.size() * kMaxActionsPerNode);
    masks.reserve(samples.size() * kMaxActionsPerNode);
    for (const RegretSample& sample : samples) {
      inputs.insert(inputs.end(), sample.features.begin(),
                    sample.features.end());
      targets.insert(targets.end(), sample.regrets.begin(),
                     sample.regrets.end());
      masks.insert(masks.end(), sample.mask.begin(), sample.mask.end());
    }

    const int64_t count = static_cast<int64_t>(samples.size());
    const torch::Tensor input =
        torch::from_blob(inputs.data(), {count, kFeatureCount}).clone();
    const torch::Tensor target = torch::from_blob(
        targets.data(), {count, kMaxActionsPerNode}).clone();
    const torch::Tensor mask =
        torch::from_blob(masks.data(), {count, kMaxActionsPerNode}).clone();
    const auto loss = [&] {
      return ((network->forward(input) - target).square() * mask).sum() /
             mask.sum();
    };

    float before;
    {
      torch::NoGradGuard no_grad;
      before = loss().item<float>();
    }
    torch::optim::Adam optimizer(
        network->parameters(), torch::optim::AdamOptions(1e-2));
    for (int step = 0; step < steps; ++step) {
      optimizer.zero_grad();
      const torch::Tensor value = loss();
      value.backward();
      optimizer.step();
    }
    torch::NoGradGuard no_grad;
    return {before, loss().item<float>()};
  }
};

}  // namespace
}  // namespace poker

int main() {
  torch::manual_seed(1);
  torch::set_num_threads(1);

  poker::SolverConfig config;
  config.bet_abstraction = poker::SmallBettingConfig();
  config.card_abstraction.public_mode = poker::PublicCardMode::CompactTexture;
  config.card_abstraction.private_kind =
      poker::PrivateAbstractionKind::Handcrafted36;
  config.card_abstraction.recall_mode = poker::RecallMode::CurrentBucketOnly;
  config.external_sampling = true;
  config.accumulate_average_strategy = false;
  config.max_info_sets = 1;
  const poker::ComboRange range = poker::UniformComboRange();
  const poker::ExactPublicState root = poker::MakeInitialState(
      config.betting_rules, {8, 8}, {1, 2});
  auto game = poker::CFRSolver::Create({config, root, {range, range}});
  if (!game.ok()) {
    std::cerr << game.status() << '\n';
    return 1;
  }

  const poker::SolveSpec& spec = game->solve_spec();
  const poker::HistoryTree& history = game->history_tree();
  const poker::Position position = poker::internal::RootPosition(spec);
  std::mt19937 rng(7);
  poker::SolverStats stats;
  poker::DeepCfrBackend backend;
  constexpr uint64_t kTraversals = 256;
  for (uint64_t iteration = 0; iteration < kTraversals; ++iteration) {
    const poker::Deal deal = game->deal_distribution().sample(rng);
    const poker::internal::TraversalFrame frame =
        poker::internal::InitialTraversalFrame(spec, deal, position);
    poker::internal::TraversalContext context{
        .deal = deal,
        .mode = poker::internal::TraversalMode::Train,
        .update_player = (iteration & 1) == 0 ? poker::Player::A
                                             : poker::Player::B,
        .iteration = iteration,
        .rng = rng,
        .stats = stats,
    };
    poker::internal::Traverse(spec, history, position.history,
                              position.public_state, frame, context, backend);
  }

  if (backend.samples.empty()) {
    std::cerr << "no regret samples collected\n";
    return 1;
  }
  const auto [before, after] = backend.train(100);
  std::cout << "regret_samples=" << backend.samples.size() << '\n'
            << "decision_visits=" << stats.decision_visits << '\n'
            << "loss_before=" << before << '\n'
            << "loss_after=" << after << '\n';
  return after < before ? 0 : 1;
}
