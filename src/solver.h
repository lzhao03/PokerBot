#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/bet_abstraction.h"
#include "src/card_abstraction.h"
#include "src/history.h"
#include "src/poker.h"

namespace poker {

struct SolverConfig {
  BetAbstractionConfig bet_abstraction;
  CardAbstractionConfig card_abstraction;
  BettingRules betting_rules = {2};
  int chance_samples = 1;
  bool accumulate_average_strategy = true;
  bool external_sampling = false;
  int max_info_sets = 500000;
};

absl::Status ValidateSolverConfig(const SolverConfig& config);

struct ComboRange {
  std::array<float, kComboCount> weights = {};

  void add(ComboId combo, float weight = 1.0f) {
    if (weight > 0.0f) {
      weights[combo.index()] += weight;
    }
  }

  size_t count() const {
    return static_cast<size_t>(std::ranges::count_if(
        weights, [](float weight) { return weight > 0.0f; }));
  }
  float weight(ComboId combo) const { return weights[combo.index()]; }
};

struct SolveSpec {
  SolverConfig config;
  ExactPublicState root;
  std::array<ComboRange, kPlayerCount> ranges;
};

enum class ModelFingerprint : uint64_t {};

ComboRange UniformComboRange();
ModelFingerprint ModelFingerprintFor(
    const SolverConfig& config,
    const ExactPublicState& root,
    const std::array<ComboRange, kPlayerCount>& ranges) noexcept;

struct InfoSetKey {
  PublicObservationId public_observation;
  HistoryId history;
  PrivateObservationId private_observation;

  friend auto operator<=>(const InfoSetKey&, const InfoSetKey&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const InfoSetKey& key) {
    return H::combine(std::move(h), std::to_underlying(key.history),
                      std::to_underlying(key.public_observation),
                      std::to_underlying(key.private_observation));
  }
};
static_assert(sizeof(InfoSetKey) == 16);

struct CfrState {
  CfrState(const SolverConfig& config, bool accumulate_average_strategy);

  std::vector<float> regret_sum;
  std::vector<float> strategy_sum;
  uint64_t iterations = 0;
  double root_value_sum = 0.0;

  void strategy(std::span<float> values,
                std::optional<uint32_t> offset,
                std::span<float> probabilities) const;
  void add_regret(uint32_t offset, size_t action, double delta);
  void add_strategy(uint32_t offset,
                    std::span<const float> probabilities,
                    double weight);
  size_t row_count() const { return rows_.size(); }
  std::optional<uint32_t> find(InfoSetKey key) const;
  std::vector<std::pair<InfoSetKey, uint32_t>> row_entries() const;
  bool at_capacity() const { return row_count() >= max_info_sets_; }
  std::optional<uint32_t> find_or_create(InfoSetKey key, uint8_t action_count);

 private:
  absl::flat_hash_map<InfoSetKey, uint32_t> rows_;
  size_t max_info_sets_;
  bool accumulate_average_strategy_;
};

struct Policy {
  absl::flat_hash_map<InfoSetKey, uint32_t> rows;
  std::vector<float> probabilities;
  ModelFingerprint model{};

  bool strategy(InfoSetKey key, std::span<float> output) const;
};

absl::StatusOr<Policy> ExtractAveragePolicy(const CfrState& state,
    const HistoryTree& history, ModelFingerprint model);

struct SolverStats {
  uint64_t decision_visits = 0;
  uint64_t chance_samples = 0;
  uint64_t terminal_visits = 0;
};

struct Deal {
  std::array<ComboId, kPlayerCount> hands = {};

  ComboId hand(Player player) const {
    return hands[Index(player)];
  }
  CardMask blocked_mask() const {
    return hands[0].mask() | hands[1].mask();
  }
};

struct ObservedPosition {
  PublicPosition public_position;
  std::array<PrivateObservationId, kPlayerCount> private_observations;

  static ObservedPosition Observe(PublicPosition public_position,
                                  const Deal& deal) {
    std::array<PrivateObservationId, kPlayerCount> observations;
    for (Player player : {Player::A, Player::B}) {
      observations[Index(player)] =
          ObservePrivate(deal.hand(player), public_position);
    }
    return {std::move(public_position), observations};
  }

  ObservedPosition advance(PublicPosition child_public,
                           const BettingState& child_state,
                           const Deal& deal) const {
    auto child_observations = private_observations;
    if (std::holds_alternative<DecisionState>(child_state)) {
      for (Player player : {Player::A, Player::B}) {
        auto& observation = child_observations[Index(player)];
        observation = ObservePrivate(
            deal.hand(player), child_public, observation);
      }
    }
    return {std::move(child_public), child_observations};
  }

  const Board& board() const noexcept { return public_position.board(); }
  PublicObservationId public_observation() const noexcept {
    return public_position.observation();
  }
  PrivateObservationId private_observation(Player player) const noexcept {
    return private_observations[Index(player)];
  }
  InfoSetKey info_set_key(HistoryId history, Player player) const noexcept {
    return {public_observation(), history, private_observation(player)};
  }
};

struct ChanceSampler {
  const CardAbstractionConfig& card_abstraction;
  const Deal& deal;
  std::mt19937& rng;

  ObservedPosition operator()(const ObservedPosition& position,
                              const ChanceState& chance,
                              const BettingState& child_state) const;
};

class DealDistribution {
 public:
  static absl::StatusOr<DealDistribution> Create(
      const ComboRange& player_a,
      const ComboRange& player_b);

 Deal sample(std::mt19937& rng) const;

 private:
  std::array<std::vector<ComboId>, kPlayerCount> hands_;
  std::array<std::vector<float>, kPlayerCount> cumulative_weights_;
};

struct TabularCfrSolverTestAccess;

class TabularCfrSolver {
 public:
  static absl::StatusOr<TabularCfrSolver> Create(SolveSpec spec);

  void run(uint64_t iterations, int threads = 1);

  double evaluate_current(int samples);
  absl::StatusOr<double> evaluate_average(int samples);
  absl::StatusOr<Policy> extract_average_policy() const;

  double expected_value(Player player) const;
  uint64_t iterations() const noexcept { return state_.iterations; }
  size_t info_set_count() const { return state_.row_count(); }
  size_t history_count() const noexcept { return history_.nodes.size(); }
  size_t regret_bytes() const noexcept {
    return state_.regret_sum.size() * sizeof(float);
  }
  size_t strategy_bytes() const noexcept {
    return state_.strategy_sum.size() * sizeof(float);
  }
  const SolverConfig& config() const noexcept { return config_; }
  const DealDistribution& deals() const noexcept { return deals_; }
  const HistoryTree& history() const noexcept { return history_; }
  const PublicPosition& initial_public() const noexcept {
    return initial_public_;
  }
  ModelFingerprint model() const noexcept { return model_; }
  const SolverStats& stats() const noexcept { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  friend struct TabularCfrSolverTestAccess;

  TabularCfrSolver(SolverConfig config,
                   DealDistribution deals,
                   HistoryTree history,
                   PublicPosition initial_public,
                   ModelFingerprint model);

  enum class EvaluationMode : uint8_t {
    Current,
    Average,
  };

  double evaluate_deal(const Deal& deal, EvaluationMode mode);
  double evaluate_deals(int samples, EvaluationMode mode);
  SolverConfig config_;
  DealDistribution deals_;
  HistoryTree history_;
  PublicPosition initial_public_;
  ModelFingerprint model_;
  std::mt19937 rng_;
  CfrState state_;
  SolverStats stats_;
};

}  // namespace poker
