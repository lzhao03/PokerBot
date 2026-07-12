#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/bet_abstraction.h"
#include "src/card_abstraction.h"
#include "src/fingerprint.h"
#include "src/poker.h"

namespace poker {

struct SolverConfig {
  BetAbstractionConfig bet_abstraction;
  CardAbstractionConfig card_abstraction;
  BettingRules betting_rules = {2};
  int chance_samples = 1;
  bool accumulate_average_strategy = true;
  int max_info_sets = 500000;

  static absl::StatusOr<SolverConfig> Create(SolverConfig config);
};

struct ComboRange {
  std::array<float, kComboCount> weights = {};
  std::vector<ComboId> combos;

  void add(ComboId combo, float weight = 1.0f) {
    if (weight > 0.0f) {
      if (weights[combo.index()] == 0.0f) {
        combos.push_back(combo);
      }
      weights[combo.index()] += weight;
    }
  }

  size_t count() const { return combos.size(); }
  float weight(ComboId combo) const { return weights[combo.index()]; }
};

struct SolveSpec {
  SolverConfig config;
  ExactPublicState root;
  std::array<ComboRange, kPlayerCount> ranges;
};

absl::StatusOr<ComboRange> ParseRange(std::string_view text);
ComboRange UniformComboRange();
ComboRange SingleComboRange(ComboId combo, float weight = 1.0f);

class HistoryId {
 public:
  constexpr HistoryId() = default;
  explicit constexpr HistoryId(uint32_t value) noexcept : value_(value) {}

  constexpr size_t index() const noexcept { return value_; }
  constexpr uint32_t value() const noexcept { return value_; }
  friend constexpr auto operator<=>(const HistoryId&,
                                    const HistoryId&) = default;

 private:
  uint32_t value_ = 0;
};

struct HistoryNode {
  BettingState state;
  uint32_t children_begin = 0;
  uint8_t child_count = 0;
};

struct HistoryTree {
  std::vector<HistoryNode> nodes;
  std::vector<HistoryId> children;
};

struct Position {
  HistoryId history;
  PublicPosition public_state;
};

struct InfoSetKey {
  HistoryId history;
  PublicObservationId public_observation;
  PrivateObservationId private_observation;

  friend auto operator<=>(const InfoSetKey&, const InfoSetKey&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const InfoSetKey& key) {
    return H::combine(std::move(h), key.history.value(),
                      std::to_underlying(key.public_observation),
                      std::to_underlying(key.private_observation));
  }
};

struct CfrState {
  absl::flat_hash_map<InfoSetKey, size_t> rows;
  std::vector<float> regret_sum;
  std::vector<float> strategy_sum;
  uint64_t iterations = 0;
  double cumulative_root_utility = 0.0;

  void reserve(const SolverConfig& config,
               bool accumulate_average_strategy);
  void regret_matching_strategy(
      const size_t* offset,
      absl::Span<double> probabilities,
      bool concurrent = false) const;
  void add_regret(size_t offset,
                  size_t action,
                  float delta,
                  bool concurrent = false);
  void add_strategy(size_t offset,
                    absl::Span<const double> probabilities,
                    double weight,
                    bool concurrent = false);
  std::optional<size_t> find_or_create(
      InfoSetKey key,
      uint8_t action_count,
      size_t max_info_sets,
      bool accumulate_average_strategy);
};

struct Policy {
  absl::flat_hash_map<InfoSetKey, size_t> rows;
  std::vector<float> probabilities;
  ModelFingerprint model;

  bool strategy(InfoSetKey key, absl::Span<float> output) const;
};

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model);

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path);

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

struct CFRSolverTestAccess;

class CFRSolver {
 public:
  static absl::StatusOr<std::unique_ptr<CFRSolver>> Create(SolveSpec spec);

  void run(uint64_t iterations, int threads = 1);

  double evaluate_current(ComboId player_a, ComboId player_b);
  double evaluate_current(int samples);
  absl::StatusOr<double> evaluate_average(ComboId player_a,
                                          ComboId player_b);
  absl::StatusOr<double> evaluate_average(int samples);
  absl::StatusOr<Policy> extract_average_policy() const;

  double get_expected_value(Player player) const;
  uint64_t get_iterations_run() const { return state_.iterations; }
  size_t get_info_set_count() const { return state_.rows.size(); }
  size_t get_history_count() const { return history_.nodes.size(); }
  size_t get_regret_bytes() const {
    return state_.regret_sum.size() * sizeof(float);
  }
  size_t get_strategy_bytes() const {
    return state_.strategy_sum.size() * sizeof(float);
  }
  const ModelFingerprint& model_fingerprint() const noexcept {
    return model_;
  }
  const SolveSpec& solve_spec() const noexcept { return spec_; }
  const HistoryTree& history_tree() const noexcept { return history_; }
  const DealDistribution& deal_distribution() const noexcept {
    return deals_;
  }
  const CardAbstractionConfig& card_abstraction() const noexcept {
    return spec_.config.card_abstraction;
  }
  SolverStats get_stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  friend struct CFRSolverTestAccess;

  CFRSolver(SolveSpec spec, DealDistribution deals);

  struct TraversalFrame {
    std::array<double, kPlayerCount> reach = {1.0, 1.0};
    std::array<PrivateObservationId, kPlayerCount> private_observations = {};
    std::optional<int8_t> showdown_comparison;
  };

  enum class TraversalMode : uint8_t {
    Train,
    EvaluateCurrent,
    EvaluateAverage,
  };

  struct TraversalContext {
    const Deal& deal;
    TraversalMode mode = TraversalMode::Train;
    uint64_t iteration = 0;
    std::mt19937& rng;
    SolverStats& stats;
    bool concurrent_updates = false;
  };

  Position root_position() const;
  Position sample_chance_child(const HistoryNode& node,
                               const PublicPosition& public_state,
                               const Deal& deal,
                               std::mt19937& rng);
  TraversalFrame initial_frame(const Deal& deal,
                               const Position& position) const;
  void advance_private_observations(TraversalFrame& frame,
                                    const Deal& deal,
                                    const Position& child) const;
  double traverse(HistoryId history,
                  const PublicPosition& public_state,
                  const TraversalFrame& frame,
                  TraversalContext& context);
  double evaluate_deal(const Deal& deal, TraversalMode mode);
  double evaluate_deals(int samples, TraversalMode mode);
  SolveSpec spec_;
  DealDistribution deals_;
  std::mt19937 rng_;
  HistoryTree history_;
  ModelFingerprint model_;
  CfrState state_;
  SolverStats stats_;
};

}  // namespace poker
