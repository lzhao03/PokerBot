#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
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

struct SolverConfigOptions {
  BetAbstractionConfig bet_abstraction;
  CardAbstractionConfig card_abstraction;
  Chips starting_stack = 100;
  Chips small_blind = 1;
  Chips big_blind = 2;
  int chance_samples = 1;
  bool accumulate_average_strategy = true;
  int max_info_sets = 500000;
};

class SolverConfig {
 public:
  static absl::StatusOr<SolverConfig> Create(SolverConfigOptions options);
  static SolverConfig Default();

  const BetAbstractionConfig& bet_abstraction() const noexcept {
    return options_.bet_abstraction;
  }
  const CardAbstractionConfig& card_abstraction() const noexcept {
    return options_.card_abstraction;
  }
  Chips starting_stack() const noexcept { return options_.starting_stack; }
  Chips small_blind() const noexcept { return options_.small_blind; }
  Chips big_blind() const noexcept { return options_.big_blind; }
  int chance_samples() const noexcept { return options_.chance_samples; }
  bool accumulate_average_strategy() const noexcept {
    return options_.accumulate_average_strategy;
  }
  int max_info_sets() const noexcept { return options_.max_info_sets; }

 private:
  explicit SolverConfig(SolverConfigOptions options)
      : options_(std::move(options)) {}

  SolverConfigOptions options_;
};

struct ComboRange {
  std::array<float, kComboCount> weights = {};
  std::array<ComboId, kComboCount> active = {};
  uint16_t active_count = 0;

  void add(ComboId combo, float weight = 1.0f) {
    if (weight > 0.0f) {
      if (weights[combo.index()] == 0.0f) {
        active[active_count++] = combo;
      }
      weights[combo.index()] += weight;
    }
  }

  size_t count() const { return active_count; }
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

struct HistoryEdge {
  GameAction action;
  HistoryId child;
};

struct EdgeRange {
  uint32_t begin = 0;
  uint8_t count = 0;
};

struct DecisionNode {
  DecisionState state;
  EdgeRange edges;
};

struct ChanceNode {
  ChanceState state;
  HistoryId child;
};

struct FoldTerminalNode {
  FoldTerminalState state;
};

struct ShowdownNode {
  ShowdownState state;
};

using HistoryNode = std::variant<DecisionNode, ChanceNode,
                                 FoldTerminalNode, ShowdownNode>;

struct HistoryTree {
  HistoryId root;
  std::vector<HistoryNode> nodes;
  std::vector<HistoryEdge> edges;
};

struct Position {
  HistoryId history;
  PublicPosition public_state;
};

struct InfoSetKey {
  HistoryId history;
  PublicObservationId public_observation;
  PrivateObservationId private_observation;

  friend bool operator==(const InfoSetKey&, const InfoSetKey&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const InfoSetKey& key) {
    return H::combine(std::move(h), key.history.value(),
                      key.public_observation.value(),
                      key.private_observation.value());
  }
};

struct InfoSetRow {
  size_t action_offset = 0;

  friend bool operator==(const InfoSetRow&, const InfoSetRow&) = default;
};

struct CfrState {
  absl::flat_hash_map<InfoSetKey, InfoSetRow> rows;
  std::vector<float> regret_sum;
  std::vector<float> strategy_sum;
  uint64_t iterations = 0;
  double cumulative_root_utility = 0.0;
};

struct PolicyRow {
  size_t action_offset = 0;
  uint8_t action_count = 0;

  friend bool operator==(const PolicyRow&, const PolicyRow&) = default;
};

struct Policy {
  absl::flat_hash_map<InfoSetKey, PolicyRow> rows;
  std::vector<float> probabilities;
  ModelFingerprint model;

  bool strategy(InfoSetKey key, absl::Span<float> output) const;
};

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model);
absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model,
    size_t max_serialized_bytes);

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path);
absl::StatusOr<Policy> LoadPolicy(const std::filesystem::path& path);

struct SerializedRngState {
  std::string text;

  friend bool operator==(const SerializedRngState&,
                         const SerializedRngState&) = default;
};

struct SolverCheckpoint {
  uint32_t format_version = 1;
  ModelFingerprint model;
  CfrState state;
  SerializedRngState rng;
};

absl::Status SaveCheckpoint(const SolverCheckpoint& checkpoint,
                            const std::filesystem::path& path);
absl::StatusOr<SolverCheckpoint> LoadCheckpoint(
    const std::filesystem::path& path);

struct SolverStats {
  uint64_t decision_visits = 0;
  uint64_t chance_samples = 0;
  uint64_t terminal_visits = 0;
};

enum class TrainingStopReason : uint8_t {
  IterationsCompleted,
  InfoSetLimit,
};

struct TrainingResult {
  uint64_t iterations_completed = 0;
  TrainingStopReason stop_reason =
      TrainingStopReason::IterationsCompleted;
};

struct Deal {
  std::array<HoleCards, kPlayerCount> hands = {};
  CardMask blocked_mask = 0;

  const HoleCards& hand(Player player) const {
    return hands[Index(player)];
  }
};

class DealDistribution {
 public:
  static absl::StatusOr<DealDistribution> Create(
      const ComboRange& player_a,
      const ComboRange& player_b);

  Deal sample(std::mt19937& rng) const;

 private:
  static size_t SampleIndex(absl::Span<const float> cumulative,
                            float total,
                            std::mt19937& rng);

  std::vector<ComboId> a_hands_;
  std::vector<float> a_cumulative_;
  std::vector<uint32_t> b_offsets_;
  std::vector<uint16_t> b_counts_;
  std::vector<ComboId> b_hands_;
  std::vector<float> b_cumulative_;
  float total_ = 0.0f;
};

struct CFRSolverTestAccess;

class CFRSolver {
 public:
  static absl::StatusOr<std::unique_ptr<CFRSolver>> Create(SolveSpec spec);

  TrainingResult run(uint64_t iterations);

  double evaluate_current(HoleCards player_a,
                          HoleCards player_b);
  double evaluate_current(int samples);
  absl::StatusOr<double> evaluate_average(HoleCards player_a,
                                          HoleCards player_b);
  absl::StatusOr<double> evaluate_average(int samples);
  absl::StatusOr<Policy> extract_average_policy() const;
  absl::StatusOr<Policy> extract_average_policy(
      size_t max_serialized_bytes) const;
  SolverCheckpoint checkpoint() const;
  absl::Status restore(SolverCheckpoint checkpoint);

  double get_expected_value(Player player) const;
  uint64_t get_iterations_run() const { return state_.iterations; }
  uint64_t get_cfr_update_count() const { return stats_.decision_visits; }
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
  const CardAbstraction& card_abstraction() const noexcept {
    return card_abstraction_;
  }
  SolverStats get_stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  friend struct CFRSolverTestAccess;

  CFRSolver(SolveSpec spec,
            DealDistribution deals,
            CardAbstraction card_abstraction);

  struct TraversalFrame {
    std::array<double, kPlayerCount> reach = {1.0, 1.0};
    std::array<PrivateObservationId, kPlayerCount> private_observations = {};
  };

  enum class TraversalMode : uint8_t {
    Train,
    EvaluateCurrent,
    EvaluateAverage,
  };

  struct TraversalContext {
    const Deal& deal;
    TraversalMode mode = TraversalMode::Train;
    std::optional<Player> update_player;
    uint64_t iteration = 0;
    bool info_set_limit_reached = false;
  };

  Position root_position() const;
  Position action_child(Position position, uint8_t action_index) const;
  Position sample_chance_child(Position position, const Deal& deal);
  std::array<PrivateObservationId, kPlayerCount>
  private_observations_for_position(const Deal& deal,
                                    const Position& position) const;
  void advance_private_observations(TraversalFrame& frame,
                                    const Deal& deal,
                                    const Position& child) const;
  double traverse(Position position,
                  TraversalFrame frame,
                  TraversalContext& context);
  double evaluate_deal(const Deal& deal, TraversalMode mode);
  double evaluate_deals(int samples, TraversalMode mode);
  std::optional<InfoSetRow> find_or_create_row(InfoSetKey key,
                                                uint8_t action_count);
  std::optional<InfoSetRow> find_row(InfoSetKey key) const;
  void log_training_summary() const;

  SolveSpec spec_;
  BettingRules betting_rules_;
  DealDistribution deals_;
  CardAbstraction card_abstraction_;
  std::mt19937 rng_;
  HistoryTree history_;
  ModelFingerprint model_;
  CfrState state_;
  SolverStats stats_;
};

}  // namespace poker
