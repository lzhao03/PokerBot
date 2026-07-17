#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

  static absl::StatusOr<SolverConfig> Create(SolverConfig config);
};

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

absl::StatusOr<ComboRange> ParseRange(std::string_view text);
ComboRange UniformComboRange();
ComboRange SingleComboRange(ComboId combo, float weight = 1.0f);

enum class HistoryId : uint32_t {};

constexpr size_t Index(HistoryId history) noexcept {
  return std::to_underlying(history);
}

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
  CfrState(const SolverConfig& config,
           bool accumulate_average_strategy);

  std::vector<float> regret_sum;
  std::vector<float> strategy_sum;
  uint64_t iterations = 0;
  double cumulative_root_utility = 0.0;

  void strategy(absl::Span<const float> values,
                std::optional<size_t> offset,
                absl::Span<float> probabilities,
                bool concurrent = false) const;
  void add_regret(size_t offset,
                  size_t action,
                  float delta,
                  bool concurrent = false);
  void add_strategy(size_t offset,
                    absl::Span<const float> probabilities,
                    double weight,
                    bool concurrent = false);
  size_t row_count() const;
  std::optional<size_t> find(InfoSetKey key) const;
  bool contains(InfoSetKey key) const { return find(key).has_value(); }
  std::vector<std::pair<InfoSetKey, size_t>> row_entries() const;
  bool at_capacity() const { return row_count() >= max_info_sets_; }
  std::optional<size_t> find_or_create(
      InfoSetKey key,
      uint8_t action_count);

 private:
  bool can_pack(InfoSetKey key) const;
  uint64_t pack(InfoSetKey key) const;
  InfoSetKey unpack(uint64_t key) const;
  void use_full_keys();

  absl::flat_hash_map<uint64_t, size_t> packed_rows_;
  absl::flat_hash_map<InfoSetKey, size_t> full_rows_;
  size_t max_info_sets_;
  bool accumulate_average_strategy_;
  uint8_t private_bits_ = 0;
  uint8_t history_bits_ = 0;
  bool packed_keys_ = false;
};

struct Policy {
  absl::flat_hash_map<InfoSetKey, size_t> rows;
  std::vector<float> probabilities;
  ModelFingerprint model{};

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

struct CompiledGame {
  SolveSpec spec;
  DealDistribution deals;
  HistoryTree history;
  ModelFingerprint model{};
};

absl::StatusOr<CompiledGame> CompileGame(SolveSpec spec);

struct TabularCfrSolverTestAccess;

class TabularCfrSolver {
 public:
  static absl::StatusOr<TabularCfrSolver> Create(SolveSpec spec);

  void run(uint64_t iterations, int threads = 1);

  double evaluate_current(ComboId player_a, ComboId player_b);
  double evaluate_current(int samples);
  absl::StatusOr<double> evaluate_average(ComboId player_a,
                                          ComboId player_b);
  absl::StatusOr<double> evaluate_average(int samples);
  absl::StatusOr<Policy> extract_average_policy() const;

  double expected_value(Player player) const;
  uint64_t iterations() const noexcept { return state_.iterations; }
  size_t info_set_count() const { return state_.row_count(); }
  size_t history_count() const noexcept { return game_.history.nodes.size(); }
  size_t regret_bytes() const noexcept {
    return state_.regret_sum.size() * sizeof(float);
  }
  size_t strategy_bytes() const noexcept {
    return state_.strategy_sum.size() * sizeof(float);
  }
  const CompiledGame& game() const noexcept { return game_; }
  const SolverStats& stats() const noexcept { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  friend struct TabularCfrSolverTestAccess;

  explicit TabularCfrSolver(CompiledGame game);

  enum class EvaluationMode : uint8_t {
    Current,
    Average,
  };

  double evaluate_deal(const Deal& deal, EvaluationMode mode);
  double evaluate_deals(int samples, EvaluationMode mode);
  CompiledGame game_;
  std::mt19937 rng_;
  CfrState state_;
  SolverStats stats_;
};

}  // namespace poker
