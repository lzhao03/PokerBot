#include "src/solver.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

static_assert(std::atomic_ref<float>::is_always_lock_free);

constexpr uint32_t kModelFingerprintSchemaVersion = 4;

template <std::integral Integer>
void AppendInteger(std::vector<uint8_t>& bytes, Integer value) {
  using Unsigned = std::make_unsigned_t<Integer>;
  const Unsigned bits = static_cast<Unsigned>(value);
  for (size_t index = 0; index < sizeof(Integer); ++index) {
    bytes.push_back(
        static_cast<uint8_t>(bits >> static_cast<unsigned>(index * 8)));
  }
}

// ponytail: Use a library hash if fingerprints ever cross a hostile boundary.
ModelFingerprint Fingerprint(std::span<const uint8_t> bytes) noexcept {
  uint64_t hash = 14695981039346656037ULL;
  for (uint8_t byte : bytes) {
    hash = (hash ^ byte) * 1099511628211ULL;
  }
  return ModelFingerprint{hash};
}

void AddBettingData(std::vector<uint8_t>& bytes,
                    const BettingData& data) noexcept {
  for (Chips value : data.stack) AppendInteger(bytes, value);
  for (Chips value : data.total_committed) AppendInteger(bytes, value);
  for (Chips value : data.street_committed) AppendInteger(bytes, value);
  AppendInteger(bytes, data.last_full_raise);
  bytes.push_back(std::to_underlying(data.street));
  bytes.push_back(data.actions_remaining);
}

void AddBettingState(std::vector<uint8_t>& bytes,
                     const BettingState& state) noexcept {
  bytes.push_back(static_cast<uint8_t>(state.index()));
  AddBettingData(bytes, Data(state));
  if (const auto* decision = std::get_if<DecisionState>(&state)) {
    bytes.push_back(std::to_underlying(decision->actor));
  } else if (const auto* fold = std::get_if<FoldTerminalState>(&state)) {
    bytes.push_back(std::to_underlying(fold->folded));
  }
}

void AddBoard(std::vector<uint8_t>& bytes, const Board& board) noexcept {
  bytes.push_back(static_cast<uint8_t>(board.count()));
  for (Card card : board.cards()) {
    bytes.push_back(static_cast<uint8_t>(card.index()));
  }
}

void AddRange(std::vector<uint8_t>& bytes, const ComboRange& range) noexcept {
  for (float weight : range.weights) {
    AppendInteger(bytes, std::bit_cast<uint32_t>(weight));
  }
}

}  // namespace

ModelFingerprint ModelFingerprintFor(
    const SolverConfig& config,
    const ExactPublicState& root,
    const std::array<ComboRange, kPlayerCount>& ranges) noexcept {
  std::vector<uint8_t> bytes;
  const uint32_t schema = kModelFingerprintSchemaVersion |
      (static_cast<uint32_t>(kBetAbstractionSchemaVersion) << 8) |
      (static_cast<uint32_t>(kCardAbstractionSchemaVersion) << 16) |
      (static_cast<uint32_t>(kHistorySchemaVersion) << 24);
  AppendInteger(bytes, schema);

  AppendInteger(bytes, config.betting_rules.minimum_bet);
  bytes.push_back(std::to_underlying(config.card_abstraction.public_mode));
  bytes.push_back(std::to_underlying(config.card_abstraction.private_kind));
  bytes.push_back(std::to_underlying(config.card_abstraction.recall_mode));
  for (const auto& fractions :
       config.bet_abstraction.pot_fractions) {
    AppendInteger(bytes, static_cast<uint32_t>(fractions.size()));
    for (double fraction : fractions) {
      AppendInteger(bytes, std::bit_cast<uint64_t>(fraction));
    }
  }

  AddBettingState(bytes, root.betting);
  AddBoard(bytes, root.board);
  for (const ComboRange& range : ranges) AddRange(bytes, range);
  return Fingerprint(bytes);
}

absl::StatusOr<DealDistribution> DealDistribution::Create(
    const ComboRange& player_a,
    const ComboRange& player_b) {
  DealDistribution distribution;
  const std::array ranges = {&player_a, &player_b};
  for (size_t player = 0; player < ranges.size(); ++player) {
    float total = 0.0f;
    for (size_t first = 0; first < kDeck.size(); ++first) {
      for (size_t second = first + 1; second < kDeck.size(); ++second) {
        const ComboId hand = CardsToComboId(kDeck[first], kDeck[second]);
        const float weight = ranges[player]->weight(hand);
        if (weight <= 0.0f) continue;
        distribution.hands_[player].push_back(hand);
        total += weight;
        distribution.cumulative_weights_[player].push_back(total);
      }
    }
  }
  const bool compatible = std::ranges::any_of(
      distribution.hands_[0], [&](ComboId a) {
        return std::ranges::any_of(
            distribution.hands_[1], [&](ComboId b) {
              return (a.mask() & b.mask()) == 0;
            });
      });
  if (!compatible) {
    return absl::InvalidArgumentError(
        "ranges contain no non-overlapping hands");
  }
  return distribution;
}

Deal DealDistribution::sample(std::mt19937& rng) const {
  auto sample_player = [&](size_t player) {
    const auto& cumulative = cumulative_weights_[player];
    std::uniform_real_distribution<float> distribution(
        0.0f, cumulative.back());
    const auto found = std::upper_bound(
        cumulative.begin(), cumulative.end(), distribution(rng));
    const size_t index = found == cumulative.end()
                             ? cumulative.size() - 1
                             : static_cast<size_t>(found - cumulative.begin());
    return hands_[player][index];
  };
  Deal deal;
  do {
    deal.hands = {sample_player(0), sample_player(1)};
  } while ((deal.hands[0].mask() & deal.hands[1].mask()) != 0);
  // ponytail: precompute conditional ranges only if near-total overlap is
  // measured as a sampling bottleneck.
  return deal;
}

namespace {

void FillUniform(std::span<float> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0f / static_cast<float>(probabilities.size()));
  }
}

}  // namespace

void CfrState::add_regret(uint32_t offset, size_t action, double delta) {
  float& regret = regret_sum[offset + action];
  const float update = static_cast<float>(delta);
  std::atomic_ref<float> atomic(regret);
  float old = atomic.load(std::memory_order_relaxed);
  while (!atomic.compare_exchange_weak(
      old, std::max(0.0f, old + update), std::memory_order_relaxed)) {}
}

void CfrState::add_strategy(uint32_t offset,
                            std::span<const float> probabilities,
                            double weight) {
  if (!accumulate_average_strategy_) return;
  for (float probability : probabilities) {
    float& sum = strategy_sum[offset++];
    const float delta = static_cast<float>(weight * probability);
    std::atomic_ref<float>(sum).fetch_add(delta, std::memory_order_relaxed);
  }
}

void CfrState::strategy(std::span<float> values,
                        std::optional<uint32_t> offset,
                        std::span<float> probabilities) const {
  if (!offset) {
    FillUniform(probabilities);
    return;
  }
  float sum = 0.0f;
  float* value = values.data() + *offset;
  for (float& probability : probabilities) {
    probability = std::atomic_ref(*value++).load(std::memory_order_relaxed);
    sum += probability;
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
  } else {
    const float scale = 1.0f / sum;
    for (float& probability : probabilities) probability *= scale;
  }
}

CfrState::CfrState(const SolverConfig& config, bool accumulate_average_strategy)
    : max_info_sets_(static_cast<size_t>(config.max_info_sets)),
      accumulate_average_strategy_(accumulate_average_strategy) {
  size_t max_actions = 3;
  for (const auto& fractions : config.bet_abstraction.pot_fractions) {
    max_actions = std::max(max_actions, fractions.size() + 3);
  }
  rows_.reserve(max_info_sets_);
  regret_sum.reserve(max_info_sets_ * max_actions);
  if (accumulate_average_strategy) {
    strategy_sum.reserve(max_info_sets_ * max_actions);
  }
}

std::optional<uint32_t> CfrState::find(InfoSetKey key) const {
  const auto row = rows_.find(key);
  return row == rows_.end() ? std::nullopt : std::optional(row->second);
}

std::vector<std::pair<InfoSetKey, uint32_t>> CfrState::row_entries() const {
  std::vector<std::pair<InfoSetKey, uint32_t>> rows(rows_.begin(), rows_.end());
  std::ranges::sort(rows);
  return rows;
}

std::optional<uint32_t> CfrState::find_or_create(
    InfoSetKey key, uint8_t action_count) {
  if (at_capacity()) return find(key);
  const uint32_t offset = static_cast<uint32_t>(regret_sum.size());
  const auto [row, inserted] = rows_.try_emplace(key, offset);
  if (!inserted) return row->second;
  regret_sum.resize(offset + action_count);
  if (accumulate_average_strategy_) strategy_sum.resize(offset + action_count);
  return offset;
}

bool Policy::strategy(InfoSetKey key, std::span<float> output) const {
  const auto row = rows.find(key);
  const size_t count = output.size();
  if (row == rows.end() || row->second + count > probabilities.size()) {
    FillUniform(output);
    return false;
  }
  std::copy_n(probabilities.data() + row->second, count, output.data());
  return true;
}

absl::Status ValidateSolverConfig(const SolverConfig& config) {
  if (config.betting_rules.minimum_bet <= 0) {
    return absl::InvalidArgumentError("minimum bet must be positive");
  }
  if (config.chance_samples <= 0) {
    return absl::InvalidArgumentError("chance_samples must be positive");
  }
  if (config.max_info_sets <= 0) {
    return absl::InvalidArgumentError("max_info_sets must be positive");
  }
  if (config.max_info_sets >
      static_cast<int>(std::numeric_limits<uint32_t>::max() /
                       kMaxActionsPerNode)) {
    return absl::InvalidArgumentError("max_info_sets is too large");
  }
  for (const auto& fractions : config.bet_abstraction.pot_fractions) {
    if (fractions.size() > kMaxActionsPerNode - size_t{3}) {
      return absl::InvalidArgumentError("too many pot fractions");
    }
    for (size_t index = 0; index < fractions.size(); ++index) {
      const double fraction = fractions[index];
      if (!std::isfinite(fraction) || fraction <= 0.0) {
        return absl::InvalidArgumentError(
            "pot fractions must be finite and positive");
      }
      if (index > 0 && fractions[index - 1] >= fraction) {
        return absl::InvalidArgumentError(
            "pot fractions must be strictly increasing");
      }
    }
  }
  return absl::OkStatus();
}

ComboRange UniformComboRange() {
  ComboRange range;
  range.weights.fill(1.0f);
  return range;
}

TabularCfrSolver::TabularCfrSolver(SolverConfig config,
                                   DealDistribution deals,
                                   HistoryTree history,
                                   PublicPosition initial_public,
                                   ModelFingerprint model)
    : config_(std::move(config)),
      deals_(std::move(deals)),
      history_(std::move(history)),
      initial_public_(std::move(initial_public)),
      model_(model),
      rng_(12345),
      state_(config_, config_.accumulate_average_strategy) {}

absl::StatusOr<TabularCfrSolver> TabularCfrSolver::Create(SolveSpec spec) {
  const absl::Status config_status = ValidateSolverConfig(spec.config);
  if (!config_status.ok()) return config_status;
  if (!IsValidBettingData(Data(spec.root.betting))) {
    return absl::InvalidArgumentError("invalid root betting state");
  }
  auto deals = DealDistribution::Create(spec.ranges[Index(Player::A)],
                                        spec.ranges[Index(Player::B)]);
  if (!deals.ok()) return deals.status();
  PublicPosition initial_public(
      spec.config.card_abstraction, spec.root.board);
  const ModelFingerprint model =
      ModelFingerprintFor(spec.config, spec.root, spec.ranges);
  HistoryTree history = BuildHistoryTree(
      spec.root.betting, spec.config.betting_rules,
      spec.config.bet_abstraction);
  return TabularCfrSolver(
      std::move(spec.config), std::move(*deals), std::move(history),
      std::move(initial_public), model);
}

ObservedPosition ChanceSampler::operator()(
    const ObservedPosition& position,
    const ChanceState& chance,
    const BettingState& child_state) const {
  const auto sampled = SampleStreetCards(
      chance.data.street, position.board(), deal.blocked_mask(), rng);
  assert(sampled.ok());
  return position.advance(
      PublicPosition(
          card_abstraction, DealCards(position.board(), *sampled)),
      child_state, deal);
}

void TabularCfrSolver::run(uint64_t iterations, int threads) {
  if (iterations == 0) return;

  auto run_one = [&](uint64_t iteration, std::mt19937& rng,
                     SolverStats& stats) {
    const Deal deal = deals_.sample(rng);
    const Player update_player = iteration % 2 ? Player::B : Player::A;
    TerminalEvaluator terminal_utility(deal.hands);
    const ObservedPosition initial_position =
        ObservedPosition::Observe(initial_public_, deal);
    ChanceSampler sample_chance{config_.card_abstraction, deal, rng};
    auto cfr = [&](auto&& self,
                   HistoryId history,
                   const ObservedPosition& position,
                   std::array<double, kPlayerCount> reach) -> double {
      while (true) {
        const HistoryNode& node = history_.nodes[Index(history)];
        if (IsTerminal(node.state)) {
          ++stats.terminal_visits;
          return terminal_utility(node.state, position.board(), update_player);
        }
        if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
          stats.chance_samples += static_cast<uint64_t>(config_.chance_samples);
          const HistoryId child = history_.children[node.children_begin];
          const auto& child_state = history_.nodes[Index(child)].state;
          double value = 0.0;
          for (int draw = 0; draw < config_.chance_samples; ++draw) {
            const auto next = sample_chance(position, *chance, child_state);
            value += self(self, child, next, reach);
          }
          return value / config_.chance_samples;
        }

        const Player player = std::get<DecisionState>(node.state).actor;
        const uint8_t action_count = node.child_count;
        const bool traverser = update_player == player;
        const InfoSetKey key = position.info_set_key(history, player);
        const auto offset = traverser || config_.external_sampling
            ? state_.find_or_create(key, action_count) : state_.find(key);
        std::array<float, kMaxActionsPerNode> probabilities;
        const std::span<float> strategy(probabilities.data(), action_count);
        state_.strategy(state_.regret_sum, offset, strategy);
        ++stats.decision_visits;

        if (config_.external_sampling && !traverser) {
          if (offset) {
            state_.add_strategy(
                *offset, strategy, static_cast<double>(iteration + 1));
          }
          float sample = std::uniform_real_distribution<float>{}(rng);
          uint8_t sampled_action = 0;
          while (sampled_action + 1 < action_count &&
                 sample >= probabilities[sampled_action]) {
            sample -= probabilities[sampled_action];
            ++sampled_action;
          }
          history = history_.children[node.children_begin + sampled_action];
          continue;
        }

        std::array<double, kMaxActionsPerNode> action_values;
        double node_value = 0.0;
        for (uint8_t action = 0; action < action_count; ++action) {
          auto child_reach = reach;
          if (!config_.external_sampling) {
            child_reach[Index(player)] *= probabilities[action];
          }
          const HistoryId child =
              history_.children[node.children_begin + action];
          action_values[action] = self(self, child, position, child_reach);
          node_value += probabilities[action] * action_values[action];
        }
        if (!traverser || !offset) return node_value;

        for (uint8_t action = 0; action < action_count; ++action) {
          double regret = action_values[action] - node_value;
          if (!config_.external_sampling) {
            regret *= reach[Index(Opponent(player))];
          }
          state_.add_regret(*offset, action, regret);
        }
        if (!config_.external_sampling) {
          const double weight =
              reach[Index(player)] * static_cast<double>(iteration + 1);
          state_.add_strategy(*offset, strategy, weight);
        }
        return node_value;
      }
    };

    const double value =
        cfr(cfr, HistoryId{}, initial_position, {1.0, 1.0});
    return update_player == Player::A ? value : -value;
  };

  uint64_t remaining = iterations;
  // Fill the infoset map serially; workers only update it at capacity.
  while (remaining > 0 && (threads <= 1 || !state_.at_capacity())) {
    state_.root_value_sum += run_one(state_.iterations, rng_, stats_);
    ++state_.iterations;
    --remaining;
  }
  if (remaining == 0) return;

  const size_t pool_size = std::min<size_t>(
      static_cast<size_t>(threads), static_cast<size_t>(remaining));
  struct WorkerResult {
    double value_sum = 0.0;
    SolverStats stats;
  };
  std::vector<WorkerResult> worker_results(pool_size);
  std::vector<std::thread> workers;
  workers.reserve(pool_size);
  const uint64_t start = state_.iterations;
  for (size_t worker = 0; worker < pool_size; ++worker) {
    const uint32_t worker_seed = static_cast<uint32_t>(rng_());
    workers.emplace_back([&, worker, worker_seed] {
      std::seed_seq seed{worker_seed, static_cast<uint32_t>(worker)};
      std::mt19937 rng(seed);
      WorkerResult& output = worker_results[worker];
      for (uint64_t offset = worker; offset < remaining; offset += pool_size) {
        output.value_sum += run_one(start + offset, rng, output.stats);
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  for (const WorkerResult& worker : worker_results) {
    state_.root_value_sum += worker.value_sum;
    stats_.decision_visits += worker.stats.decision_visits;
    stats_.chance_samples += worker.stats.chance_samples;
    stats_.terminal_visits += worker.stats.terminal_visits;
  }
  state_.iterations += remaining;
}

double TabularCfrSolver::evaluate_deal(const Deal& deal,
                                       EvaluationMode mode) {
  TerminalEvaluator terminal_utility(deal.hands);
  const ObservedPosition initial_position =
      ObservedPosition::Observe(initial_public_, deal);
  ChanceSampler sample_chance{config_.card_abstraction, deal, rng_};
  auto evaluate = [&](auto&& self,
                      HistoryId history,
                      const ObservedPosition& position) -> double {
    const HistoryNode& node = history_.nodes[Index(history)];
    if (IsTerminal(node.state)) {
      ++stats_.terminal_visits;
      return terminal_utility(node.state, position.board(), Player::A);
    }
    if (const auto* chance = std::get_if<ChanceState>(&node.state)) {
      stats_.chance_samples += static_cast<uint64_t>(config_.chance_samples);
      const HistoryId child = history_.children[node.children_begin];
      const auto& child_state = history_.nodes[Index(child)].state;
      double value = 0.0;
      for (int draw = 0; draw < config_.chance_samples; ++draw) {
        const auto next = sample_chance(position, *chance, child_state);
        value += self(self, child, next);
      }
      return value / config_.chance_samples;
    }

    const Player player = std::get<DecisionState>(node.state).actor;
    const uint8_t action_count = node.child_count;
    const InfoSetKey key = position.info_set_key(history, player);
    std::array<float, kMaxActionsPerNode> probabilities;
    const std::span<float> strategy(probabilities.data(), action_count);
    std::span<float> values = mode == EvaluationMode::Average
                                  ? std::span<float>(state_.strategy_sum)
                                  : std::span<float>(state_.regret_sum);
    state_.strategy(values, state_.find(key), strategy);

    double value = 0.0;
    for (uint8_t action = 0; action < action_count; ++action) {
      const HistoryId child = history_.children[node.children_begin + action];
      value += probabilities[action] * self(self, child, position);
    }
    return value;
  };

  return evaluate(evaluate, HistoryId{}, initial_position);
}

double TabularCfrSolver::evaluate_deals(int samples, EvaluationMode mode) {
  if (samples <= 0) return 0.0;
  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    value += evaluate_deal(deals_.sample(rng_), mode);
  }
  return value / samples;
}

double TabularCfrSolver::evaluate_current(ComboId player_a,
                                          ComboId player_b) {
  const Deal deal{{player_a, player_b}};
  return evaluate_deal(deal, EvaluationMode::Current);
}

double TabularCfrSolver::evaluate_current(int samples) {
  return evaluate_deals(samples, EvaluationMode::Current);
}

absl::StatusOr<double> TabularCfrSolver::evaluate_average(
    ComboId player_a,
    ComboId player_b) {
  if (!config_.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  const Deal deal{{player_a, player_b}};
  return evaluate_deal(deal, EvaluationMode::Average);
}

absl::StatusOr<double> TabularCfrSolver::evaluate_average(int samples) {
  if (!config_.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return evaluate_deals(samples, EvaluationMode::Average);
}

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model) {
  std::vector<std::pair<InfoSetKey, uint32_t>> rows = state.row_entries();

  Policy policy;
  policy.model = model;
  for (const auto& [key, offset] : rows) {
    if (Index(key.history) >= history.nodes.size()) {
      return absl::DataLossError("infoset references an invalid history");
    }
    const HistoryNode& node = history.nodes[Index(key.history)];
    if (!std::holds_alternative<DecisionState>(node.state) ||
        offset + node.child_count > state.strategy_sum.size()) {
      return absl::DataLossError("infoset strategy span is invalid");
    }

    const uint32_t output_offset =
        static_cast<uint32_t>(policy.probabilities.size());
    double mass = 0.0;
    for (size_t action = 0; action < node.child_count; ++action) {
      const float value = state.strategy_sum[offset + action];
      if (!std::isfinite(value)) {
        return absl::DataLossError("nonfinite average strategy value");
      }
      policy.probabilities.push_back(std::max(0.0f, value));
      mass += policy.probabilities.back();
    }

    policy.rows.try_emplace(key, output_offset);
    std::span<float> probabilities(
        policy.probabilities.data() + output_offset, node.child_count);
    if (mass > 0.0) {
      for (float& probability : probabilities) {
        probability = static_cast<float>(probability / mass);
      }
    } else {
      std::fill(probabilities.begin(), probabilities.end(),
                1.0f / node.child_count);
    }
  }
  return policy;
}

absl::StatusOr<Policy> TabularCfrSolver::extract_average_policy() const {
  if (!config_.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return ExtractAveragePolicy(state_, history_, model_);
}

double TabularCfrSolver::expected_value(Player player) const {
  if (state_.iterations == 0) return 0.0;
  const double player_a_ev =
      state_.root_value_sum / static_cast<double>(state_.iterations);
  return player == Player::A ? player_a_ev : -player_a_ev;
}

}  // namespace poker
