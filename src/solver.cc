#include "src/solver.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "src/cfr_traversal.h"
#include "src/hand_evaluator.h"

namespace poker {
namespace {

static_assert(__atomic_always_lock_free(sizeof(float), nullptr));

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
ModelFingerprint Fingerprint(absl::Span<const uint8_t> bytes) noexcept {
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

ModelFingerprint FingerprintModel(const SolveSpec& spec) noexcept {
  std::vector<uint8_t> bytes;
  AppendInteger<uint32_t>(bytes, 4);  // Fingerprint schema.

  const SolverConfig& config = spec.config;
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

  AddBettingState(bytes, spec.root.betting);
  AddBoard(bytes, spec.root.board);
  for (const ComboRange& range : spec.ranges) AddRange(bytes, range);
  return Fingerprint(bytes);
}

enum class HandShape {
  Suited,
  Offsuit,
  Any,
};

struct HandType {
  int high;
  int low;
  HandShape shape;
};

std::optional<int> ParseRank(char rank) {
  const size_t index = std::string_view("23456789TJQKA").find(rank);
  if (index == std::string_view::npos) return std::nullopt;
  return static_cast<int>(index) + 2;
}

std::optional<HandType> ParseHandType(std::string_view text) {
  if (text.size() != 2 && text.size() != 3) {
    return std::nullopt;
  }
  const auto first = ParseRank(text[0]);
  const auto second = ParseRank(text[1]);
  if (!first || !second) {
    return std::nullopt;
  }
  HandType type{std::max(*first, *second), std::min(*first, *second),
                HandShape::Any};
  if (type.high == type.low) {
    return text.size() == 2 ? std::optional<HandType>(type) : std::nullopt;
  }
  if (text.size() == 2) return type;
  if (text[2] == 's') {
    type.shape = HandShape::Suited;
  } else if (text[2] == 'o') {
    type.shape = HandShape::Offsuit;
  } else {
    return std::nullopt;
  }
  return type;
}

std::string_view Trim(std::string_view text) {
  const size_t first = text.find_first_not_of(" \t");
  if (first == std::string_view::npos) {
    return {};
  }
  const size_t last = text.find_last_not_of(" \t");
  return text.substr(first, last - first + 1);
}

HistoryId AppendHistory(HistoryTree& tree,
                        const BettingState& state,
                        const BettingRules& rules,
                        const SolverConfig& config) {
  const HistoryId id{static_cast<uint32_t>(tree.nodes.size())};
  if (const auto* decision = std::get_if<DecisionState>(&state)) {
    const AbstractActions actions = SelectAbstractActions(
        config.bet_abstraction, *decision);
    const uint32_t begin = static_cast<uint32_t>(tree.children.size());
    tree.children.resize(begin + actions.size(), id);
    tree.nodes.push_back(
        {state, begin, static_cast<uint8_t>(actions.size())});
    for (size_t index = 0; index < actions.size(); ++index) {
      const auto child_state = ApplyAction(*decision, actions[index]);
      assert(child_state.ok());
      tree.children[begin + index] =
          AppendHistory(tree, *child_state, rules, config);
    }
    return id;
  }

  if (const auto* chance = std::get_if<ChanceState>(&state)) {
    const uint32_t begin = static_cast<uint32_t>(tree.children.size());
    tree.children.push_back(id);
    tree.nodes.push_back({state, begin, 1});
    const BettingState child_state = AdvanceBettingStreet(*chance, rules);
    tree.children[begin] = AppendHistory(tree, child_state, rules, config);
    return id;
  }

  tree.nodes.push_back(
      {state, static_cast<uint32_t>(tree.children.size()), 0});
  return id;
}

}  // namespace

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

void FillUniform(absl::Span<float> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0f / static_cast<float>(probabilities.size()));
  }
}

float LoadValue(const float& value, bool concurrent) {
  if (!concurrent) return value;
  float loaded;
  __atomic_load(&value, &loaded, __ATOMIC_RELAXED);
  return loaded;
}

void AtomicAdd(float& target, float delta) {
  float old = LoadValue(target, true);
  float next;
  do {
    next = old + delta;
  } while (!__atomic_compare_exchange(
      &target, &old, &next, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

}  // namespace

void CfrState::add_regret(size_t offset,
                          size_t action,
                          float delta,
                          bool concurrent) {
  const size_t index = offset + action;
  float& regret = regret_sum[index];
  if (!concurrent) {
    regret = std::max(0.0f, regret + delta);
    return;
  }
  float old = LoadValue(regret, true);
  float next;
  do {
    next = std::max(0.0f, old + delta);
  } while (!__atomic_compare_exchange(
      &regret, &old, &next, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

void CfrState::add_strategy(size_t offset,
                            absl::Span<const float> probabilities,
                            double weight,
                            bool concurrent) {
  if (!accumulate_average_strategy_) return;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = offset + action;
    float& sum = strategy_sum[index];
    const float delta = static_cast<float>(weight * probabilities[action]);
    if (concurrent) {
      AtomicAdd(sum, delta);
    } else {
      sum += delta;
    }
  }
}

void CfrState::strategy(absl::Span<const float> values,
                        std::optional<size_t> offset,
                        absl::Span<float> probabilities,
                        bool concurrent) const {
  if (!offset) {
    FillUniform(probabilities);
    return;
  }
  float sum = 0.0f;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    probabilities[action] =
        LoadValue(values[*offset + action], concurrent);
    sum += probabilities[action];
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
  } else {
    const float scale = 1.0f / sum;
    for (float& probability : probabilities) probability *= scale;
  }
}

CfrState::CfrState(const SolverConfig& config,
                   bool accumulate_average_strategy)
    : max_info_sets_(static_cast<size_t>(config.max_info_sets)),
      accumulate_average_strategy_(accumulate_average_strategy) {
  const CardAbstractionConfig& cards = config.card_abstraction;
  packed_keys_ = cards.public_mode != PublicCardMode::ExactCanonical;
  if (packed_keys_) {
    private_bits_ = cards.private_kind == PrivateAbstractionKind::ExactCanonical
                        ? 11
                        : cards.recall_mode == RecallMode::CurrentBucketOnly
                              ? 6
                              : 21;
    history_bits_ = std::min<uint8_t>(
        32, static_cast<uint8_t>(43 - private_bits_));
  }
  size_t max_actions = 3;
  for (const auto& fractions : config.bet_abstraction.pot_fractions) {
    max_actions = std::max(max_actions, fractions.size() + 3);
  }
  if (packed_keys_) {
    packed_rows_.reserve(max_info_sets_);
  } else {
    full_rows_.reserve(max_info_sets_);
  }
  regret_sum.reserve(max_info_sets_ * max_actions);
  if (accumulate_average_strategy) {
    strategy_sum.reserve(max_info_sets_ * max_actions);
  }
}

bool CfrState::can_pack(InfoSetKey key) const {
  constexpr uint64_t kPublicMask = (uint64_t{1} << 21) - 1;
  const auto mask = [](uint8_t bits) { return (uint64_t{1} << bits) - 1; };
  return std::to_underlying(key.public_observation) <= kPublicMask &&
         std::to_underlying(key.history) <= mask(history_bits_) &&
         std::to_underlying(key.private_observation) <= mask(private_bits_);
}

uint64_t CfrState::pack(InfoSetKey key) const {
  return std::to_underlying(key.private_observation) |
         uint64_t{std::to_underlying(key.history)} << private_bits_ |
         std::to_underlying(key.public_observation)
             << (private_bits_ + history_bits_);
}

InfoSetKey CfrState::unpack(uint64_t key) const {
  const auto mask = [](uint8_t bits) { return (uint64_t{1} << bits) - 1; };
  return {
      PublicObservationId(key >> (private_bits_ + history_bits_)),
      HistoryId(static_cast<uint32_t>(key >> private_bits_ &
                                      mask(history_bits_))),
      PrivateObservationId(static_cast<uint32_t>(key & mask(private_bits_)))};
}

void CfrState::use_full_keys() {
  full_rows_.reserve(max_info_sets_);
  for (const auto& [key, offset] : packed_rows_) {
    full_rows_.try_emplace(unpack(key), offset);
  }
  packed_rows_ = {};
  packed_keys_ = false;
}

size_t CfrState::row_count() const {
  return packed_keys_ ? packed_rows_.size() : full_rows_.size();
}

std::optional<size_t> CfrState::find(InfoSetKey key) const {
  if (packed_keys_) {
    if (!can_pack(key)) return std::nullopt;
    const auto found = packed_rows_.find(pack(key));
    return found == packed_rows_.end() ? std::nullopt
                                       : std::optional(found->second);
  }
  const auto found = full_rows_.find(key);
  return found == full_rows_.end() ? std::nullopt
                                    : std::optional(found->second);
}

std::vector<std::pair<InfoSetKey, size_t>> CfrState::row_entries() const {
  std::vector<std::pair<InfoSetKey, size_t>> entries;
  entries.reserve(row_count());
  if (packed_keys_) {
    for (const auto& [key, offset] : packed_rows_) {
      entries.emplace_back(unpack(key), offset);
    }
  } else {
    entries.assign(full_rows_.begin(), full_rows_.end());
  }
  std::ranges::sort(entries);
  return entries;
}

std::optional<size_t> CfrState::find_or_create(
    InfoSetKey key,
    uint8_t action_count) {
  if (packed_keys_ && !can_pack(key)) use_full_keys();
  auto insert = [&](auto& rows, auto row_key) -> std::optional<size_t> {
    if (rows.size() >= max_info_sets_) {
      const auto found = rows.find(row_key);
      return found == rows.end() ? std::nullopt
                                 : std::optional(found->second);
    }
    const size_t offset = regret_sum.size();
    const auto [row, inserted] = rows.try_emplace(row_key, offset);
    if (!inserted) return row->second;
    regret_sum.resize(offset + action_count, 0.0f);
    if (accumulate_average_strategy_) {
      strategy_sum.resize(offset + action_count, 0.0f);
    }
    return offset;
  };
  return packed_keys_ ? insert(packed_rows_, pack(key))
                      : insert(full_rows_, key);
}

namespace {

struct TabularBackend {
  using UpdateHandle = size_t;

  CfrState& state;
  bool may_create_infosets;
  bool concurrent_updates;

  std::optional<size_t> current_strategy(
      const internal::DecisionView& decision,
      internal::StrategyAccess access,
      absl::Span<float> probabilities) {
    const std::optional<size_t> offset =
        access == internal::StrategyAccess::Writable && may_create_infosets
            ? state.find_or_create(decision.key, decision.action_count)
            : state.find(decision.key);
    state.strategy(state.regret_sum, offset, probabilities,
                   concurrent_updates);
    return access == internal::StrategyAccess::ReadOnly ? std::nullopt
                                                        : offset;
  }

  void average_strategy(const internal::DecisionView& decision,
                        absl::Span<float> probabilities) {
    state.strategy(state.strategy_sum, state.find(decision.key), probabilities,
                   concurrent_updates);
  }

  void record_regrets(const internal::DecisionView&,
                      size_t offset,
                      absl::Span<const float> regrets) {
    for (size_t action = 0; action < regrets.size(); ++action) {
      state.add_regret(offset, action, regrets[action], concurrent_updates);
    }
  }

  void record_strategy(const internal::DecisionView&,
                       size_t offset,
                       absl::Span<const float> probabilities,
                       double weight) {
    state.add_strategy(offset, probabilities, weight, concurrent_updates);
  }
};

}  // namespace

bool Policy::strategy(InfoSetKey key, absl::Span<float> output) const {
  const auto found = rows.find(key);
  if (found == rows.end() ||
      found->second + output.size() > probabilities.size()) {
    if (!output.empty()) {
      std::fill(output.begin(), output.end(), 1.0f / output.size());
    }
    return false;
  }
  const size_t offset = found->second;
  std::copy_n(probabilities.data() + offset,
              output.size(), output.begin());
  return true;
}

absl::StatusOr<SolverConfig> SolverConfig::Create(SolverConfig config) {
  if (config.betting_rules.minimum_bet <= 0) {
    return absl::InvalidArgumentError("minimum bet must be positive");
  }
  if (config.chance_samples <= 0) {
    return absl::InvalidArgumentError("chance_samples must be positive");
  }
  if (config.max_info_sets <= 0) {
    return absl::InvalidArgumentError("max_info_sets must be positive");
  }
  for (auto& fractions : config.bet_abstraction.pot_fractions) {
    for (double fraction : fractions) {
      if (!std::isfinite(fraction) || fraction <= 0.0) {
        return absl::InvalidArgumentError(
            "pot fractions must be finite and positive");
      }
    }
    std::sort(fractions.begin(), fractions.end());
    fractions.erase(std::unique(fractions.begin(), fractions.end()),
                    fractions.end());
    if (fractions.size() > kMaxActionsPerNode - size_t{3}) {
      return absl::InvalidArgumentError("too many pot fractions");
    }
  }
  return config;
}

absl::StatusOr<ComboRange> ParseRange(std::string_view text) {
  ComboRange range;
  auto select = [&](HandType type) {
    constexpr std::array suits = {
        Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades};
    const Rank high = static_cast<Rank>(type.high - 2);
    const Rank low = static_cast<Rank>(type.low - 2);
    const bool pair = type.high == type.low;
    for (Suit first : suits) {
      for (Suit second : suits) {
        const bool suited = first == second;
        if ((pair && first >= second) ||
            (!pair && type.shape != HandShape::Any &&
             suited != (type.shape == HandShape::Suited))) {
          continue;
        }
        const ComboId combo = CardsToComboId(
            Card(high, first), Card(low, second));
        range.weights[combo.index()] = 1.0f;
      }
    }
  };
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view part = Trim(text.substr(0, comma));
    text = comma == std::string_view::npos ? std::string_view()
                                           : text.substr(comma + 1);
    if (part.empty()) {
      return absl::InvalidArgumentError("range contains an empty item");
    }
    if (part.size() == 3 && part[0] == part[1] && part[2] == '+') {
      const auto rank = ParseRank(part[0]);
      if (!rank) {
        return absl::InvalidArgumentError("invalid pair range");
      }
      for (int value = *rank; value <= 14; ++value) {
        select(HandType{value, value, HandShape::Any});
      }
      continue;
    }
    const auto type = ParseHandType(part);
    if (!type) {
      return absl::InvalidArgumentError("invalid hand range item");
    }
    select(*type);
  }
  if (range.count() == 0) {
    return absl::InvalidArgumentError("range is empty");
  }
  return range;
}

ComboRange UniformComboRange() {
  ComboRange range;
  range.weights.fill(1.0f);
  return range;
}

ComboRange SingleComboRange(ComboId combo, float weight) {
  ComboRange range;
  range.add(combo, weight);
  return range;
}

TabularCfrSolver::TabularCfrSolver(CompiledGame game)
    : game_(std::move(game)),
      rng_(12345),
      state_(game_.spec.config,
             game_.spec.config.accumulate_average_strategy) {}

absl::StatusOr<CompiledGame> CompileGame(SolveSpec spec) {
  auto config = SolverConfig::Create(std::move(spec.config));
  if (!config.ok()) return config.status();
  spec.config = std::move(*config);
  if (!IsValidBettingData(Data(spec.root.betting))) {
    return absl::InvalidArgumentError("invalid root betting state");
  }
  auto deals = DealDistribution::Create(spec.ranges[Index(Player::A)],
                                        spec.ranges[Index(Player::B)]);
  if (!deals.ok()) return deals.status();
  CompiledGame game{std::move(spec), std::move(*deals)};
  game.history.nodes.reserve(4096);
  game.history.children.reserve(4096);
  AppendHistory(game.history, game.spec.root.betting,
                game.spec.config.betting_rules, game.spec.config);
  game.model = FingerprintModel(game.spec);
  return game;
}

absl::StatusOr<TabularCfrSolver> TabularCfrSolver::Create(SolveSpec spec) {
  auto game = CompileGame(std::move(spec));
  if (!game.ok()) return game.status();
  return TabularCfrSolver(std::move(*game));
}

Position internal::RootPosition(const CompiledGame& game) {
  return {HistoryId{},
          PublicPosition(game.spec.config.card_abstraction,
                         game.spec.root.board)};
}

Position internal::SampleChanceChild(const CompiledGame& game,
                                     const HistoryNode& node,
                                     const PublicPosition& public_state,
                                     const Deal& deal,
                                     std::mt19937& rng) {
  const ChanceState& chance = std::get<ChanceState>(node.state);
  const auto sampled = SampleStreetCards(
      chance.data.street, public_state.board(), deal.blocked_mask(), rng);
  assert(sampled.ok());
  return {
      game.history.children[node.children_begin],
      PublicPosition(game.spec.config.card_abstraction,
                     DealCards(public_state.board(), *sampled))};
}

internal::TraversalFrame internal::InitialTraversalFrame(
    const CompiledGame& game,
    const Deal& deal,
    const Position& position) {
  internal::TraversalFrame frame;
  for (Player player : {Player::A, Player::B}) {
    frame.private_observations[Index(player)] =
        ObservePrivate(game.spec.config.card_abstraction, deal.hand(player),
                       position.public_state.board());
  }
  if (position.public_state.board().count() == kMaxBoardCards) {
    frame.showdown_comparison = static_cast<int8_t>(CompareHands(
        deal.hand(Player::A), deal.hand(Player::B),
        position.public_state.board()));
  }
  return frame;
}

void internal::AdvancePrivateObservations(
    const CompiledGame& game,
    internal::TraversalFrame& frame,
    const Deal& deal,
    const Position& child) {
  const HistoryNode& child_node = game.history.nodes[Index(child.history)];
  if (!std::holds_alternative<DecisionState>(child_node.state)) return;
  for (Player player : {Player::A, Player::B}) {
    frame.private_observations[Index(player)] =
        ObservePrivate(game.spec.config.card_abstraction, deal.hand(player),
                       child.public_state.board(),
                       frame.private_observations[Index(player)]);
  }
}

void TabularCfrSolver::run(uint64_t iterations, int threads) {
  if (iterations == 0) return;

  auto run_iteration = [&](uint64_t iteration, std::mt19937& rng,
                           SolverStats& stats, bool concurrent) {
    const Deal deal = game_.deals.sample(rng);
    internal::TraversalContext context{
        .deal = deal,
        .mode = internal::TraversalMode::Train,
        .update_player = (iteration & 1) == 0 ? Player::A : Player::B,
        .iteration = iteration,
        .external_sampling = game_.spec.config.external_sampling,
        .rng = rng,
        .stats = stats,
    };
    TabularBackend backend{state_, !state_.at_capacity(), concurrent};
    return internal::Traverse(game_, context, backend);
  };

  uint64_t serial_iterations = 0;
  // Fill the infoset map serially; workers only update it at capacity.
  while (serial_iterations < iterations &&
         (threads <= 1 || !state_.at_capacity())) {
    state_.cumulative_root_utility += run_iteration(
        state_.iterations, rng_, stats_, false);
    ++state_.iterations;
    ++serial_iterations;
  }

  const uint64_t remaining = iterations - serial_iterations;
  if (remaining > 0) {
    const size_t worker_count = std::min<size_t>(
        static_cast<size_t>(threads), static_cast<size_t>(remaining));
    struct WorkerResult {
      double utility = 0.0;
      SolverStats stats;
    };
    std::vector<WorkerResult> worker_results(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const uint64_t first_iteration = state_.iterations;
    for (size_t worker = 0; worker < worker_count; ++worker) {
      const uint32_t worker_seed = rng_();
      workers.emplace_back([&, worker, worker_seed] {
        std::seed_seq seed{worker_seed, static_cast<uint32_t>(worker)};
        std::mt19937 rng(seed);
        WorkerResult& output = worker_results[worker];
        for (uint64_t offset = worker; offset < remaining;
             offset += worker_count) {
          output.utility += run_iteration(
              first_iteration + offset, rng, output.stats, true);
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
    for (const WorkerResult& worker : worker_results) {
      state_.cumulative_root_utility += worker.utility;
      stats_.decision_visits += worker.stats.decision_visits;
      stats_.chance_samples += worker.stats.chance_samples;
      stats_.terminal_visits += worker.stats.terminal_visits;
    }
    state_.iterations += remaining;
  }
}

double TabularCfrSolver::evaluate_deal(const Deal& deal,
                                       EvaluationMode mode) {
  internal::TraversalContext context{
      .deal = deal,
      .mode = mode == EvaluationMode::Average
                  ? internal::TraversalMode::EvaluateAverage
                  : internal::TraversalMode::EvaluateCurrent,
      .update_player = Player::A,
      .iteration = 0,
      .external_sampling = false,
      .rng = rng_,
      .stats = stats_,
  };
  TabularBackend backend{state_, false, false};
  return internal::Traverse(game_, context, backend);
}

double TabularCfrSolver::evaluate_deals(int samples, EvaluationMode mode) {
  if (samples <= 0) return 0.0;
  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    value += evaluate_deal(game_.deals.sample(rng_), mode);
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
  if (!game_.spec.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  const Deal deal{{player_a, player_b}};
  return evaluate_deal(deal, EvaluationMode::Average);
}

absl::StatusOr<double> TabularCfrSolver::evaluate_average(int samples) {
  if (!game_.spec.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return evaluate_deals(samples, EvaluationMode::Average);
}

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model) {
  std::vector<std::pair<InfoSetKey, size_t>> rows = state.row_entries();

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

    const size_t output_offset = policy.probabilities.size();
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
    absl::Span<float> probabilities(
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
  if (!game_.spec.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return ExtractAveragePolicy(state_, game_.history, game_.model);
}

double TabularCfrSolver::expected_value(Player player) const {
  if (state_.iterations == 0) return 0.0;
  const double player_a_ev = state_.cumulative_root_utility / state_.iterations;
  return player == Player::A ? player_a_ev : -player_a_ev;
}

}  // namespace poker
