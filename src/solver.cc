#include "src/solver.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <fstream>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
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
  AppendInteger(bytes, static_cast<uint32_t>(range.combos.size()));
  for (ComboId combo : range.combos) {
    AppendInteger(bytes, static_cast<uint32_t>(combo.index()));
  }
  for (float weight : range.weights) {
    AppendInteger(bytes, std::bit_cast<uint32_t>(weight));
  }
}

ModelFingerprint FingerprintModel(const SolveSpec& spec,
                                  const HistoryTree& history) noexcept {
  std::vector<uint8_t> bytes;
  AppendInteger<uint32_t>(bytes, 2);  // Fingerprint schema.
  AppendInteger<uint32_t>(bytes, 1);  // Exact poker rules.
  AppendInteger<uint32_t>(bytes, 1);  // Card abstraction implementation.
  AppendInteger<uint32_t>(bytes, 1);  // Perfect-recall observation encoding.

  const SolverConfig& config = spec.config;
  AppendInteger(bytes, config.betting_rules.minimum_bet);
  AppendInteger(bytes, config.chance_samples);
  AppendInteger(bytes, config.max_info_sets);
  bytes.push_back(config.accumulate_average_strategy ? 1 : 0);
  bytes.push_back(std::to_underlying(config.card_abstraction.public_mode));
  bytes.push_back(std::to_underlying(config.card_abstraction.private_kind));
  bytes.push_back(std::to_underlying(config.card_abstraction.recall_mode));
  bytes.push_back(0);  // Reserved model extension.
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

  AppendInteger<uint32_t>(bytes, 0);  // Root history ID.
  AppendInteger(bytes, static_cast<uint32_t>(history.nodes.size()));
  for (const HistoryNode& node : history.nodes) {
    AddBettingState(bytes, node.state);
    AppendInteger(bytes, node.children_begin);
    bytes.push_back(node.child_count);
  }
  AppendInteger(bytes, static_cast<uint32_t>(history.children.size()));
  for (HistoryId child : history.children) {
    AppendInteger(bytes, std::to_underlying(child));
  }
  return Sha256(bytes);
}

absl::Status WriteBytes(const std::filesystem::path& path,
                        absl::Span<const uint8_t> bytes) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    return absl::UnavailableError("could not open output file");
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return absl::DataLossError("could not write output file");
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return absl::UnavailableError("could not replace output file");
  }
  return absl::OkStatus();
}

enum class HandShape {
  Pair,
  Suited,
  Offsuit,
  Any,
};

struct HandType {
  int high = 0;
  int low = 0;
  HandShape shape = HandShape::Pair;
};

std::optional<int> ParseRank(char rank) {
  switch (rank) {
    case 'A': return 14;
    case 'K': return 13;
    case 'Q': return 12;
    case 'J': return 11;
    case 'T': return 10;
    default:
      return rank >= '2' && rank <= '9'
                 ? std::optional<int>(rank - '0')
                 : std::nullopt;
  }
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
                HandShape::Pair};
  if (type.high == type.low) {
    return text.size() == 2 ? std::optional<HandType>(type) : std::nullopt;
  }
  if (text.size() == 2) {
    type.shape = HandShape::Any;
  } else if (text[2] == 's') {
    type.shape = HandShape::Suited;
  } else if (text[2] == 'o') {
    type.shape = HandShape::Offsuit;
  } else {
    return std::nullopt;
  }
  return type;
}

std::vector<ComboId> Expand(HandType type) {
  constexpr std::array<Suit, 4> suits = {
      Suit::Hearts, Suit::Diamonds,
      Suit::Clubs, Suit::Spades};
  std::vector<ComboId> combos;
  for (size_t first = 0; first < suits.size(); ++first) {
    for (size_t second = 0; second < suits.size(); ++second) {
      if (type.high == type.low && first >= second) {
        continue;
      }
      const bool suited = first == second;
      if (type.high != type.low && type.shape != HandShape::Any &&
          suited != (type.shape == HandShape::Suited)) {
        continue;
      }
      combos.push_back(CardsToComboId(
          Card(static_cast<Rank>(type.high - 2), suits[first]),
          Card(static_cast<Rank>(type.low - 2), suits[second])));
    }
  }
  return combos;
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
    for (ComboId hand : ranges[player]->combos) {
      distribution.hands_[player].push_back(hand);
      total += ranges[player]->weight(hand);
      distribution.cumulative_weights_[player].push_back(total);
    }
  }
  const bool compatible = std::ranges::any_of(
      player_a.combos, [&](ComboId a) {
        return std::ranges::any_of(player_b.combos, [&](ComboId b) {
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

void FillUniform(absl::Span<double> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0 / probabilities.size());
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
                            absl::Span<const double> probabilities,
                            double weight,
                            bool concurrent) {
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
                        absl::Span<double> probabilities,
                        bool concurrent) const {
  if (!offset) {
    FillUniform(probabilities);
    return;
  }
  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    probabilities[action] = std::max(
        0.0f, LoadValue(values[*offset + action], concurrent));
    sum += probabilities[action];
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
  } else {
    for (double& probability : probabilities) probability /= sum;
  }
}

CfrState::CfrState(const SolverConfig& config,
                   bool accumulate_average_strategy)
    : max_info_sets_(static_cast<size_t>(config.max_info_sets)),
      accumulate_average_strategy_(accumulate_average_strategy) {
  size_t max_actions = 3;
  for (const auto& fractions : config.bet_abstraction.pot_fractions) {
    max_actions = std::max(max_actions, fractions.size() + 3);
  }
  rows.reserve(max_info_sets_);
  regret_sum.reserve(max_info_sets_ * max_actions);
  if (accumulate_average_strategy) {
    strategy_sum.reserve(max_info_sets_ * max_actions);
  }
}

std::optional<size_t> CfrState::find_or_create(
    InfoSetKey key,
    uint8_t action_count) {
  if (const auto found = rows.find(key); found != rows.end()) {
    return found->second;
  }
  if (rows.size() >= max_info_sets_) return std::nullopt;
  const size_t offset = regret_sum.size();
  regret_sum.resize(offset + action_count, 0.0f);
  if (accumulate_average_strategy_) {
    strategy_sum.resize(offset + action_count, 0.0f);
  }
  return rows.emplace(key, offset).first->second;
}

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

namespace {

absl::Status ValidatePolicy(const Policy& policy) {
  std::vector<size_t> rows;
  rows.reserve(policy.rows.size());
  for (const auto& [key, offset] : policy.rows) {
    (void)key;
    rows.push_back(offset);
  }
  std::sort(rows.begin(), rows.end());
  if (rows.empty()) {
    return policy.probabilities.empty()
               ? absl::OkStatus()
               : absl::DataLossError("policy probabilities have no rows");
  }
  if (rows.front() != 0) {
    return absl::DataLossError("policy rows are not contiguous");
  }
  for (size_t index = 0; index < rows.size(); ++index) {
    const size_t begin = rows[index];
    const size_t end = index + 1 < rows.size()
                           ? rows[index + 1]
                           : policy.probabilities.size();
    if (begin >= end || end > policy.probabilities.size() ||
        end - begin > kMaxActionsPerNode) {
      return absl::DataLossError("invalid policy row span");
    }
    double sum = 0.0;
    for (size_t action = begin; action < end; ++action) {
      const float probability = policy.probabilities[action];
      if (!std::isfinite(probability) || probability < 0.0f ||
          probability > 1.0f) {
        return absl::DataLossError("invalid policy probability");
      }
      sum += probability;
    }
    if (std::abs(sum - 1.0) > 1e-5) {
      return absl::DataLossError("policy row is not normalized");
    }
  }
  return absl::OkStatus();
}

constexpr std::array<uint8_t, 8> kPolicyMagic = {
    'P', 'K', 'P', 'O', 'L', 'C', 'Y', '2'};

}  // namespace

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path) {
  const absl::Status valid = ValidatePolicy(policy);
  if (!valid.ok()) return valid;

  std::vector<std::pair<InfoSetKey, size_t>> rows(
      policy.rows.begin(), policy.rows.end());
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });

  std::vector<uint8_t> bytes;
  bytes.insert(bytes.end(), kPolicyMagic.begin(), kPolicyMagic.end());
  AppendInteger<uint32_t>(bytes, 2);
  bytes.insert(bytes.end(), policy.model.bytes.begin(),
               policy.model.bytes.end());
  AppendInteger<uint64_t>(bytes, rows.size());
  AppendInteger<uint64_t>(bytes, policy.probabilities.size());
  for (const auto& [key, offset] : rows) {
    AppendInteger(bytes, std::to_underlying(key.history));
    AppendInteger(bytes, std::to_underlying(key.public_observation));
    AppendInteger(bytes, std::to_underlying(key.private_observation));
    AppendInteger<uint64_t>(bytes, offset);
  }
  for (float probability : policy.probabilities) {
    AppendInteger(bytes, std::bit_cast<uint32_t>(probability));
  }
  return WriteBytes(path, bytes);
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
  std::array<bool, kComboCount> selected = {};
  ComboRange range;
  auto select = [&](HandType type) {
    auto add = [&](HandType candidate) {
      for (ComboId combo : Expand(candidate)) {
        if (!selected[combo.index()]) {
          selected[combo.index()] = true;
          range.add(combo);
        }
      }
    };
    if (type.shape == HandShape::Any) {
      type.shape = HandShape::Suited;
      add(type);
      type.shape = HandShape::Offsuit;
    }
    add(type);
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
        select(HandType{value, value, HandShape::Pair});
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
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      range.add(CardsToComboId(kDeck[first], kDeck[second]));
    }
  }
  return range;
}

ComboRange SingleComboRange(ComboId combo, float weight) {
  ComboRange range;
  range.add(combo, weight);
  return range;
}

CFRSolver::CFRSolver(SolveSpec spec, DealDistribution deals)
    : spec_(std::move(spec)),
      deals_(std::move(deals)),
      rng_(12345),
      state_(spec_.config,
             spec_.config.accumulate_average_strategy) {
  history_.nodes.reserve(4096);
  history_.children.reserve(4096);
  AppendHistory(history_, spec_.root.betting,
                spec_.config.betting_rules, spec_.config);
  model_ = FingerprintModel(spec_, history_);
}

absl::StatusOr<CFRSolver> CFRSolver::Create(SolveSpec spec) {
  auto config = SolverConfig::Create(std::move(spec.config));
  if (!config.ok()) return config.status();
  spec.config = std::move(*config);
  if (!IsValidBettingData(Data(spec.root.betting))) {
    return absl::InvalidArgumentError("invalid root betting state");
  }
  auto deals = DealDistribution::Create(spec.ranges[Index(Player::A)],
                                        spec.ranges[Index(Player::B)]);
  if (!deals.ok()) {
    return deals.status();
  }
  return CFRSolver(std::move(spec), std::move(*deals));
}

Position CFRSolver::root_position() const {
  return {HistoryId{},
          PublicPosition(card_abstraction(), spec_.root.board)};
}

Position CFRSolver::sample_chance_child(const HistoryNode& node,
                                        const PublicPosition& public_state,
                                        const Deal& deal,
                                        std::mt19937& rng) {
  const ChanceState& chance = std::get<ChanceState>(node.state);
  const BettingData& data = chance.data;
  const auto sampled = SampleStreetCards(data.street,
                                         public_state.board(),
                                         deal.blocked_mask(), rng);
  assert(sampled.ok());
  return {
      history_.children[node.children_begin],
      PublicPosition(card_abstraction(),
                     DealCards(public_state.board(), *sampled))};
}

CFRSolver::TraversalFrame CFRSolver::initial_frame(
    const Deal& deal,
    const Position& position) const {
  TraversalFrame frame;
  for (Player player : {Player::A, Player::B}) {
    const ComboId hand = deal.hand(player);
    frame.private_observations[Index(player)] =
        ObservePrivate(card_abstraction(), hand,
                       position.public_state.board());
  }
  if (position.public_state.board().count() == kMaxBoardCards) {
    frame.showdown_comparison = static_cast<int8_t>(CompareHands(
        deal.hand(Player::A), deal.hand(Player::B),
        position.public_state.board()));
  }
  return frame;
}

void CFRSolver::advance_private_observations(
    TraversalFrame& frame,
    const Deal& deal,
    const Position& child) const {
  for (Player player : {Player::A, Player::B}) {
    const ComboId hand = deal.hand(player);
    frame.private_observations[Index(player)] =
        ObservePrivate(card_abstraction(), hand, child.public_state.board());
  }
}

double CFRSolver::traverse(HistoryId history,
                           const PublicPosition& public_state,
                           const TraversalFrame& frame,
                           TraversalContext& context) {
  const Deal& deal = context.deal;
  const HistoryNode& history_node = history_.nodes[Index(history)];
  return std::visit([&](const auto& state) -> double {
    using State = std::decay_t<decltype(state)>;
    if constexpr (std::is_same_v<State, FoldTerminalState>) {
      ++context.stats.terminal_visits;
      return TerminalUtility(state, Player::A);
    } else if constexpr (std::is_same_v<State, ShowdownState>) {
      ++context.stats.terminal_visits;
      assert(frame.showdown_comparison.has_value());
      return TerminalUtilityFromComparison(
          state, *frame.showdown_comparison, Player::A);
    } else if constexpr (std::is_same_v<State, ChanceState>) {
      const int samples = spec_.config.chance_samples;
      context.stats.chance_samples += static_cast<uint64_t>(samples);
      double value = 0.0;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = sample_chance_child(
            history_node, public_state, deal, context.rng);
        TraversalFrame child_frame = frame;
        if (child.public_state.board().count() == kMaxBoardCards) {
          child_frame.showdown_comparison = static_cast<int8_t>(CompareHands(
              deal.hand(Player::A), deal.hand(Player::B),
              child.public_state.board()));
        }
        advance_private_observations(child_frame, deal, child);
        value += traverse(
            child.history, child.public_state, child_frame, context);
      }
      return value / samples;
    } else {
      const Player player = state.actor;
      const size_t player_index = Index(player);
      const uint8_t action_count = history_node.child_count;
      const InfoSetKey key{
          history, public_state.observation(),
          frame.private_observations[player_index]};
      const bool training = context.mode == TraversalMode::Train;
      const bool updates =
          training && context.iteration % kPlayerCount == player_index;
      std::optional<size_t> offset;
      if (updates) {
        offset = state_.find_or_create(key, action_count);
      } else {
        const auto found = state_.rows.find(key);
        if (found != state_.rows.end()) offset = found->second;
      }
      std::array<double, kMaxActionsPerNode> probabilities;
      std::array<double, kMaxActionsPerNode> values;
      const absl::Span<double> probability_span(
          probabilities.data(), action_count);
      if (context.mode == TraversalMode::EvaluateAverage) {
        state_.strategy(state_.strategy_sum, offset, probability_span,
                        context.concurrent_updates);
      } else {
        state_.strategy(state_.regret_sum, offset, probability_span,
                        context.concurrent_updates);
      }

      double node_value = 0.0;
      for (uint8_t action = 0; action < action_count; ++action) {
        TraversalFrame child_frame = frame;
        child_frame.reach[player_index] *= probabilities[action];
        const HistoryId child = history_.children[
            history_node.children_begin + action];
        values[action] = traverse(
            child, public_state, child_frame, context);
        node_value += probabilities[action] * values[action];
      }
      if (!training) {
        return node_value;
      }
      ++context.stats.decision_visits;
      if (!updates || !offset.has_value()) {
        return node_value;
      }

      const double sign = player == Player::A ? 1.0 : -1.0;
      const double opponent_reach = frame.reach[Index(Opponent(player))];
      for (uint8_t action = 0; action < action_count; ++action) {
        const double regret =
            opponent_reach * sign * (values[action] - node_value);
        state_.add_regret(*offset, action, static_cast<float>(regret),
                          context.concurrent_updates);
      }
      if (spec_.config.accumulate_average_strategy) {
        const double weight = frame.reach[player_index] *
                              (context.iteration + 1);
        state_.add_strategy(
            *offset,
            absl::Span<const double>(probabilities.data(), action_count),
            weight, context.concurrent_updates);
      }
      return node_value;
    }
  }, history_node.state);
}

void CFRSolver::run(uint64_t iterations, int threads) {
  if (iterations == 0) return;

  const Position root = root_position();
  auto run_iteration = [&](uint64_t iteration, std::mt19937& rng,
                           SolverStats& stats, bool concurrent) {
    const Deal deal = deals_.sample(rng);
    const TraversalFrame frame = initial_frame(deal, root);
    TraversalContext context{
        deal, TraversalMode::Train, iteration, rng, stats, concurrent};
    return traverse(root.history, root.public_state, frame, context);
  };

  const size_t capacity = static_cast<size_t>(spec_.config.max_info_sets);
  uint64_t serial_iterations = 0;
  while (serial_iterations < iterations &&
         (threads <= 1 || state_.rows.size() < capacity)) {
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
    std::vector<uint32_t> seeds(worker_count);
    for (uint32_t& seed : seeds) seed = rng_();
    std::vector<WorkerResult> worker_results(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const uint64_t first_iteration = state_.iterations;
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        std::seed_seq seed{seeds[worker], static_cast<uint32_t>(worker)};
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

double CFRSolver::evaluate_deal(const Deal& deal, TraversalMode mode) {
  const Position root = root_position();
  const TraversalFrame frame = initial_frame(deal, root);
  TraversalContext context{deal, mode, 0, rng_, stats_, false};
  return traverse(root.history, root.public_state, frame, context);
}

double CFRSolver::evaluate_deals(int samples, TraversalMode mode) {
  if (samples <= 0) {
    return 0.0;
  }
  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    value += evaluate_deal(deals_.sample(rng_), mode);
  }
  return value / samples;
}

double CFRSolver::evaluate_current(ComboId player_a, ComboId player_b) {
  const Deal deal{{player_a, player_b}};
  return evaluate_deal(deal, TraversalMode::EvaluateCurrent);
}

double CFRSolver::evaluate_current(int samples) {
  return evaluate_deals(samples, TraversalMode::EvaluateCurrent);
}

absl::StatusOr<double> CFRSolver::evaluate_average(
    ComboId player_a,
    ComboId player_b) {
  if (!spec_.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  const Deal deal{{player_a, player_b}};
  return evaluate_deal(deal, TraversalMode::EvaluateAverage);
}

absl::StatusOr<double> CFRSolver::evaluate_average(int samples) {
  if (!spec_.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return evaluate_deals(samples, TraversalMode::EvaluateAverage);
}

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model) {
  std::vector<std::pair<InfoSetKey, size_t>> rows(
      state.rows.begin(), state.rows.end());
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });

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

    double mass = 0.0;
    for (size_t action = 0; action < node.child_count; ++action) {
      const float value = state.strategy_sum[offset + action];
      if (!std::isfinite(value)) {
        return absl::DataLossError("nonfinite average strategy value");
      }
      mass += std::max(0.0f, value);
    }

    const size_t output_offset = policy.probabilities.size();
    policy.rows.emplace(key, output_offset);
    for (size_t action = 0; action < node.child_count; ++action) {
      const float value = state.strategy_sum[offset + action];
      policy.probabilities.push_back(
          mass > 0.0 ? static_cast<float>(std::max(0.0f, value) / mass)
                     : 1.0f / node.child_count);
    }
  }
  return policy;
}

absl::StatusOr<Policy> CFRSolver::extract_average_policy() const {
  if (!spec_.config.accumulate_average_strategy) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return ExtractAveragePolicy(state_, history_, model_);
}

double CFRSolver::get_expected_value(Player player) const {
  if (state_.iterations == 0) {
    return 0.0;
  }
  const double player_a_ev =
      state_.cumulative_root_utility / state_.iterations;
  return player == Player::A ? player_a_ev : -player_a_ev;
}

}  // namespace poker
