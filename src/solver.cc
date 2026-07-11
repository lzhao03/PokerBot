#include "src/solver.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace poker {
namespace {

constexpr int kHandTypeCount = 169;

enum class HandShape {
  kPair,
  kSuited,
  kOffsuit,
  kAny,
};

struct HandType {
  int high = 0;
  int low = 0;
  HandShape shape = HandShape::kPair;
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

int NonPairOffset(int high, int low) {
  high -= 2;
  low -= 2;
  return high * (high - 1) / 2 + low;
}

std::optional<int> HandTypeIndex(HandType type) {
  if (type.high < type.low) {
    std::swap(type.high, type.low);
  }
  if (type.low < 2 || type.high > 14) {
    return std::nullopt;
  }
  if (type.high == type.low) {
    return type.shape == HandShape::kPair
               ? std::optional<int>(type.high - 2)
               : std::nullopt;
  }
  const int offset = NonPairOffset(type.high, type.low);
  if (type.shape == HandShape::kSuited) {
    return 13 + offset;
  }
  return type.shape == HandShape::kOffsuit
             ? std::optional<int>(91 + offset)
             : std::nullopt;
}

std::optional<HandType> DecodeHandType(int index) {
  if (index < 0 || index >= kHandTypeCount) {
    return std::nullopt;
  }
  if (index < 13) {
    return HandType{index + 2, index + 2, HandShape::kPair};
  }
  const bool suited = index < 91;
  int offset = suited ? index - 13 : index - 91;
  int high = 1;
  while (offset >= high * (high + 1) / 2) {
    ++high;
  }
  const int low = offset - high * (high - 1) / 2;
  return HandType{high + 2, low + 2,
                  suited ? HandShape::kSuited : HandShape::kOffsuit};
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
                HandShape::kPair};
  if (type.high == type.low) {
    return text.size() == 2 ? std::optional<HandType>(type) : std::nullopt;
  }
  if (text.size() == 2) {
    type.shape = HandShape::kAny;
  } else if (text[2] == 's') {
    type.shape = HandShape::kSuited;
  } else if (text[2] == 'o') {
    type.shape = HandShape::kOffsuit;
  } else {
    return std::nullopt;
  }
  return type;
}

std::vector<ComboId> Expand(HandType type) {
  constexpr std::array<Suit, 4> suits = {
      Suit::kHearts, Suit::kDiamonds,
      Suit::kClubs, Suit::kSpades};
  std::vector<ComboId> combos;
  for (size_t first = 0; first < suits.size(); ++first) {
    for (size_t second = 0; second < suits.size(); ++second) {
      if (type.high == type.low && first >= second) {
        continue;
      }
      const bool suited = first == second;
      if (type.high != type.low && type.shape != HandShape::kAny &&
          suited != (type.shape == HandShape::kSuited)) {
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

ComboRange ExpandSelected(const std::vector<int>& selected) {
  ComboRange range;
  for (int index : selected) {
    const auto combos = Expand(*DecodeHandType(index));
    const float weight = 1.0f / combos.size();
    for (ComboId combo : combos) {
      range.add(combo, weight);
    }
  }
  return range;
}

ExactPublicState DefaultInitialState(const SolverConfig& config) {
  const Chips small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const Chips big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const Chips stack = config.starting_stack;
  return MakeInitialState(BettingRules{big_blind}, {stack, stack},
                          {small_blind, big_blind});
}

HistoryId AppendHistory(HistoryTree& tree,
                        const BettingState& state,
                        const BettingRules& rules,
                        const SolverConfig& config) {
  const HistoryId id(static_cast<uint32_t>(tree.nodes.size()));
  if (const auto* decision = std::get_if<DecisionState>(&state)) {
    const SolverTransitions transitions = GenerateTransitions(config,
                                                               *decision);
    assert(transitions.size() <= std::numeric_limits<uint8_t>::max());
    const uint32_t begin = static_cast<uint32_t>(tree.edges.size());
    for (const SolverTransition& transition : transitions) {
      tree.edges.push_back({transition.action, kInvalidHistoryId});
    }
    tree.nodes.push_back(
        DecisionNode{*decision, {begin, static_cast<uint8_t>(transitions.size())}});
    for (size_t action = 0; action < transitions.size(); ++action) {
      const SolverTransition& transition = transitions[action];
      const HistoryId child =
          AppendHistory(tree, transition.child, rules, config);
      tree.edges[begin + action] = {transition.action, child};
    }
    return id;
  }

  if (const auto* chance = std::get_if<ChanceState>(&state)) {
    tree.nodes.push_back(ChanceNode{*chance});
    const BettingState child_state = AdvanceBettingStreet(*chance, rules);
    const HistoryId child = AppendHistory(tree, child_state, rules, config);
    std::get<ChanceNode>(tree.nodes[id.index()]).child = child;
    return id;
  }

  if (const auto* fold = std::get_if<FoldTerminalState>(&state)) {
    tree.nodes.push_back(FoldTerminalNode{*fold});
  } else {
    tree.nodes.push_back(
        ShowdownNode{std::get<ShowdownState>(state)});
  }
  return id;
}

HistoryTree BuildHistoryTree(const BettingState& root,
                             const BettingRules& rules,
                             const SolverConfig& config) {
  HistoryTree tree;
  tree.nodes.reserve(4096);
  tree.edges.reserve(4096);
  tree.root = AppendHistory(tree, root, rules, config);
  return tree;
}

}  // namespace

absl::StatusOr<DealDistribution> DealDistribution::Create(
    const ComboRange& player_a,
    const ComboRange& player_b) {
  DealDistribution distribution;
  distribution.a_hands_.reserve(player_a.count());
  distribution.a_cumulative_.reserve(player_a.count());
  distribution.b_offsets_.reserve(player_a.count());
  distribution.b_counts_.reserve(player_a.count());
  distribution.b_hands_.reserve(player_a.count() * player_b.count());
  distribution.b_cumulative_.reserve(player_a.count() * player_b.count());

  for (uint16_t a_index = 0; a_index < player_a.active_count; ++a_index) {
    const ComboId a = player_a.active[a_index];
    const float a_weight = player_a.weight(a);
    if (a_weight <= 0.0f) {
      continue;
    }
    const uint32_t offset =
        static_cast<uint32_t>(distribution.b_hands_.size());
    float b_total = 0.0f;
    for (uint16_t b_index = 0; b_index < player_b.active_count; ++b_index) {
      const ComboId b = player_b.active[b_index];
      const float b_weight = player_b.weight(b);
      if (b_weight <= 0.0f || (ComboMask(a) & ComboMask(b)) != 0) {
        continue;
      }
      b_total += b_weight;
      distribution.b_hands_.push_back(b);
      distribution.b_cumulative_.push_back(b_total);
    }
    if (b_total <= 0.0f) {
      continue;
    }
    distribution.a_hands_.push_back(a);
    distribution.b_offsets_.push_back(offset);
    distribution.b_counts_.push_back(static_cast<uint16_t>(
        distribution.b_hands_.size() - offset));
    distribution.total_ += a_weight * b_total;
    distribution.a_cumulative_.push_back(distribution.total_);
  }
  if (distribution.total_ <= 0.0f) {
    return absl::InvalidArgumentError(
        "ranges contain no non-overlapping hands");
  }
  return distribution;
}

Deal DealDistribution::sample(std::mt19937& rng) const {
  const size_t a_index = SampleIndex(a_cumulative_, total_, rng);
  const size_t offset = b_offsets_[a_index];
  const uint16_t count = b_counts_[a_index];
  const float b_total = b_cumulative_[offset + count - 1];
  const size_t relative = SampleIndex(
      absl::Span<const float>(b_cumulative_).subspan(offset, count),
      b_total, rng);
  const ComboId a = a_hands_[a_index];
  const ComboId b = b_hands_[offset + relative];
  return {{HoleCards(a), HoleCards(b)}, ComboMask(a) | ComboMask(b)};
}

size_t DealDistribution::SampleIndex(absl::Span<const float> cumulative,
                                     float total,
                                     std::mt19937& rng) {
  std::uniform_real_distribution<float> distribution(0.0f, total);
  const auto found = std::upper_bound(
      cumulative.begin(), cumulative.end(), distribution(rng));
  return found == cumulative.end()
             ? cumulative.size() - 1
             : static_cast<size_t>(found - cumulative.begin());
}

namespace {

void FillUniform(absl::Span<double> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0 / probabilities.size());
  }
}

void RegretMatch(const CfrState& state,
                 const InfoSetRow* row,
                 absl::Span<double> probabilities) {
  if (row == nullptr) {
    FillUniform(probabilities);
    return;
  }

  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row->action_offset) + action;
    const double regret = std::max(0.0, static_cast<double>(state.regret_sum[index]));
    probabilities[action] = regret;
    sum += regret;
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
    return;
  }
  for (double& probability : probabilities) {
    probability /= sum;
  }
}

void AverageStrategy(const CfrState& state,
                     const InfoSetRow* row,
                     absl::Span<double> probabilities) {
  if (row == nullptr) {
    FillUniform(probabilities);
    return;
  }

  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row->action_offset) + action;
    const float value = state.strategy_sum[index];
    probabilities[action] = std::max(0.0, static_cast<double>(value));
    sum += probabilities[action];
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
    return;
  }
  for (double& probability : probabilities) {
    probability /= sum;
  }
}

void AddCfrPlusRegret(CfrState& state,
                      InfoSetRow row,
                      size_t action,
                      float delta) {
  const size_t index = static_cast<size_t>(row.action_offset) + action;
  state.regret_sum[index] = std::max(0.0f, state.regret_sum[index] + delta);
}

void AddStrategySum(CfrState& state,
                    InfoSetRow row,
                    absl::Span<const double> probabilities,
                    double weight) {
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row.action_offset) + action;
    state.strategy_sum[index] +=
        static_cast<float>(weight * probabilities[action]);
  }
}

}  // namespace

absl::StatusOr<ComboRange> ParseRange(std::string_view text) {
  std::array<bool, kHandTypeCount> seen = {};
  std::vector<int> selected;
  auto select = [&](HandType type) {
    auto add = [&](HandType candidate) {
      const auto index = HandTypeIndex(candidate);
      if (index && !seen[*index]) {
        seen[*index] = true;
        selected.push_back(*index);
      }
    };
    if (type.shape == HandShape::kAny) {
      type.shape = HandShape::kSuited;
      add(type);
      type.shape = HandShape::kOffsuit;
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
        select(HandType{value, value, HandShape::kPair});
      }
      continue;
    }
    const auto type = ParseHandType(part);
    if (!type) {
      return absl::InvalidArgumentError("invalid hand range item");
    }
    select(*type);
  }
  if (selected.empty()) {
    return absl::InvalidArgumentError("range is empty");
  }
  return ExpandSelected(selected);
}

absl::Status ValidateSolverConfig(const SolverConfig& config) {
  if (config.starting_stack <= 0 || config.small_blind <= 0 ||
      config.big_blind < config.small_blind ||
      config.starting_stack < config.big_blind) {
    return absl::InvalidArgumentError("invalid stack or blind configuration");
  }
  if (config.chance_samples <= 0) {
    return absl::InvalidArgumentError("chance_samples must be positive");
  }
  if (config.max_info_sets <= 0) {
    return absl::InvalidArgumentError("max_info_sets must be positive");
  }
  for (const auto& street_sizes : config.bet_sizes) {
    for (double size : street_sizes) {
      if (!std::isfinite(size) || size <= 0.0) {
        return absl::InvalidArgumentError(
            "bet sizes must be finite and positive");
      }
    }
  }
  return absl::OkStatus();
}

ComboRange UniformRange() {
  std::vector<int> selected(kHandTypeCount);
  for (int index = 0; index < kHandTypeCount; ++index) {
    selected[index] = index;
  }
  return ExpandSelected(selected);
}

ComboRange SingleComboRange(ComboId combo, float weight) {
  ComboRange range;
  range.add(combo, weight);
  return range;
}

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, DefaultInitialState(config)) {}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const ExactPublicState& initial_state)
    : config_(config),
      betting_rules_{config_.big_blind > 0 ? config_.big_blind : 2},
      initial_state_(initial_state),
      rng_(12345) {
  if (!IsValidBettingData(Data(initial_state_.betting))) {
    throw std::invalid_argument("initial betting state is invalid");
  }
  history_ = BuildHistoryTree(initial_state_.betting, betting_rules_, config_);
  if (config_.max_info_sets > 0) {
    const size_t rows = static_cast<size_t>(config_.max_info_sets);
    state_.rows.reserve(rows);
    state_.regret_sum.reserve(rows * 4);
    if (config_.accumulate_average_strategy) {
      state_.strategy_sum.reserve(rows * 4);
    }
  }
}

Position CFRSolver::root_position() const {
  return {history_.root,
          PublicPosition::Root(Data(initial_state_.betting).street,
                               initial_state_.board)};
}

Position CFRSolver::action_child(Position position, int action_index) const {
  if (position.history.index() >= history_.nodes.size()) {
    throw std::logic_error("action parent history is invalid");
  }
  const auto* node =
      std::get_if<DecisionNode>(&history_.nodes[position.history.index()]);
  if (node == nullptr || action_index < 0 ||
      action_index >= node->edges.count) {
    throw std::logic_error("action index is out of range");
  }
  position.history = history_.edges[node->edges.begin + action_index].child;
  return position;
}

Position CFRSolver::sample_chance_child(Position position,
                                        const Deal& deal) {
  if (position.history.index() >= history_.nodes.size()) {
    throw std::logic_error("chance parent history is invalid");
  }
  const auto* node =
      std::get_if<ChanceNode>(&history_.nodes[position.history.index()]);
  if (node == nullptr || node->child == kInvalidHistoryId) {
    throw std::logic_error("expected a chance history");
  }

  const BettingData& data = node->state.data;
  const auto sampled = SampleStreetCards(data.street,
                                         position.public_state.board(),
                                         deal.blocked_mask, rng_);
  assert(sampled.ok());
  const ExactPublicState child = AdvanceChance(
      node->state, position.public_state.board(), *sampled, betting_rules_);
  position.history = node->child;
  position.public_state = position.public_state.after_chance(
      Data(child.betting).street, child.board);
  return position;
}

std::array<PrivateObservationId, kPlayerCount>
CFRSolver::private_observations_for_position(
    const Deal& deal,
    const Position& position) const {
  std::array<PrivateObservationId, kPlayerCount> observations;
  for (int player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    observations[player] =
        private_observation_for_runout(hand, position.public_state);
  }
  return observations;
}

void CFRSolver::advance_private_observations(
    TraversalFrame& frame,
    const Deal& deal,
    const Position& child) const {
  for (int player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] = advance_private_observation(
        frame.private_observations[player], hand, child.public_state);
    assert(frame.private_observations[player] ==
           private_observation_for_runout(hand, child.public_state));
  }
}

const InfoSetRow* CFRSolver::find_row(InfoSetKey key,
                                     uint8_t action_count) const {
  const auto row = state_.rows.find(key);
  if (row == state_.rows.end()) {
    return nullptr;
  }
  if (row->second.action_count != action_count) {
    throw std::logic_error("infoset action count mismatch");
  }
  return &row->second;
}

std::optional<InfoSetRow> CFRSolver::find_or_create_row(
    InfoSetKey key,
    uint8_t action_count) {
  if (const InfoSetRow* row = find_row(key, action_count)) {
    return *row;
  }
  if (state_.rows.size() >= static_cast<size_t>(config_.max_info_sets)) {
    return std::nullopt;
  }

  const size_t offset = state_.regret_sum.size();
  state_.regret_sum.resize(offset + action_count, 0.0f);
  if (config_.accumulate_average_strategy) {
    state_.strategy_sum.resize(offset + action_count, 0.0f);
  }
  const InfoSetRow row{offset, action_count};
  return state_.rows.emplace(key, row).first->second;
}

double CFRSolver::traverse(Position position,
                           TraversalFrame frame,
                           TraversalContext& context) {
  const Deal& deal = context.deal;
  if (position.history.index() >= history_.nodes.size()) {
    throw std::logic_error("traversal history is invalid");
  }
  return std::visit([&](const auto& node) -> double {
    using Node = std::decay_t<decltype(node)>;
    if constexpr (std::is_same_v<Node, FoldTerminalNode>) {
      ++stats_.terminal_visits;
      return TerminalUtility(node.state, Player::kA);
    } else if constexpr (std::is_same_v<Node, ShowdownNode>) {
      ++stats_.terminal_visits;
      return TerminalUtility(
          node.state,
          std::get<RiverBoard>(position.public_state.board()),
          deal.hand(Player::kA), deal.hand(Player::kB));
    } else if constexpr (std::is_same_v<Node, ChanceNode>) {
      const int samples = std::max(1, config_.chance_samples);
      stats_.chance_samples += samples;
      double value = 0.0;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = sample_chance_child(position, deal);
        TraversalFrame child_frame = frame;
        advance_private_observations(child_frame, deal, child);
        value += traverse(child, child_frame, context);
      }
      return value / samples;
    } else {
      const Player player = node.state.actor;
      const size_t player_index = Index(player);
      const uint8_t action_count = node.edges.count;
      assert(action_count > 0);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player_index]};
      const ComboId hand = deal.hand(player).combo();
      assert(key.private_observation ==
             private_observation_for_runout(hand, position.public_state));
      const bool training = context.mode == TraversalMode::kTrain;
      const bool updates = training && context.update_player == player;
      InfoSetRow row;
      if (updates) {
        const auto created = find_or_create_row(key, action_count);
        if (created.has_value()) {
          row = *created;
        } else {
          context.info_set_limit_reached = true;
        }
      } else if (const InfoSetRow* existing = find_row(key, action_count)) {
        row = *existing;
      }
      const InfoSetRow* strategy_row =
          updates || row.action_count != 0 ? &row : nullptr;

      absl::InlinedVector<double, 8> probabilities(action_count, 0.0);
      absl::InlinedVector<double, 8> values(action_count, 0.0);
      if (context.mode == TraversalMode::kEvaluateAverage) {
        AverageStrategy(state_, strategy_row, absl::MakeSpan(probabilities));
      } else {
        RegretMatch(state_, strategy_row, absl::MakeSpan(probabilities));
      }

      double node_value = 0.0;
      for (uint8_t action = 0; action < action_count; ++action) {
        TraversalFrame child_frame = frame;
        child_frame.reach[player_index] *= probabilities[action];
        values[action] =
            traverse(action_child(position, action), child_frame, context);
        node_value += probabilities[action] * values[action];
      }
      if (!training) {
        return node_value;
      }
      ++stats_.decision_visits;
      if (!updates || row.action_count == 0) {
        return node_value;
      }

      const double sign = player == Player::kA ? 1.0 : -1.0;
      const double opponent_reach = frame.reach[Index(Opponent(player))];
      for (uint8_t action = 0; action < action_count; ++action) {
        const double regret =
            opponent_reach * sign * (values[action] - node_value);
        AddCfrPlusRegret(state_, row, action, static_cast<float>(regret));
      }
      if (config_.accumulate_average_strategy) {
        const double weight = frame.reach[player_index] *
                              static_cast<double>(context.iteration + 1);
        AddStrategySum(state_, row, absl::MakeConstSpan(probabilities), weight);
      }
      return node_value;
    }
  }, history_.nodes[position.history.index()]);
}

TrainingResult CFRSolver::run(uint64_t iterations,
                              const DealDistribution& deals) {
  TrainingResult result;
  if (iterations <= 0) {
    return result;
  }

  const Position root = root_position();
  for (uint64_t i = 0; i < iterations; ++i) {
    const Deal deal = deals.sample(rng_);
    TraversalFrame frame;
    frame.private_observations = private_observations_for_position(deal, root);
    const Player update_player =
        state_.iterations % kPlayerCount == 0 ? Player::kA : Player::kB;
    TraversalContext context{
        deal, TraversalMode::kTrain, update_player, state_.iterations};
    state_.cumulative_root_utility +=
        traverse(root, frame, context);
    ++state_.iterations;
    ++result.iterations_completed;
    if (context.info_set_limit_reached) {
      result.stop_reason = TrainingStopReason::kInfoSetLimit;
      break;
    }
  }

  log_training_summary();
  return result;
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand,
                                    StrategySource source) {
  if (source == StrategySource::kAverage &&
      !config_.accumulate_average_strategy) {
    throw std::logic_error("average strategy accumulation is disabled");
  }
  const Position root = root_position();
  const Deal deal{{HoleCards(player_a_hand), HoleCards(player_b_hand)},
                  ComboMask(player_a_hand) | ComboMask(player_b_hand)};
  TraversalFrame frame;
  frame.private_observations = private_observations_for_position(deal, root);
  const TraversalMode mode = source == StrategySource::kCurrent
                                 ? TraversalMode::kEvaluateCurrent
                                 : TraversalMode::kEvaluateAverage;
  TraversalContext context{deal, mode};
  return traverse(root, frame, context);
}

double CFRSolver::evaluate_strategy(int samples,
                                    const DealDistribution& deals,
                                    StrategySource source) {
  if (samples <= 0) {
    return 0.0;
  }
  if (source == StrategySource::kAverage &&
      !config_.accumulate_average_strategy) {
    throw std::logic_error("average strategy accumulation is disabled");
  }
  const Position root = root_position();
  const TraversalMode mode = source == StrategySource::kCurrent
                                 ? TraversalMode::kEvaluateCurrent
                                 : TraversalMode::kEvaluateAverage;

  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    const Deal deal = deals.sample(rng_);
    TraversalFrame frame;
    frame.private_observations = private_observations_for_position(deal, root);
    TraversalContext context{deal, mode};
    value += traverse(root, frame, context);
  }
  return value / samples;
}

double CFRSolver::get_expected_value(Player player) const {
  if (state_.iterations == 0) {
    return 0.0;
  }
  const double player_a_ev =
      state_.cumulative_root_utility / state_.iterations;
  return player == Player::kA ? player_a_ev : -player_a_ev;
}

void CFRSolver::log_training_summary() const {
  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << state_.iterations;
  LOG(INFO) << "Information sets: " << state_.rows.size();
  LOG(INFO) << "History nodes: " << history_.nodes.size();
  LOG(INFO) << "Player A average EV: " << get_expected_value(Player::kA);
}

}  // namespace poker
