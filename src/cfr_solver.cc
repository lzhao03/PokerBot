#include "src/cfr_solver.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "src/build_flags.h"
#include "src/card_utils.h"
#include "src/combo.h"
#include "src/game_rules.h"

namespace poker {
namespace {

SolverConfig NormalizedConfig(SolverConfig config) {
  if (config.num_training_threads > 1) {
    throw std::invalid_argument(
        "parallel training is not supported by the reduced solver");
  }
  config.num_training_threads = 1;
  return config;
}

ExactPublicState DefaultInitialState(const SolverConfig& config) {
  const Chips small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const Chips big_blind = config.big_blind > 0 ? config.big_blind : 2;
  const Chips stack = config.starting_stack_size;
  return MakeInitialState(BettingRules{big_blind}, {stack, stack},
                          {small_blind, big_blind});
}

HistoryNodeKind KindFor(const BettingState& state) {
  if (state.folded_player >= 0 ||
      (state.street == StreetKind::kRiver && IsBettingRoundOver(state))) {
    return HistoryNodeKind::kTerminal;
  }
  return IsBettingRoundOver(state) ? HistoryNodeKind::kChance
                                   : HistoryNodeKind::kDecision;
}

HistoryId AppendHistory(HistoryTree& tree,
                        const BettingState& state,
                        const BettingRules& rules,
                        const BettingAbstraction& abstraction) {
  const HistoryId id = static_cast<HistoryId>(tree.nodes.size());
  tree.nodes.push_back(HistoryNode{state, 0, 0, kInvalidHistoryId,
                                   KindFor(state)});

  const HistoryNodeKind kind = tree.nodes[id].kind;
  if (kind == HistoryNodeKind::kDecision) {
    const ActionMenu menu = abstraction.actions_for_betting_node(state);
    const uint32_t begin = static_cast<uint32_t>(tree.edges.size());
    tree.edges.resize(tree.edges.size() + menu.count);
    tree.nodes[id].action_begin = begin;
    tree.nodes[id].action_count = menu.count;

    for (uint8_t action = 0; action < menu.count; ++action) {
      const GameAction edge_action = menu.actions[action];
      const BettingState child_state = ApplyAction(state, edge_action);
      const HistoryId child =
          AppendHistory(tree, child_state, rules, abstraction);
      tree.edges[begin + action] = {edge_action, child};
    }
  } else if (kind == HistoryNodeKind::kChance) {
    const BettingState child_state = AdvanceBettingStreet(state, rules);
    const HistoryId child =
        AppendHistory(tree, child_state, rules, abstraction);
    tree.nodes[id].chance_child = child;
  }
  return id;
}

HistoryTree BuildHistoryTree(const BettingState& root,
                             const BettingRules& rules,
                             const BettingAbstraction& abstraction) {
  HistoryTree tree;
  tree.nodes.reserve(4096);
  tree.edges.reserve(4096);
  tree.root = AppendHistory(tree, root, rules, abstraction);
  return tree;
}

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

CFRSolver::CFRSolver(const SolverConfig& config)
    : CFRSolver(config, DefaultInitialState(config)) {}

CFRSolver::CFRSolver(const SolverConfig& config,
                     const ExactPublicState& initial_state)
    : config_(NormalizedConfig(config)),
      betting_rules_{config_.big_blind > 0 ? config_.big_blind : 2},
      initial_state_(initial_state),
      rng_(12345),
      betting_abstraction_(config_) {
  if constexpr (kCoarsePublicBuckets && kCoarsePrivateBuckets) {
    throw std::invalid_argument(
        "coarse public + coarse private abstraction does not provide "
        "exhaustive history-aware private observation support");
  }
  if (!IsValidBettingState(initial_state_.betting)) {
    throw std::invalid_argument("initial betting state is invalid");
  }
  history_ = BuildHistoryTree(initial_state_.betting, betting_rules_,
                              betting_abstraction_);
  if (config_.max_info_sets > 0) {
    const size_t rows = static_cast<size_t>(config_.max_info_sets);
    state_.rows.reserve(rows);
    state_.regret_sum.reserve(rows * 4);
    if (config_.accumulate_average_strategy) {
      state_.strategy_sum.reserve(rows * 4);
    }
  }
}

CFRSolver::Deal CFRSolver::traversal_deal(RangeDeal deal) const {
  const std::array<ComboId, kPlayerCount> hands = {
      deal.player_a_combo,
      deal.player_b_combo,
  };
  return {hands, ComboMask(hands[0]) | ComboMask(hands[1])};
}

Position CFRSolver::root_position() const {
  return {history_.root, initial_state_.board,
          public_observation_id(initial_state_.betting.street,
                                initial_state_.board)};
}

Position CFRSolver::action_child(Position position, int action_index) const {
  if (position.history >= history_.nodes.size()) {
    throw std::logic_error("action parent history is invalid");
  }
  const HistoryNode& node = history_.nodes[position.history];
  if (action_index < 0 || action_index >= node.action_count) {
    throw std::logic_error("action index is out of range");
  }
  position.history = history_.edges[node.action_begin + action_index].child;
  return position;
}

Position CFRSolver::sample_chance_child(Position position,
                                        const Deal& deal) {
  if (position.history >= history_.nodes.size()) {
    throw std::logic_error("chance parent history is invalid");
  }
  const HistoryNode& node = history_.nodes[position.history];
  if (node.kind != HistoryNodeKind::kChance ||
      node.chance_child == kInvalidHistoryId) {
    throw std::logic_error("expected a chance history");
  }

  const auto cards = SampleStreetCards(node.state.street, position.board,
                                       deal.blocked_mask, rng_);
  const ExactPublicState child =
      ApplyChance({node.state, position.board}, cards, betting_rules_);
  position.history = node.chance_child;
  position.board = child.board;
  position.public_observation = public_observation_after_chance(
      position.public_observation, child.betting.street, child.board);
  return position;
}

std::array<PrivateObservationId, kPlayerCount>
CFRSolver::private_observations_for_position(
    const Deal& deal,
    const Position& position) const {
  std::array<PrivateObservationId, kPlayerCount> observations;
  for (int player = 0; player < kPlayerCount; ++player) {
    observations[player] = private_observation_for_runout(
        deal.hand(player), position.board, position.public_observation);
  }
  return observations;
}

void CFRSolver::advance_private_observations(
    TraversalFrame& frame,
    const Deal& deal,
    const Position& child) const {
  const StreetKind street = history_.nodes[child.history].state.street;
  for (int player = 0; player < kPlayerCount; ++player) {
    frame.private_observations[player] = advance_private_observation(
        frame.private_observations[player], deal.hand(player), street,
        child.board, child.public_observation);
    assert(frame.private_observations[player] ==
           private_observation_for_runout(
               deal.hand(player), child.board, child.public_observation));
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

InfoSetRow CFRSolver::find_or_create_row(InfoSetKey key,
                                         uint8_t action_count) {
  if (const InfoSetRow* row = find_row(key, action_count)) {
    return *row;
  }
  if (config_.max_info_sets > 0 &&
      state_.rows.size() >= static_cast<size_t>(config_.max_info_sets)) {
    throw std::runtime_error("maximum infoset count exceeded");
  }

  const size_t offset = state_.regret_sum.size();
  if (offset > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("CFR action table is too large");
  }
  state_.regret_sum.resize(offset + action_count, 0.0f);
  if (config_.accumulate_average_strategy) {
    state_.strategy_sum.resize(offset + action_count, 0.0f);
  }
  const InfoSetRow row{static_cast<uint32_t>(offset), action_count};
  return state_.rows.emplace(key, row).first->second;
}

double CFRSolver::traverse(Position position,
                           TraversalFrame frame,
                           TraversalContext& context) {
  const Deal& deal = context.deal;
  if (position.history >= history_.nodes.size()) {
    throw std::logic_error("traversal history is invalid");
  }
  const HistoryNode& node = history_.nodes[position.history];
  if (node.kind == HistoryNodeKind::kTerminal) {
    ++stats_.terminal_visits;
    return TerminalUtility({node.state, position.board}, deal.hand(0),
                           deal.hand(1));
  }
  if (node.kind == HistoryNodeKind::kChance) {
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
  }

  const int player = node.state.player_to_act;
  if (!IsPlayer(player) || node.action_count == 0) {
    throw std::logic_error("decision history has no acting player or actions");
  }
  const InfoSetKey key{position.history, position.public_observation,
                       frame.private_observations[player]};
  assert(key.private_observation == private_observation_for_runout(
      deal.hand(player), position.board, position.public_observation));
  const bool training = context.mode == TraversalMode::kTrain;
  const bool updates = training && player == context.update_player;
  InfoSetRow row;
  if (updates) {
    row = find_or_create_row(key, node.action_count);
  } else if (const InfoSetRow* existing = find_row(key, node.action_count)) {
    row = *existing;
  }
  const InfoSetRow* strategy_row =
      updates || row.action_count != 0 ? &row : nullptr;

  std::array<double, kMaxActionsPerNode> probability_storage{};
  std::array<double, kMaxActionsPerNode> value_storage{};
  absl::Span<double> probabilities(probability_storage.data(),
                                   node.action_count);
  absl::Span<double> values(value_storage.data(), node.action_count);
  if (context.mode == TraversalMode::kEvaluateAverage) {
    AverageStrategy(state_, strategy_row, probabilities);
  } else {
    RegretMatch(state_, strategy_row, probabilities);
  }

  double node_value = 0.0;
  for (uint8_t action = 0; action < node.action_count; ++action) {
    TraversalFrame child_frame = frame;
    child_frame.reach[player] *= probabilities[action];
    values[action] =
        traverse(action_child(position, action), child_frame, context);
    node_value += probabilities[action] * values[action];
  }

  if (!training) {
    return node_value;
  }
  ++stats_.decision_visits;
  if (!updates) {
    return node_value;
  }

  const double sign = player == 0 ? 1.0 : -1.0;
  const double opponent_reach = frame.reach[Opponent(player)];
  for (uint8_t action = 0; action < node.action_count; ++action) {
    const double regret = opponent_reach * sign * (values[action] - node_value);
    AddCfrPlusRegret(state_, row, action, static_cast<float>(regret));
  }

  if (config_.accumulate_average_strategy) {
    const double weight =
        frame.reach[player] * static_cast<double>(context.iteration + 1);
    AddStrategySum(state_, row, probabilities, weight);
  }
  return node_value;
}

void CFRSolver::run(int iterations,
                    const HandRange& a_range_spec,
                    const HandRange& b_range_spec) {
  if (iterations <= 0) {
    return;
  }

  const TrainingRange a_range = BuildTrainingRange(a_range_spec);
  const TrainingRange b_range = BuildTrainingRange(b_range_spec);
  RangeSampler sampler(a_range, b_range);
  const Position root = root_position();
  for (int i = 0; i < iterations; ++i) {
    const Deal deal = traversal_deal(sampler.sample(rng_));
    TraversalFrame frame;
    frame.private_observations = private_observations_for_position(deal, root);
    const int update_player = static_cast<int>(state_.iterations % kPlayerCount);
    TraversalContext context{
        deal, TraversalMode::kTrain, update_player, state_.iterations};
    state_.cumulative_root_utility +=
        traverse(root, frame, context);
    ++state_.iterations;
  }

  log_training_summary();
}

double CFRSolver::evaluate_strategy(ComboId player_a_hand,
                                    ComboId player_b_hand,
                                    StrategySource source) {
  if (source == StrategySource::kAverage &&
      !config_.accumulate_average_strategy) {
    throw std::logic_error("average strategy accumulation is disabled");
  }
  const Position root = root_position();
  const Deal deal{{player_a_hand, player_b_hand},
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
                                    const HandRange& player_a_range,
                                    const HandRange& player_b_range,
                                    StrategySource source) {
  if (samples <= 0) {
    return 0.0;
  }
  if (source == StrategySource::kAverage &&
      !config_.accumulate_average_strategy) {
    throw std::logic_error("average strategy accumulation is disabled");
  }
  const TrainingRange a_range = BuildTrainingRange(player_a_range);
  const TrainingRange b_range = BuildTrainingRange(player_b_range);
  RangeSampler sampler(a_range, b_range);
  const Position root = root_position();
  const TraversalMode mode = source == StrategySource::kCurrent
                                 ? TraversalMode::kEvaluateCurrent
                                 : TraversalMode::kEvaluateAverage;

  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    const Deal deal = traversal_deal(sampler.sample(rng_));
    TraversalFrame frame;
    frame.private_observations = private_observations_for_position(deal, root);
    TraversalContext context{deal, mode};
    value += traverse(root, frame, context);
  }
  return value / samples;
}

double CFRSolver::get_expected_value(int player_id) const {
  if (state_.iterations == 0) {
    return 0.0;
  }
  const double player_a_ev =
      state_.cumulative_root_utility / state_.iterations;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

void CFRSolver::log_training_summary() const {
  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << state_.iterations;
  LOG(INFO) << "Information sets: " << state_.rows.size();
  LOG(INFO) << "History nodes: " << history_.nodes.size();
  LOG(INFO) << "Player A average EV: " << get_expected_value(0);
}

}  // namespace poker
