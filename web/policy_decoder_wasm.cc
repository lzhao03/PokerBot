#include "web/policy_decoder_wasm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "src/bet_abstraction.h"
#include "src/card_abstraction.h"
#include "src/hand_evaluator.h"
#include "src/history.h"
#include "src/neural_features.h"
#include "src/poker.h"

#ifdef __EMSCRIPTEN__
#include "emscripten/emscripten.h"
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

constexpr std::array<uint8_t, 8> kPolicyMagic = {
    'P', 'K', 'C', 'O', 'D', 'E', 'C', '1'};
constexpr uint32_t kNeuralMagic = 0x314e4e50;  // PNN1
constexpr uint16_t kMaxUnits = 256;
constexpr size_t kMaxLoggedActions = 64;

uint64_t Binomial(size_t n, size_t k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  uint64_t result = 1;
  for (size_t i = 1; i <= k; ++i) {
    result = result * static_cast<uint64_t>(n - k + i) / i;
  }
  return result;
}

uint64_t DistributionCount(size_t units, size_t actions) {
  return Binomial(units + actions - 1, actions - 1);
}

class Reader {
 public:
  Reader(const uint8_t* bytes, size_t size)
      : current_(bytes), end_(bytes + size) {}

  template <typename Integer>
  bool integer(Integer& value) {
    if (remaining() < sizeof(Integer)) return false;
    value = 0;
    for (size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Integer>(current_[index]) << (index * 8);
    }
    current_ += sizeof(Integer);
    return true;
  }

  bool varint(uint64_t& value) {
    value = 0;
    for (size_t shift = 0; shift < 64; shift += 7) {
      uint8_t byte;
      if (!integer(byte) || (shift == 63 && byte > 1)) return false;
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return true;
    }
    return false;
  }

  bool floating(float& value) {
    uint32_t bits;
    if (!integer(bits)) return false;
    value = std::bit_cast<float>(bits);
    return true;
  }

  size_t remaining() const {
    return static_cast<size_t>(end_ - current_);
  }

 private:
  const uint8_t* current_;
  const uint8_t* end_;
};

struct Key {
  uint64_t public_observation;
  uint32_t history;
  uint32_t private_observation;

  friend bool operator==(const Key&, const Key&) = default;
};

auto Order(const Key& key) {
  return std::tuple(key.history, key.public_observation,
                    key.private_observation);
}

struct Row {
  Key key;
  uint64_t code;
};

struct CompactPolicy {
  uint64_t model = 0;
  uint16_t units = 0;
  uint8_t max_actions = 0;
  std::array<std::vector<Row>, poker::kMaxActionsPerNode + 1> rows;
};

bool DecodePolicy(const uint8_t* bytes,
                  size_t size,
                  CompactPolicy& output) {
  Reader reader(bytes, size);
  for (uint8_t expected : kPolicyMagic) {
    uint8_t actual;
    if (!reader.integer(actual) || actual != expected) return false;
  }

  CompactPolicy decoded;
  if (!reader.integer(decoded.model) || !reader.integer(decoded.units) ||
      !reader.integer(decoded.max_actions) || decoded.units == 0 ||
      decoded.units > kMaxUnits || decoded.max_actions == 0 ||
      decoded.max_actions > poker::kMaxActionsPerNode) {
    return false;
  }

  for (size_t actions = 2; actions <= decoded.max_actions; ++actions) {
    uint32_t row_count;
    if (!reader.integer(row_count) || row_count > reader.remaining() / 4) {
      return false;
    }
    std::vector<Row>& rows = decoded.rows[actions];
    rows.reserve(row_count);
    uint64_t history = 0;
    uint64_t public_observation = 0;
    uint64_t private_observation = 0;
    for (uint32_t index = 0; index < row_count; ++index) {
      uint64_t history_delta;
      uint64_t public_delta;
      uint64_t private_delta;
      uint64_t code;
      if (!reader.varint(history_delta) || !reader.varint(public_delta) ||
          !reader.varint(private_delta) || !reader.varint(code) ||
          history_delta > std::numeric_limits<uint32_t>::max() - history) {
        return false;
      }
      history += history_delta;
      if (history_delta != 0) public_observation = private_observation = 0;
      if (public_delta >
          std::numeric_limits<uint64_t>::max() - public_observation) {
        return false;
      }
      public_observation += public_delta;
      if (public_delta != 0) private_observation = 0;
      if (private_delta >
              std::numeric_limits<uint32_t>::max() - private_observation ||
          (index > 0 && history_delta == 0 && public_delta == 0 &&
           private_delta == 0) ||
          code >= DistributionCount(decoded.units, actions)) {
        return false;
      }
      private_observation += private_delta;
      rows.push_back({
          {public_observation, static_cast<uint32_t>(history),
           static_cast<uint32_t>(private_observation)},
          code});
    }
  }
  if (reader.remaining() != 0) return false;
  output = std::move(decoded);
  return true;
}

void DecodeProbabilities(uint64_t code,
                         size_t actions,
                         uint16_t units,
                         float* output) {
  size_t remaining = units;
  for (size_t action = 0; action + 1 < actions; ++action) {
    const size_t remaining_actions = actions - action - 1;
    size_t action_units = 0;
    while (action_units < remaining) {
      const uint64_t skipped =
          DistributionCount(remaining - action_units, remaining_actions);
      if (code < skipped) break;
      code -= skipped;
      ++action_units;
    }
    output[action] = static_cast<float>(action_units) / units;
    remaining -= action_units;
  }
  output[actions - 1] = static_cast<float>(remaining) / units;
}

bool TabularStrategy(const CompactPolicy& policy,
                     Key key,
                     size_t action_count,
                     float* output) {
  const std::vector<Row>& rows = policy.rows[action_count];
  const auto found = std::lower_bound(
      rows.begin(), rows.end(), key,
      [](const Row& row, const Key& target) {
        return Order(row.key) < Order(target);
      });
  if (found == rows.end() || found->key != key) {
    std::fill_n(output, action_count, 1.0f / action_count);
    return false;
  }
  DecodeProbabilities(found->code, action_count, policy.units, output);
  return true;
}

struct PortableNeuralPolicy {
  uint64_t model = 0;
  size_t hidden_size = 0;
  std::vector<float> parameters;
  std::array<std::vector<float>, 2> hidden;
  poker::NeuralActionVector logits = {};
};

bool DecodeNeuralPolicy(const uint8_t* bytes,
                        size_t size,
                        PortableNeuralPolicy& output) {
  Reader reader(bytes, size);
  uint32_t magic;
  uint32_t version;
  uint32_t feature_schema;
  uint32_t feature_count;
  uint32_t hidden_size;
  uint32_t max_actions;
  PortableNeuralPolicy decoded;
  if (!reader.integer(magic) || !reader.integer(version) ||
      !reader.integer(feature_schema) || !reader.integer(feature_count) ||
      !reader.integer(hidden_size) || !reader.integer(max_actions) ||
      !reader.integer(decoded.model) || magic != kNeuralMagic || version != 1 ||
      feature_schema != poker::kNeuralFeatureSchemaVersion ||
      feature_count != poker::kNeuralFeatureCount || hidden_size == 0 ||
      hidden_size > 4096 || max_actions != poker::kMaxActionsPerNode ||
      decoded.model != poker::web::kNeuralModel) {
    return false;
  }
  const uint64_t hidden = hidden_size;
  const uint64_t parameter_count =
      poker::kNeuralFeatureCount * hidden + hidden +
      2 * (hidden * hidden + hidden) +
      poker::kMaxActionsPerNode * hidden + poker::kMaxActionsPerNode;
  if (parameter_count != reader.remaining() / sizeof(float) ||
      reader.remaining() % sizeof(float) != 0) {
    return false;
  }
  decoded.hidden_size = hidden_size;
  decoded.parameters.resize(static_cast<size_t>(parameter_count));
  for (float& parameter : decoded.parameters) {
    if (!reader.floating(parameter)) return false;
  }
  for (auto& hidden_values : decoded.hidden) {
    hidden_values.resize(hidden_size);
  }
  output = std::move(decoded);
  return true;
}

void Linear(const float*& parameters,
            std::span<const float> input,
            std::span<float> output,
            bool relu) {
  const float* weights = parameters;
  parameters += input.size() * output.size();
  const float* bias = parameters;
  parameters += output.size();
  for (size_t row = 0; row < output.size(); ++row) {
    float value = bias[row];
    for (size_t column = 0; column < input.size(); ++column) {
      value += weights[row * input.size() + column] * input[column];
    }
    output[row] = relu ? std::max(0.0f, value) : value;
  }
}

void NeuralStrategy(PortableNeuralPolicy& policy,
                    const poker::NeuralFeatureVector& features,
                    size_t action_count,
                    float* output) {
  const float* parameters = policy.parameters.data();
  Linear(parameters, features, policy.hidden[0], true);
  Linear(parameters, policy.hidden[0], policy.hidden[1], true);
  Linear(parameters, policy.hidden[1], policy.hidden[0], true);
  Linear(parameters, policy.hidden[0], policy.logits, false);

  const float maximum = *std::max_element(
      policy.logits.begin(), policy.logits.begin() + action_count);
  float sum = 0.0f;
  for (size_t action = 0; action < action_count; ++action) {
    output[action] = std::exp(policy.logits[action] - maximum);
    sum += output[action];
  }
  if (!std::isfinite(sum) || sum <= 0.0f) {
    std::fill_n(output, action_count, 1.0f / action_count);
  } else {
    for (size_t action = 0; action < action_count; ++action) {
      output[action] /= sum;
    }
  }
}

struct BrowserGame {
  BrowserGame()
      : abstraction(poker::SmallBettingConfig()),
        root(poker::MakeInitialState(rules, {200, 200}, {1, 2}).betting),
        history(poker::BuildHistoryTree(root, rules, abstraction)) {}

  poker::BettingRules rules{2};
  poker::BetAbstractionConfig abstraction;
  poker::BettingState root;
  poker::HistoryTree history;
};

const BrowserGame& Game() {
  static const BrowserGame game;
  return game;
}

bool IsAggressive(poker::ActionKind kind) {
  return kind == poker::ActionKind::Bet || kind == poker::ActionKind::Raise ||
         kind == poker::ActionKind::AllIn;
}

std::optional<poker::StreetKind> StreetForBoardCount(size_t board_count) {
  switch (board_count) {
    case 0:
      return poker::StreetKind::Preflop;
    case 3:
      return poker::StreetKind::Flop;
    case 4:
      return poker::StreetKind::Turn;
    case 5:
      return poker::StreetKind::River;
    default:
      return std::nullopt;
  }
}

bool ReplayHistory(std::span<const uint8_t> input_kinds,
                   std::span<const int32_t> input_targets,
                   poker::StreetKind board_street,
                   poker::HistoryId& history) {
  const BrowserGame& game = Game();
  const auto advance_chance = [&] {
    while (std::holds_alternative<poker::ChanceState>(
        game.history.nodes[poker::Index(history)].state)) {
      const poker::HistoryNode& node =
          game.history.nodes[poker::Index(history)];
      if (poker::Data(node.state).street == board_street) return true;
      if (std::to_underlying(poker::Data(node.state).street) >
          std::to_underlying(board_street)) {
        return false;
      }
      if (node.child_count != 1) return false;
      history = game.history.children[node.children_begin];
    }
    return true;
  };
  for (size_t input = 0; input < input_kinds.size(); ++input) {
    if (!advance_chance()) return false;
    const poker::HistoryNode& node =
        game.history.nodes[poker::Index(history)];
    const auto* decision = std::get_if<poker::DecisionState>(&node.state);
    if (decision == nullptr) return false;
    const poker::AbstractActions actions = poker::SelectAbstractActions(
        game.abstraction, *decision);
    size_t selected = actions.size();
    for (size_t action = 0; action < actions.size(); ++action) {
      const bool raise =
          input_kinds[input] == std::to_underlying(poker::ActionKind::Raise);
      if ((raise && IsAggressive(actions[action].kind) &&
           actions[action].target_street_commitment == input_targets[input]) ||
          (!raise && std::to_underlying(actions[action].kind) ==
                         input_kinds[input])) {
        selected = action;
        break;
      }
    }
    if (selected == actions.size()) return false;
    history = game.history.children[node.children_begin + selected];
  }
  if (!advance_chance()) return false;
  return poker::Data(game.history.nodes[poker::Index(history)].state).street ==
         board_street;
}

bool DecodeBrowserCards(const uint8_t* cards,
                        size_t board_count,
                        std::array<poker::ComboId, 2>& hands,
                        poker::Board& board) {
  if (cards == nullptr || !StreetForBoardCount(board_count)) return false;
  std::array<poker::Card, 4 + poker::kMaxBoardCards> decoded;
  poker::CardMask seen = 0;
  for (size_t index = 0; index < board_count + 4; ++index) {
    if (cards[index] >= poker::kDeck.size()) return false;
    decoded[index] = poker::kDeck[cards[index]];
    const poker::CardMask bit = poker::CardBit(decoded[index]);
    if ((seen & bit) != 0) return false;
    seen |= bit;
  }
  const auto first = poker::MaybeCardsToComboId(decoded[0], decoded[1]);
  const auto second = poker::MaybeCardsToComboId(decoded[2], decoded[3]);
  const auto decoded_board = poker::MakeBoard(
      std::span<const poker::Card>(decoded.data() + 4, board_count));
  if (!first || !second || !decoded_board.ok()) return false;
  hands = {*first, *second};
  board = *decoded_board;
  return true;
}

size_t Seat(poker::Player player, int dealer) {
  return player == poker::Player::A ? static_cast<size_t>(dealer)
                                    : static_cast<size_t>(1 - dealer);
}

void WriteBrowserState(const poker::BettingState& state,
                       const std::array<poker::ComboId, 2>& seat_hands,
                       const poker::Board& board,
                       int dealer,
                       int32_t* output) {
  using namespace poker::web;
  const poker::BettingData& data = poker::Data(state);
  std::array<poker::Chips, 2> stacks = {};
  std::array<poker::Chips, 2> bets = {};
  for (poker::Player player : {poker::Player::A, poker::Player::B}) {
    const size_t model = poker::Index(player);
    const size_t seat = Seat(player, dealer);
    stacks[seat] = data.stack[model];
    bets[seat] = data.street_committed[model];
  }

  int32_t phase = DecisionPhase;
  int32_t actor = -1;
  int32_t winner_mask = 0;
  int32_t cards_needed = 0;
  if (const auto* decision = std::get_if<poker::DecisionState>(&state)) {
    actor = static_cast<int32_t>(Seat(decision->actor, dealer));
  } else if (std::holds_alternative<poker::ChanceState>(state)) {
    phase = ChancePhase;
    cards_needed = static_cast<int32_t>(
        poker::CardsForNextStreet(data.street));
  } else {
    const poker::Chips pot = poker::Pot(data);
    if (const auto* fold = std::get_if<poker::FoldTerminalState>(&state)) {
      phase = FoldPhase;
      const size_t winner = Seat(poker::Opponent(fold->folded), dealer);
      stacks[winner] += pot;
      winner_mask = 1 << winner;
    } else {
      phase = ShowdownPhase;
      const poker::ComboId player_a = seat_hands[Seat(poker::Player::A, dealer)];
      const poker::ComboId player_b = seat_hands[Seat(poker::Player::B, dealer)];
      const int comparison = poker::CompareHands(player_a, player_b, board);
      if (comparison == 0) {
        stacks[0] += pot / 2;
        stacks[1] += pot - pot / 2;
        winner_mask = 3;
      } else {
        const poker::Player winner =
            comparison > 0 ? poker::Player::A : poker::Player::B;
        const size_t seat = Seat(winner, dealer);
        stacks[seat] += pot;
        winner_mask = 1 << seat;
      }
    }
  }

  output[Phase] = phase;
  output[Street] = std::to_underlying(data.street);
  output[Actor] = actor;
  output[Stack0] = stacks[0];
  output[Stack1] = stacks[1];
  output[Bet0] = bets[0];
  output[Bet1] = bets[1];
  output[Pot] = winner_mask == 0 ? poker::Pot(data) : 0;
  output[CurrentWager] = poker::CurrentWager(data);
  output[CallAmount] =
      actor >= 0 ? poker::ToCall(data, actor == dealer ? poker::Player::A
                                                       : poker::Player::B)
                 : 0;
  output[WinnerMask] = winner_mask;
  output[CardsNeeded] = cards_needed;
}

Key QueryKey(poker::ComboId hand,
             const poker::Board& board,
             poker::HistoryId history,
             bool retain_private_history) {
  poker::CardAbstractionConfig config{
      poker::PublicCardMode::CompactTexture,
      poker::PrivateAbstractionKind::Handcrafted36,
      retain_private_history ? poker::RecallMode::BucketHistory
                             : poker::RecallMode::CurrentBucketOnly};
  const poker::PublicPosition position(config, board);
  return {std::to_underlying(position.observation()),
          std::to_underlying(history),
          std::to_underlying(poker::ObservePrivate(hand, position))};
}

CompactPolicy tabular_policy;
PortableNeuralPolicy neural_policy;
int query_found = 0;

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE uint8_t* poker_allocate(size_t size) {
  return static_cast<uint8_t*>(std::malloc(size));
}

EMSCRIPTEN_KEEPALIVE void poker_free(void* memory) {
  std::free(memory);
}

EMSCRIPTEN_KEEPALIVE int poker_load_policy(const uint8_t* bytes, size_t size) {
  CompactPolicy decoded;
  if (bytes == nullptr || !DecodePolicy(bytes, size, decoded) ||
      decoded.model != poker::web::kTabularModel) {
    return 0;
  }
  tabular_policy = std::move(decoded);
  return 1;
}

EMSCRIPTEN_KEEPALIVE int poker_load_neural_policy(const uint8_t* bytes,
                                                  size_t size) {
  PortableNeuralPolicy decoded;
  if (bytes == nullptr || !DecodeNeuralPolicy(bytes, size, decoded)) return 0;
  neural_policy = std::move(decoded);
  return 1;
}

// Returns the number of legal abstract actions, or -1 for invalid input.
EMSCRIPTEN_KEEPALIVE int poker_query(
    int policy_kind,
    int dealer,
    const uint8_t* input_kinds,
    const int32_t* input_targets,
    size_t input_count,
    const uint8_t* cards,
    size_t board_count,
    uint8_t* output_kinds,
    int32_t* output_targets,
    float* output_probabilities) {
  query_found = 0;
  if ((input_count > 0 && (input_kinds == nullptr || input_targets == nullptr)) ||
      input_count > kMaxLoggedActions || output_kinds == nullptr ||
      output_targets == nullptr || output_probabilities == nullptr ||
      (dealer != 0 && dealer != 1) ||
      policy_kind < poker::web::kTabularPolicy ||
      policy_kind > poker::web::kNeuralPolicy) {
    return -1;
  }
  std::array<poker::ComboId, 2> hands;
  poker::Board board;
  if (!DecodeBrowserCards(cards, board_count, hands, board)) return -1;

  poker::HistoryId history{};
  if (!ReplayHistory({input_kinds, input_count},
                     {input_targets, input_count}, board.street(), history) ||
      !std::holds_alternative<poker::DecisionState>(
          Game().history.nodes[poker::Index(history)].state)) {
    return -1;
  }
  const poker::HistoryNode& node = Game().history.nodes[poker::Index(history)];
  const poker::AbstractActions actions = poker::SelectAbstractActions(
      Game().abstraction, std::get<poker::DecisionState>(node.state));
  for (size_t action = 0; action < actions.size(); ++action) {
    output_kinds[action] = std::to_underlying(actions[action].kind);
    output_targets[action] = actions[action].target_street_commitment;
  }
  const auto& decision = std::get<poker::DecisionState>(node.state);
  const Key key = QueryKey(
      hands[Seat(decision.actor, dealer)], board, history,
      policy_kind == poker::web::kNeuralPolicy);
  if (policy_kind == poker::web::kTabularPolicy) {
    if (tabular_policy.model != poker::web::kTabularModel) return -1;
    query_found = TabularStrategy(
        tabular_policy,
        key,
        actions.size(), output_probabilities);
  } else {
    if (neural_policy.model != poker::web::kNeuralModel) return -1;
    NeuralStrategy(
        neural_policy,
        poker::EncodeNeuralFeatures(
            history, poker::PublicObservationId{key.public_observation},
            poker::PrivateObservationId{key.private_observation},
            node,
            {poker::PublicCardMode::CompactTexture,
             poker::PrivateAbstractionKind::Handcrafted36,
             poker::RecallMode::BucketHistory}),
        actions.size(), output_probabilities);
    query_found = 1;
  }
  return static_cast<int>(actions.size());
}

EMSCRIPTEN_KEEPALIVE int poker_query_found() {
  return query_found;
}

// Returns the number of legal abstract actions, or -1 for invalid input.
EMSCRIPTEN_KEEPALIVE int poker_replay(
    int dealer,
    const uint8_t* input_kinds,
    const int32_t* input_targets,
    size_t input_count,
    const uint8_t* cards,
    size_t board_count,
    int32_t* output_state,
    uint8_t* output_kinds,
    int32_t* output_targets) {
  if ((input_count > 0 && (input_kinds == nullptr || input_targets == nullptr)) ||
      input_count > kMaxLoggedActions || (dealer != 0 && dealer != 1) ||
      output_state == nullptr || output_kinds == nullptr ||
      output_targets == nullptr) {
    return -1;
  }
  std::array<poker::ComboId, 2> hands;
  poker::Board board;
  if (!DecodeBrowserCards(cards, board_count, hands, board)) return -1;

  poker::HistoryId history{};
  if (!ReplayHistory({input_kinds, input_count},
                     {input_targets, input_count}, board.street(), history)) {
    return -1;
  }
  const poker::HistoryNode& node = Game().history.nodes[poker::Index(history)];
  WriteBrowserState(node.state, hands, board, dealer, output_state);
  const auto* decision = std::get_if<poker::DecisionState>(&node.state);
  if (decision == nullptr) return 0;

  const poker::AbstractActions actions =
      poker::SelectAbstractActions(Game().abstraction, *decision);
  for (size_t action = 0; action < actions.size(); ++action) {
    output_kinds[action] = std::to_underlying(actions[action].kind);
    output_targets[action] = actions[action].target_street_commitment;
  }
  return static_cast<int>(actions.size());
}

}  // extern "C"
