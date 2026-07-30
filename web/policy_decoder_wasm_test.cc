#include "web/policy_decoder_wasm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "src/bet_abstraction.h"
#include "src/card_abstraction.h"
#include "src/neural_features.h"
#include "src/policy_codec.h"

namespace poker {
namespace {

void AppendU32(std::vector<uint8_t>& bytes, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendU64(std::vector<uint8_t>& bytes, uint64_t value) {
  AppendU32(bytes, static_cast<uint32_t>(value));
  AppendU32(bytes, static_cast<uint32_t>(value >> 32));
}

std::vector<uint8_t> ZeroNeuralPolicy() {
  constexpr uint32_t kHiddenSize = 1;
  constexpr size_t kParameterCount =
      kNeuralFeatureCount + 1 + 2 * (1 + 1) +
      kMaxActionsPerNode + kMaxActionsPerNode;
  std::vector<uint8_t> bytes;
  AppendU32(bytes, 0x314e4e50);
  AppendU32(bytes, 1);
  AppendU32(bytes, kNeuralFeatureSchemaVersion);
  AppendU32(bytes, kNeuralFeatureCount);
  AppendU32(bytes, kHiddenSize);
  AppendU32(bytes, kMaxActionsPerNode);
  AppendU64(bytes, web::kNeuralModel);
  bytes.resize(bytes.size() + kParameterCount * sizeof(float));
  return bytes;
}

TEST_CASE("standalone decoder matches the compact policy codec") {
  constexpr uint64_t kPublic = (uint64_t{5} << 32) | 7;
  const InfoSetKey key{PublicObservationId(kPublic), HistoryId(20),
                       PrivateObservationId(3)};
  Policy policy;
  policy.model = ModelFingerprint{web::kTabularModel};
  policy.rows.try_emplace(key, 0);
  policy.probabilities = {0.7f, 0.3f};
  const auto encoded = EncodePolicy(
      policy, {.max_actions = kMaxActionsPerNode});
  REQUIRE(encoded.ok());
  REQUIRE(poker_load_policy(encoded->data(), encoded->size()) == 1);

  std::array<float, 2> output;
  CHECK(poker_strategy(7, 5, 20, 3, output.size(), output.data()) == 1);
  CHECK((output == std::array<float, 2>{11.0f / 16, 5.0f / 16}));
  CHECK(poker_strategy(0, 0, 0, 0, output.size(), output.data()) == 0);
  CHECK((output == std::array<float, 2>{0.5f, 0.5f}));
  CHECK(poker_model_low() == static_cast<uint32_t>(web::kTabularModel));
  CHECK(poker_model_high() ==
        static_cast<uint32_t>(web::kTabularModel >> 32));

  poker_unload_policy();
  CHECK(poker_strategy(0, 0, 0, 0, output.size(), output.data()) == -1);
}

TEST_CASE("browser query owns history actions observations and inference") {
  const Card first(Rank::Ace, Suit::Hearts);
  const Card second(Rank::King, Suit::Spades);
  const ComboId hand = CardsToComboId(first, second);
  const CardAbstractionConfig cards{
      PublicCardMode::CompactTexture,
      PrivateAbstractionKind::Handcrafted36,
      RecallMode::CurrentBucketOnly};
  const PublicPosition position(cards, Board{});
  const InfoSetKey root_key{
      position.observation(), HistoryId{}, ObservePrivate(hand, position)};
  Policy policy;
  policy.model = ModelFingerprint{web::kTabularModel};
  policy.rows.try_emplace(root_key, 0);
  policy.probabilities = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  const auto encoded = EncodePolicy(
      policy, {.max_actions = kMaxActionsPerNode});
  REQUIRE(encoded.ok());
  REQUIRE(poker_load_policy(encoded->data(), encoded->size()) == 1);

  const std::array<uint8_t, 2> input_cards = {
      static_cast<uint8_t>(first.index()),
      static_cast<uint8_t>(second.index())};
  std::array<uint8_t, kMaxActionsPerNode> kinds = {};
  std::array<int32_t, kMaxActionsPerNode> targets = {};
  std::array<float, kMaxActionsPerNode> probabilities = {};
  const int action_count = poker_query(
      web::kTabularPolicy, nullptr, nullptr, 0,
      input_cards.data(), 0, kinds.data(), targets.data(),
      probabilities.data());
  REQUIRE(action_count == 5);
  CHECK(poker_history_node_count() == 136689);
  CHECK(poker_query_found() == 1);
  CHECK(kinds[0] == std::to_underlying(ActionKind::Fold));
  CHECK(kinds[1] == std::to_underlying(ActionKind::Call));
  CHECK(kinds[2] == std::to_underlying(ActionKind::Raise));
  CHECK(kinds[4] == std::to_underlying(ActionKind::AllIn));
  CHECK(std::ranges::equal(
      std::span(targets).first(5),
      std::array<int32_t, 5>{0, 2, 4, 6, 200}));
  CHECK(std::ranges::equal(
      std::span(probabilities).first(5),
      std::array<float, 5>{0.0f, 0.0f, 1.0f, 0.0f, 0.0f}));

  const std::vector<uint8_t> neural = ZeroNeuralPolicy();
  REQUIRE(poker_load_neural_policy(neural.data(), neural.size()) == 1);
  REQUIRE(poker_query(
      web::kNeuralPolicy, nullptr, nullptr, 0,
      input_cards.data(), 0, kinds.data(), targets.data(),
      probabilities.data()) == 5);
  CHECK(poker_query_found() == 1);
  for (float probability : std::span(probabilities).first(5)) {
    CHECK(probability == doctest::Approx(0.2f));
  }
  poker_unload_policy();
  poker_unload_neural_policy();
}

TEST_CASE("browser replay returns canonical state and terminal payouts") {
  const std::array<uint8_t, 9> cards = {
      static_cast<uint8_t>(Card(Rank::Ace, Suit::Hearts).index()),
      static_cast<uint8_t>(Card(Rank::Ace, Suit::Spades).index()),
      static_cast<uint8_t>(Card(Rank::King, Suit::Hearts).index()),
      static_cast<uint8_t>(Card(Rank::King, Suit::Spades).index()),
      static_cast<uint8_t>(Card(Rank::Two, Suit::Clubs).index()),
      static_cast<uint8_t>(Card(Rank::Three, Suit::Diamonds).index()),
      static_cast<uint8_t>(Card(Rank::Four, Suit::Spades).index()),
      static_cast<uint8_t>(Card(Rank::Eight, Suit::Clubs).index()),
      static_cast<uint8_t>(Card(Rank::Nine, Suit::Diamonds).index()),
  };
  std::array<int32_t, web::BrowserStateFieldCount> state = {};
  std::array<uint8_t, kMaxActionsPerNode> output_kinds = {};
  std::array<int32_t, kMaxActionsPerNode> output_targets = {};

  CHECK(poker_replay(
            0, nullptr, nullptr, 0, cards.data(), 0, state.data(),
            output_kinds.data(), output_targets.data()) == 5);
  CHECK(state[web::Phase] == web::DecisionPhase);
  CHECK(state[web::Actor] == 0);
  CHECK(state[web::Stack0] == 199);
  CHECK(state[web::Stack1] == 198);
  CHECK(state[web::Bet0] == 1);
  CHECK(state[web::Bet1] == 2);
  CHECK(state[web::Pot] == 3);
  CHECK(state[web::CallAmount] == 1);

  const std::array<uint8_t, 2> preflop = {
      std::to_underlying(ActionKind::Call),
      std::to_underlying(ActionKind::Check)};
  const std::array<int32_t, 2> passive_targets = {2, 0};
  CHECK(poker_replay(
            0, preflop.data(), passive_targets.data(), preflop.size(),
            cards.data(), 0, state.data(), output_kinds.data(),
            output_targets.data()) == 0);
  CHECK(state[web::Phase] == web::ChancePhase);
  CHECK(state[web::CardsNeeded] == 3);

  CHECK(poker_replay(
            0, preflop.data(), passive_targets.data(), preflop.size(),
            cards.data(), 3, state.data(), output_kinds.data(),
            output_targets.data()) == 4);
  CHECK(state[web::Phase] == web::DecisionPhase);
  CHECK(state[web::Street] == std::to_underlying(StreetKind::Flop));
  CHECK(state[web::Actor] == 1);

  const std::array<uint8_t, 8> showdown_actions = {
      std::to_underlying(ActionKind::Call),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check),
      std::to_underlying(ActionKind::Check)};
  const std::array<int32_t, 8> showdown_targets = {2, 0, 0, 0, 0, 0, 0, 0};
  CHECK(poker_replay(
            0, showdown_actions.data(), showdown_targets.data(),
            showdown_actions.size(), cards.data(), 5, state.data(),
            output_kinds.data(), output_targets.data()) == 0);
  CHECK(state[web::Phase] == web::ShowdownPhase);
  CHECK(state[web::WinnerMask] == 1);
  CHECK(state[web::Stack0] == 202);
  CHECK(state[web::Stack1] == 198);
  CHECK(state[web::Pot] == 0);

  const uint8_t fold = std::to_underlying(ActionKind::Fold);
  const int32_t zero = 0;
  CHECK(poker_replay(
            1, &fold, &zero, 1, cards.data(), 0, state.data(),
            output_kinds.data(), output_targets.data()) == 0);
  CHECK(state[web::Phase] == web::FoldPhase);
  CHECK(state[web::WinnerMask] == 1);
  CHECK(state[web::Stack0] == 201);
  CHECK(state[web::Stack1] == 199);
}

}  // namespace
}  // namespace poker
