#include "tests/identity_test_support.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace poker {
namespace {

using S = SuitKind;

[[maybe_unused]] CardId C(int rank, S suit) {
  return MakeCardId(rank, suit);
}

struct PublicObservationTrace {
  std::optional<PublicStreetObservation> flop;
  std::optional<PublicStreetObservation> turn;
  std::optional<PublicStreetObservation> river;
  bool operator==(const PublicObservationTrace&) const = default;
};

struct PrivateObservationTrace {
  PrivateStreetObservation preflop;
  std::optional<PrivateStreetObservation> flop;
  std::optional<PrivateStreetObservation> turn;
  std::optional<PrivateStreetObservation> river;
  bool operator==(const PrivateObservationTrace&) const = default;
};

[[maybe_unused]] std::optional<BoardRunout> DecodeExactPublicObservation(
    PublicObservationId observation) {
  const uint8_t count = static_cast<uint8_t>(observation & 0x7);
  if (count != 0 && count != 3 && count != 4 && count != 5) {
    return std::nullopt;
  }
  std::array<CardId, kMaxBoardCards> cards = {};
  PublicObservationId encoded = observation >> 3;
  for (uint8_t i = 0; i < count; ++i) {
    cards[i] = static_cast<CardId>(encoded & 0x3F);
    if (cards[i] >= kDeckCardCount) return std::nullopt;
    encoded >>= 6;
  }
  if (encoded != 0) return std::nullopt;

  BoardRunout board = BoardRunout::Preflop();
  try {
    if (count >= 3) board.deal_flop({cards.data(), 3});
    if (count >= 4) board.deal_turn(cards[3]);
    if (count == 5) board.deal_river(cards[4]);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  return exact_public_observation(board) == observation
             ? std::optional<BoardRunout>(board)
             : std::nullopt;
}

[[maybe_unused]] PublicObservationTrace PublicTraceForRunout(
    const BoardRunout& board) {
  PublicObservationTrace trace;
  if (board.count() == 0) return trace;
  const auto cards = board.cards();
  BoardRunout prefix = BoardRunout::Preflop();
  prefix.deal_flop(cards.first(3));
  trace.flop = observe_public_street(StreetKind::kFlop, prefix);
  if (board.count() == 3) return trace;
  prefix.deal_turn(cards[3]);
  trace.turn = observe_public_street(StreetKind::kTurn, prefix);
  if (board.count() == 4) return trace;
  prefix.deal_river(cards[4]);
  trace.river = observe_public_street(StreetKind::kRiver, prefix);
  return trace;
}

std::optional<PublicObservationTrace> DecodePublicObservation(
    PublicObservationId observation) {
  if constexpr (!kCoarsePublicBuckets) {
    const auto board = DecodeExactPublicObservation(observation);
    if (!board.has_value()) return std::nullopt;
    return PublicTraceForRunout(*board);
  }

  constexpr PublicObservationId kSlotMask =
      (PublicObservationId{1} << kPublicObservationBitsPerStreet) - 1;
  if ((observation >> (3 * kPublicObservationBitsPerStreet)) != 0) {
    return std::nullopt;
  }
  PublicObservationTrace trace;
  std::array<std::optional<PublicStreetObservation>*, 3> slots = {
      &trace.flop, &trace.turn, &trace.river};
  bool missing = false;
  for (size_t i = 0; i < slots.size(); ++i) {
    const PublicObservationId encoded =
        (observation >> (i * kPublicObservationBitsPerStreet)) & kSlotMask;
    if (encoded == 0) {
      missing = true;
    } else if (missing || encoded > kCoarsePublicStreetObservationCount) {
      return std::nullopt;
    } else {
      *slots[i] = PublicStreetObservation{encoded - 1, {}};
    }
  }
  return trace;
}

std::optional<PrivateObservationTrace> DecodePrivateObservation(
    PrivateObservationId observation,
    StreetKind street) {
  PrivateObservationTrace trace;
  if constexpr (!kCoarsePrivateBuckets) {
    if (observation >= kComboCount) return std::nullopt;
    const PrivateStreetObservation hand{
        static_cast<PrivateBucketId>(observation)};
    trace.preflop = hand;
    if (street >= StreetKind::kFlop) trace.flop = hand;
    if (street >= StreetKind::kTurn) trace.turn = hand;
    if (street >= StreetKind::kRiver) trace.river = hand;
    return trace;
  }

  constexpr PrivateObservationId kSlotMask =
      (PrivateObservationId{1} << kPrivateObservationBitsPerStreet) - 1;
  std::array<std::optional<PrivateStreetObservation>*, 4> slots = {
      nullptr, &trace.flop, &trace.turn, &trace.river};
  for (int index = 0; index < 4; ++index) {
    const PrivateObservationId encoded =
        (observation >> (index * kPrivateObservationBitsPerStreet)) &
        kSlotMask;
    const bool reached = index <= static_cast<int>(street);
    if (reached != (encoded != 0) ||
        encoded > kCoarsePrivateStreetObservationCount) {
      return std::nullopt;
    }
    if (!reached) continue;
    const PrivateStreetObservation current{
        static_cast<PrivateBucketId>(encoded - 1)};
    if (index == 0) {
      trace.preflop = current;
    } else {
      *slots[static_cast<size_t>(index)] = current;
    }
  }
  if ((observation >> (4 * kPrivateObservationBitsPerStreet)) != 0) {
    return std::nullopt;
  }
  return trace;
}

int ObservationCount(const PublicObservationTrace& trace) {
  return static_cast<int>(trace.flop.has_value()) +
         static_cast<int>(trace.turn.has_value()) +
         static_cast<int>(trace.river.has_value());
}

bool IsStrictPrefix(const PublicObservationTrace& parent,
                    const PublicObservationTrace& child) {
  return ObservationCount(child) == ObservationCount(parent) + 1 &&
         (!parent.flop.has_value() || parent.flop == child.flop) &&
         (!parent.turn.has_value() || parent.turn == child.turn) &&
         (!parent.river.has_value() || parent.river == child.river);
}

int ObservationCount(const PrivateObservationTrace& trace) {
  return 1 + static_cast<int>(trace.flop.has_value()) +
         static_cast<int>(trace.turn.has_value()) +
         static_cast<int>(trace.river.has_value());
}

bool IsStrictPrefix(const PrivateObservationTrace& parent,
                    const PrivateObservationTrace& child) {
  return ObservationCount(child) == ObservationCount(parent) + 1 &&
         parent.preflop == child.preflop &&
         (!parent.flop.has_value() || parent.flop == child.flop) &&
         (!parent.turn.has_value() || parent.turn == child.turn) &&
         (!parent.river.has_value() || parent.river == child.river);
}

CardId SelectAvailable(CardMask blocked, uint16_t choice) {
  std::array<CardId, kDeckCardCount> available = {};
  size_t count = 0;
  for (int id = 0; id < kDeckCardCount; ++id) {
    const CardId card = static_cast<CardId>(id);
    if ((blocked & CardBit(card)) == 0) {
      available[count++] = card;
    }
  }
  return available[choice % count];
}

TEST_CASE("encoded observation traces advance one street at a time") {
  CHECK(rc::check("valid runout observation traces", [] {
    std::array<CardId, 7> cards = {};
    CardMask used = 0;
    for (CardId& card : cards) {
      card = SelectAvailable(
          used, *rc::gen::arbitrary<uint16_t>());
      used |= CardBit(card);
    }

    const ComboId hand = CardsToComboId(cards[0], cards[1]);
    BoardRunout board = BoardRunout::Preflop();
    PublicObservationId public_id =
        public_observation_id(StreetKind::kPreflop, board);
    PrivateObservationId private_id = initial_private_observation(hand);
    auto public_trace = DecodePublicObservation(public_id);
    auto private_trace = DecodePrivateObservation(
        private_id, StreetKind::kPreflop);
    RC_ASSERT(public_trace.has_value());
    RC_ASSERT(private_trace.has_value());

    const std::array<CardId, 3> flop = {cards[2], cards[3], cards[4]};
    board.deal_flop(flop);
    const std::array<StreetKind, 3> streets = {
        StreetKind::kFlop, StreetKind::kTurn, StreetKind::kRiver};
    for (StreetKind street : streets) {
      if (street == StreetKind::kTurn) board.deal_turn(cards[5]);
      if (street == StreetKind::kRiver) board.deal_river(cards[6]);

      const PublicObservationId child_public =
          public_observation_after_chance(public_id, street, board);
      const auto child_public_trace =
          DecodePublicObservation(child_public);
      RC_ASSERT(child_public_trace.has_value());
      RC_ASSERT(IsStrictPrefix(*public_trace, *child_public_trace));

      const PrivateObservationId child_private =
          advance_private_observation(
              private_id, hand, street, board, child_public);
      const auto child_private_trace =
          DecodePrivateObservation(child_private, street);
      RC_ASSERT(child_private_trace.has_value());
      RC_ASSERT(IsStrictPrefix(*private_trace, *child_private_trace));

      public_id = child_public;
      private_id = child_private;
      public_trace = child_public_trace;
      private_trace = child_private_trace;
    }

    if constexpr (!kCoarsePublicBuckets) {
      const auto decoded = DecodeExactPublicObservation(public_id);
      RC_ASSERT(decoded.has_value());
      RC_ASSERT(*decoded == board);
    }
  }));
}

TEST_CASE("packed public observations preserve arbitrary prefixes") {
  if constexpr (kCoarsePublicBuckets) {
    CHECK(rc::check("coarse public observation prefixes", [] {
      PublicObservationId history = initial_public_observation();
      auto trace = DecodePublicObservation(history);
      RC_ASSERT(trace.has_value());
      const std::array<StreetKind, 3> streets = {
          StreetKind::kFlop, StreetKind::kTurn, StreetKind::kRiver};
      for (StreetKind street : streets) {
        const auto bucket = static_cast<BoardBucketId>(
            *rc::gen::inRange(
                0, static_cast<int>(kCoarsePublicStreetObservationCount)));
        history = advance_public_observation(
            history, street, PublicStreetObservation{bucket, {}});
        const auto child = DecodePublicObservation(history);
        RC_ASSERT(child.has_value());
        RC_ASSERT(IsStrictPrefix(*trace, *child));
        trace = child;
      }
    }));
  }
}

#if !POKER_COARSE_PUBLIC_BUCKETS
TEST_CASE("exact public observations round-trip every canonical flop") {
  for (int a = 0; a < kDeckCardCount; ++a) {
    for (int b = a + 1; b < kDeckCardCount; ++b) {
      for (int c = b + 1; c < kDeckCardCount; ++c) {
        BoardRunout board = BoardRunout::Preflop();
        const std::array<CardId, 3> flop = {
            static_cast<CardId>(a), static_cast<CardId>(b),
            static_cast<CardId>(c)};
        board.deal_flop(flop);
        auto decoded = DecodeExactPublicObservation(
            exact_public_observation(board));
        REQUIRE(decoded.has_value());
        CHECK(*decoded == board);

        const CardId turn = SelectAvailable(board.mask(), 0);
        board.deal_turn(turn);
        decoded = DecodeExactPublicObservation(
            exact_public_observation(board));
        REQUIRE(decoded.has_value());
        CHECK(*decoded == board);

        const CardId river = SelectAvailable(board.mask(), 1);
        board.deal_river(river);
        decoded = DecodeExactPublicObservation(
            exact_public_observation(board));
        REQUIRE(decoded.has_value());
        CHECK(*decoded == board);
      }
    }
  }
}
#endif

#if !(POKER_COARSE_PUBLIC_BUCKETS && POKER_COARSE_PRIVATE_BUCKETS)
TEST_CASE("strict infoset identity preserves complete observations") {
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId parent = test::ClosePreflop(graph, preflop);
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  const std::array<CardId, 3> flop_a = {
      C(2, S::kHearts), C(3, S::kHearts), C(6, S::kHearts)};
#if POKER_COARSE_PUBLIC_BUCKETS
  const std::array<CardId, 3> flop_b = {
      C(2, S::kDiamonds), C(3, S::kDiamonds), C(6, S::kDiamonds)};
#else
  const std::array<CardId, 3> flop_b = {
      C(6, S::kHearts), C(2, S::kHearts), C(3, S::kHearts)};
#endif
  const NodeId node_a = graph.chance_child(parent, state_a, flop_a);
  const NodeId node_b = graph.chance_child(parent, state_b, flop_b);
  REQUIRE(node_a == node_b);
  CHECK(graph.access().betting_history(node_a) ==
        graph.access().betting_history(node_b));

  const PublicObservationId public_id =
      graph.access().public_observation(node_a);
  const auto public_trace = DecodePublicObservation(public_id);
  REQUIRE(public_trace.has_value());
  CHECK(public_trace ==
        DecodePublicObservation(
            graph.access().public_observation(node_b)));

  const ComboId hand = CardsToComboId(
      C(14, S::kSpades), C(13, S::kSpades));
  const PrivateObservationId private_a = private_observation_for_runout(
      hand, state_a.board, public_id);
  const PrivateObservationId private_b = private_observation_for_runout(
      hand, state_b.board, public_id);
  REQUIRE(private_a == private_b);
  CHECK(DecodePrivateObservation(private_a, StreetKind::kFlop) ==
        DecodePrivateObservation(private_b, StreetKind::kFlop));

  const StrategyTables::InfoSetKey key_a{node_a, private_a};
  const StrategyTables::InfoSetKey key_b{node_b, private_b};
  CHECK(key_a.node_id == key_b.node_id);
  CHECK(key_a.private_id == key_b.private_id);

  const BettingAbstraction abstraction(test::IdentityConfig());
  const int player = state_a.betting.player_to_act;
  REQUIRE(player == state_b.betting.player_to_act);
  CHECK(graph.own_actions(node_a, player) ==
        graph.own_actions(node_b, player));
  const ActionMenu menu_a =
      abstraction.actions_for_betting_node(state_a.betting);
  const ActionMenu menu_b =
      abstraction.actions_for_betting_node(state_b.betting);
  REQUIRE(menu_a.count == menu_b.count);
  for (uint8_t action = 0; action < menu_a.count; ++action) {
    CHECK(menu_a.actions[action] == menu_b.actions[action]);
  }
}
#endif

#if !POKER_COARSE_PUBLIC_BUCKETS
TEST_CASE("turn and river order remains part of public identity") {
  test::IdentityGraph graph;
  ExactPublicState preflop = graph.initial_state();
  const NodeId parent = test::ClosePreflop(graph, preflop);
  const std::array<CardId, 3> flop = {
      C(2, S::kHearts), C(7, S::kDiamonds), C(12, S::kClubs)};
  ExactPublicState state_a = preflop;
  ExactPublicState state_b = preflop;
  NodeId node_a = graph.chance_child(parent, state_a, flop);
  NodeId node_b = graph.chance_child(parent, state_b, flop);
  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);
  node_a = graph.chance_child(
      node_a, state_a, std::array<CardId, 1>{C(9, S::kSpades)});
  node_b = graph.chance_child(
      node_b, state_b, std::array<CardId, 1>{C(11, S::kHearts)});
  node_a = test::CheckCheck(graph, node_a, state_a);
  node_b = test::CheckCheck(graph, node_b, state_b);
  node_a = graph.chance_child(
      node_a, state_a, std::array<CardId, 1>{C(11, S::kHearts)});
  node_b = graph.chance_child(
      node_b, state_b, std::array<CardId, 1>{C(9, S::kSpades)});

  CHECK(state_a.board.mask() == state_b.board.mask());
  CHECK(state_a.board != state_b.board);
  CHECK(graph.access().public_observation(node_a) !=
        graph.access().public_observation(node_b));
  CHECK(node_a != node_b);
}
#endif

}  // namespace
}  // namespace poker
