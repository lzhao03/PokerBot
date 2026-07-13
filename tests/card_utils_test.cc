#include "src/card_abstraction.h"
#include "src/card_canonicalization.h"
#include "src/hand_evaluator.h"
#include "src/poker.h"
#include "src/solver.h"
#include "tools/hand_evaluator_table_builder.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <random>
#include <set>
#include <utility>

namespace poker {
namespace {

using S = Suit;
using hand_evaluator_generation::EvaluationScore;
using hand_evaluator_generation::HandRank;

Card C(int rank, S suit) { return Card(static_cast<Rank>(rank - 2), suit); }
ComboId H(Card a, Card b) { return CardsToComboId(a, b); }

Board Runout(absl::Span<const Card> cards) {
  if (cards.empty()) {
    return {};
  }
  Board board = DealCards(
      Board{}, absl::Span<const Card>(cards.data(), 3));
  for (size_t index = 3; index < cards.size(); ++index) {
    board = DealCards(
        board, absl::Span<const Card>(cards.data() + index, 1));
  }
  return board;
}

Board B(std::initializer_list<Card> cards) {
  return Runout(absl::Span<const Card>(cards.begin(), cards.size()));
}

EvaluationScore ReferenceBest(ComboId hand, const Board& board) {
  std::array<Card, 7> cards = {};
  const auto hole_cards = hand.cards();
  cards[0] = hole_cards[0];
  cards[1] = hole_cards[1];
  size_t count = 2;
  for (Card card : board.cards()) cards[count++] = card;

  EvaluationScore best;
  for (size_t a = 0; a + 4 < count; ++a)
    for (size_t b = a + 1; b + 3 < count; ++b)
      for (size_t c = b + 1; c + 2 < count; ++c)
        for (size_t d = c + 1; d + 1 < count; ++d)
          for (size_t e = d + 1; e < count; ++e)
            best = std::max(
                best, hand_evaluator_generation::EvaluateFiveCardScore(
                          {cards[a], cards[b], cards[c], cards[d], cards[e]}));
  return best;
}

int ReferenceCompare(ComboId a, ComboId b, const Board& board) {
  const EvaluationScore ea = ReferenceBest(a, board);
  const EvaluationScore eb = ReferenceBest(b, board);
  return ea > eb ? 1 : ea < eb ? -1 : 0;
}

Card Rename(Card card, const std::array<S, 4>& suits) {
  return C(PokerRank(card),
           suits[static_cast<size_t>(card.suit())]);
}

Board Rename(const Board& board, const std::array<S, 4>& suits) {
  std::array<Card, kMaxBoardCards> cards = {};
  const auto board_cards = board.cards();
  for (size_t i = 0; i < board_cards.size(); ++i) {
    cards[i] = Rename(board_cards[i], suits);
  }
  return Runout(absl::Span<const Card>(cards.data(), board.count()));
}

ComboId Rename(ComboId hand, const std::array<S, 4>& suits) {
  const auto cards = hand.cards();
  return H(Rename(cards[0], suits), Rename(cards[1], suits));
}

TEST_CASE("combo ids are an exhaustive canonical bijection") {
  size_t expected = 0;
  for (int a = 0; a < kDeckCardCount; ++a) {
    for (int b = a + 1; b < kDeckCardCount; ++b, ++expected) {
      const Card ca = kDeck[static_cast<size_t>(a)];
      const Card cb = kDeck[static_cast<size_t>(b)];
      CHECK(H(ca, cb).index() == expected);
      CHECK(H(cb, ca).index() == expected);
      const ComboId combo = H(ca, cb);
      CHECK((combo.cards() == std::array<Card, 2>{ca, cb}));
      CHECK(combo.mask() == (CardBit(ca) | CardBit(cb)));
    }
  }
  CHECK(expected == kComboCount);
  CHECK_FALSE(MaybeCardsToComboId(kDeck[0], kDeck[0]).has_value());
  const auto hole_cards = MaybeCardsToComboId(kDeck[0], kDeck[1]);
  REQUIRE(hole_cards.has_value());
  CHECK(*hole_cards == H(kDeck[0], kDeck[1]));
}

TEST_CASE("hand evaluator matches an independent five-card oracle") {
  const auto wheel = hand_evaluator_generation::EvaluateFiveCardScore(
      std::array<Card, 5>{C(14, S::Hearts), C(5, S::Diamonds),
                          C(4, S::Clubs), C(3, S::Spades),
                          C(2, S::Hearts)});
  CHECK(wheel.rank == HandRank::Straight);
  CHECK(wheel.kickers[0] == 5);

  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 10000; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId a = H(deck[0], deck[1]);
    const ComboId b = H(deck[2], deck[3]);
    const Board board = B({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CAPTURE(trial);
    CHECK(CompareHands(a, b, board) == ReferenceCompare(a, b, board));
  }
}

TEST_CASE("sampling and card abstractions preserve identity") {
  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(7);
  const CardAbstractionConfig current{
      PublicCardMode::Texture,
      PrivateAbstractionKind::Handcrafted36,
      RecallMode::CurrentBucketOnly};

  for (int trial = 0; trial < 150; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const StreetKind street = static_cast<StreetKind>(trial % 3);
    const size_t board_count =
        street == StreetKind::Preflop ? 0 : std::to_underlying(street) + 2;
    const Board board = Runout(
        absl::Span<const Card>(deck.data(), board_count));
    const ComboId hand = H(deck[board_count], deck[board_count + 1]);

    std::array<Card, kMaxBoardCards> permuted = {};
    const auto cards = board.cards();
    std::copy(cards.begin(), cards.end(), permuted.begin());
    if (board.count() >= 3) {
      std::swap(permuted[0], permuted[2]);
    }
    const Board reversed = Runout(
        absl::Span<const Card>(permuted.data(), board.count()));
    CHECK(ObservePublic(current, reversed) == ObservePublic(current, board));
    const PrivateObservationId private_observation =
        ObservePrivate(current, hand, board);
    CHECK(std::to_underlying(private_observation) >= 1);
    CHECK(std::to_underlying(private_observation) <= 36);

    std::array<S, 4> suits = {
        S::Hearts, S::Diamonds, S::Clubs, S::Spades};
    std::shuffle(suits.begin(), suits.end(), rng);
    const Board renamed_board = Rename(board, suits);
    CHECK(ObservePublic(current, renamed_board) ==
          ObservePublic(current, board));

    const auto sampled_result =
        SampleStreetCards(street, board, hand.mask(), rng);
    REQUIRE(sampled_result.ok());
    const auto& sampled = *sampled_result;
    CHECK(sampled.size() == static_cast<size_t>(CardsForNextStreet(street)));
    CardMask blocked = board.mask() | hand.mask();
    for (Card card : sampled) {
      CHECK((blocked & CardBit(card)) == 0);
      blocked |= CardBit(card);
    }
  }

  CardMask blocked = 0;
  for (int i = 0; i < kDeckCardCount - 2; ++i)
    blocked |= CardBit(kDeck[static_cast<size_t>(i)]);
  CHECK_FALSE(
      SampleStreetCards(StreetKind::Preflop, Board{}, blocked, rng).ok());
}

TEST_CASE("exact card observations are invariant under suit renaming") {
  const Board board = B({C(14, S::Hearts), C(13, S::Hearts),
                         C(7, S::Clubs), C(10, S::Diamonds)});
  const ComboId hand = H(C(12, S::Hearts), C(11, S::Spades));
  const PublicObservationId expected_public =
      CanonicalPublicObservation(board);
  const PrivateObservationId expected_private =
      CanonicalPrivateObservation(hand, board);

  std::array<S, 4> suits = {
      S::Hearts, S::Diamonds, S::Clubs, S::Spades};
  do {
    const Board renamed_board = Rename(board, suits);
    CHECK(CanonicalPublicObservation(renamed_board) ==
          expected_public);
    CHECK(CanonicalPrivateObservation(Rename(hand, suits), renamed_board) ==
          expected_private);
  } while (std::next_permutation(suits.begin(), suits.end()));

}

TEST_CASE("handcrafted 36 mappings remain stable") {
  const CardAbstractionConfig current{
      PublicCardMode::Texture,
      PrivateAbstractionKind::Handcrafted36,
      RecallMode::CurrentBucketOnly};
  CHECK(ObservePrivate(
            current, H(C(14, S::Hearts), C(14, S::Spades)), Board{}) ==
        PrivateObservationId(1));
  CHECK(ObservePrivate(
            current, H(C(14, S::Hearts), C(13, S::Hearts)), Board{}) ==
        PrivateObservationId(13));
  CHECK(ObservePrivate(
            current, H(C(7, S::Hearts), C(2, S::Spades)), Board{}) ==
        PrivateObservationId(36));

  const Board flop = B({C(2, S::Hearts), C(7, S::Hearts),
                        C(12, S::Hearts)});
  const ComboId hand = H(C(14, S::Hearts), C(13, S::Spades));
  CHECK(ObservePrivate(current, hand, flop) ==
        PrivateObservationId(7));
}

TEST_CASE("canonical observations preserve card relationships and order") {
  const Board monotone = B({C(14, S::Hearts), C(13, S::Hearts),
                            C(12, S::Hearts)});
  const Board rainbow = B({C(14, S::Hearts), C(13, S::Diamonds),
                           C(12, S::Clubs)});
  CHECK(CanonicalPublicObservation(monotone) !=
        CanonicalPublicObservation(rainbow));

  const Board draw_board = B({C(2, S::Hearts), C(7, S::Hearts),
                              C(12, S::Clubs)});
  const ComboId flush_draw = H(C(14, S::Hearts), C(13, S::Spades));
  const ComboId no_flush_draw =
      H(C(14, S::Diamonds), C(13, S::Spades));
  CHECK(CanonicalPrivateObservation(flush_draw, draw_board) !=
        CanonicalPrivateObservation(no_flush_draw, draw_board));

  const Board jack_then_ten =
      B({C(14, S::Hearts), C(13, S::Diamonds), C(12, S::Clubs),
         C(11, S::Spades), C(10, S::Hearts)});
  const Board ten_then_jack =
      B({C(14, S::Hearts), C(13, S::Diamonds), C(12, S::Clubs),
         C(10, S::Hearts), C(11, S::Spades)});
  CHECK(jack_then_ten.mask() == ten_then_jack.mask());
  CHECK(CanonicalPublicObservation(jack_then_ten) !=
        CanonicalPublicObservation(ten_then_jack));
}

TEST_CASE("canonical observation counts match holdem suit isomorphisms") {
  std::set<uint64_t> private_observations;
  const Board preflop;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      const ComboId hand = H(kDeck[first], kDeck[second]);
      private_observations.insert(
          std::to_underlying(
              CanonicalPrivateObservation(hand, preflop)));
    }
  }
  CHECK(private_observations.size() == 169);

  std::set<uint64_t> public_observations;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      for (size_t third = second + 1; third < kDeck.size(); ++third) {
        const Board flop = DealCards(
            Board{}, std::array<Card, 3>{
                         kDeck[first], kDeck[second], kDeck[third]});
        public_observations.insert(
            std::to_underlying(CanonicalPublicObservation(flop)));
      }
    }
  }
  CHECK(public_observations.size() == 1755);
}

TEST_CASE("all abstraction modes preserve observation history") {
  const std::array<CardAbstractionConfig, 4> configs = {{
      {PublicCardMode::ExactCanonical,
       PrivateAbstractionKind::ExactCanonical},
      {PublicCardMode::ExactCanonical,
       PrivateAbstractionKind::Handcrafted36},
      {PublicCardMode::Texture,
       PrivateAbstractionKind::ExactCanonical},
      {PublicCardMode::Texture,
       PrivateAbstractionKind::Handcrafted36},
  }};
  const ComboId hand = H(C(14, S::Hearts), C(13, S::Spades));
  const Board flop = DealCards(
      Board{}, std::array<Card, 3>{
                   C(2, S::Hearts), C(7, S::Hearts), C(12, S::Clubs)});
  const Board turn =
      DealCards(flop, std::array<Card, 1>{C(9, S::Diamonds)});
  const Board river =
      DealCards(turn, std::array<Card, 1>{C(4, S::Spades)});

  for (const CardAbstractionConfig& config : configs) {
    CAPTURE(static_cast<int>(config.public_mode));
    CAPTURE(static_cast<int>(config.private_kind));
    PrivateObservationId private_id = ObservePrivate(config, hand, Board{});
    for (const Board& board : {flop, turn, river}) {
      const PublicPosition position(config, board);
      CHECK(position.observation() ==
            ObservePublic(config, board));
      private_id = ObservePrivate(config, hand, board, private_id);
      CHECK(private_id == ObservePrivate(config, hand, board));
    }

    const PublicPosition position(config, river);
    const std::array<S, 4> renamed_suits = {
        S::Clubs, S::Spades, S::Hearts, S::Diamonds};
    const Board renamed_board = Rename(river, renamed_suits);
    const ComboId renamed_hand = Rename(hand, renamed_suits);
    const PublicPosition renamed(config, renamed_board);
    CHECK(renamed.observation() == position.observation());
    CHECK(ObservePrivate(config, renamed_hand, renamed_board) == private_id);
  }
}

TEST_CASE("coarse public exact private keeps relative flush information") {
  const CardAbstractionConfig config{
      PublicCardMode::Texture,
      PrivateAbstractionKind::ExactCanonical};
  const Board board = B({C(2, S::Hearts), C(7, S::Hearts),
                         C(12, S::Clubs)});
  const ComboId hand = H(C(14, S::Hearts), C(13, S::Spades));
  const PublicPosition position(config, board);

  const std::array<S, 4> suits = {
      S::Diamonds, S::Clubs, S::Spades, S::Hearts};
  const Board renamed_board = Rename(board, suits);
  const ComboId renamed_hand = Rename(hand, suits);
  const PublicPosition renamed(config, renamed_board);

  const InfoSetKey key{HistoryId{7}, position.observation(),
                       ObservePrivate(config, hand, board)};
  const InfoSetKey renamed_key{
      HistoryId{7}, renamed.observation(),
      ObservePrivate(config, renamed_hand, renamed_board)};
  CHECK(key == renamed_key);
}

}  // namespace
}  // namespace poker
