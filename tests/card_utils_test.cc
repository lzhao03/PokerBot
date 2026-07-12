#include "src/card_abstraction.h"
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
#include <stdexcept>

namespace poker {
namespace {

using S = Suit;

Card C(int rank, S suit) { return Card(static_cast<Rank>(rank - 2), suit); }
ComboId H(Card a, Card b) { return CardsToComboId(a, b); }

Board Runout(absl::Span<const Card> cards) {
  if (cards.empty()) {
    return PreflopBoard{};
  }
  const std::array<Card, 3> flop_cards = {cards[0], cards[1], cards[2]};
  const FlopBoard flop = DealFlop(PreflopBoard{}, flop_cards);
  if (cards.size() == 3) {
    return flop;
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  if (cards.size() == 5) {
    return DealRiver(turn, cards[4]);
  }
  return turn;
}

Board B(std::initializer_list<Card> cards) {
  return Runout(absl::Span<const Card>(cards.begin(), cards.size()));
}

HandEvaluation Score5(const std::array<Card, 5>& cards) {
  const auto score = hand_evaluator_generation::EvaluateFiveCardScore(cards);
  HandEvaluation out;
  out.rank = score.rank;
  out.kickers = score.kickers;
  out.kicker_count = score.kicker_count;
  return out;
}

HandEvaluation ReferenceBest(ComboId hand, const Board& board) {
  std::array<Card, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (Card card : BoardCards(board)) cards[count++] = card;

  HandEvaluation best;
  for (size_t a = 0; a + 4 < count; ++a)
    for (size_t b = a + 1; b + 3 < count; ++b)
      for (size_t c = b + 1; c + 2 < count; ++c)
        for (size_t d = c + 1; d + 1 < count; ++d)
          for (size_t e = d + 1; e < count; ++e)
            best = std::max(best, Score5({cards[a], cards[b], cards[c],
                                          cards[d], cards[e]}));
  return best;
}

int ReferenceCompare(ComboId a, ComboId b, const Board& board) {
  const HandEvaluation ea = ReferenceBest(a, board);
  const HandEvaluation eb = ReferenceBest(b, board);
  return ea > eb ? 1 : ea < eb ? -1 : 0;
}

Card Rename(Card card, const std::array<S, 4>& suits) {
  return C(PokerRank(card),
           suits[static_cast<size_t>(SuitIndex(CardSuit(card)))]);
}

Board Rename(const Board& board, const std::array<S, 4>& suits) {
  std::array<Card, kMaxBoardCards> cards = {};
  const auto board_cards = BoardCards(board);
  for (size_t i = 0; i < board_cards.size(); ++i) {
    cards[i] = Rename(board_cards[i], suits);
  }
  return Runout(absl::Span<const Card>(cards.data(), BoardCount(board)));
}

ComboId Rename(ComboId hand, const std::array<S, 4>& suits) {
  const ComboInfo& combo = GetComboInfo(hand);
  return H(Rename(combo.card0, suits), Rename(combo.card1, suits));
}

TEST_CASE("combo ids are an exhaustive canonical bijection") {
  size_t expected = 0;
  for (int a = 0; a < kDeckCardCount; ++a) {
    for (int b = a + 1; b < kDeckCardCount; ++b, ++expected) {
      const Card ca = kDeck[static_cast<size_t>(a)];
      const Card cb = kDeck[static_cast<size_t>(b)];
      CHECK(H(ca, cb).index() == expected);
      CHECK(H(cb, ca).index() == expected);
      const ComboInfo& info = GetComboInfo(H(ca, cb));
      CHECK(info.card0 == ca);
      CHECK(info.card1 == cb);
      CHECK(info.mask == (CardBit(ca) | CardBit(cb)));
    }
  }
  CHECK(expected == kComboCount);
  CHECK_FALSE(MakeHoleCards(kDeck[0], kDeck[0]).ok());
  const auto hole_cards = MakeHoleCards(kDeck[0], kDeck[1]);
  REQUIRE(hole_cards.ok());
  CHECK(hole_cards->combo() == H(kDeck[0], kDeck[1]));
}

TEST_CASE("hand evaluator matches an independent five-card oracle") {
  const auto wheel = EvaluateFiveCards(std::array<Card, 5>{
      C(14, S::Hearts), C(5, S::Diamonds), C(4, S::Clubs),
      C(3, S::Spades), C(2, S::Hearts)});
  CHECK(wheel.rank == HandRank::Straight);
  CHECK(wheel.kickers[0] == 5);

  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 400; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId a = H(deck[0], deck[1]);
    const ComboId b = H(deck[2], deck[3]);
    const Board board = B({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CAPTURE(trial);
    CHECK(CompareHands(a, b, std::get<RiverBoard>(board)) ==
          ReferenceCompare(a, b, board));
  }
}

TEST_CASE("sampling and card abstractions preserve identity") {
  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(7);

  for (int trial = 0; trial < 150; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const StreetKind street = static_cast<StreetKind>(trial % 3);
    const int board_count = BoardCardsForStreet(street);
    const Board board = Runout(
        absl::Span<const Card>(deck.data(), board_count));
    const ComboId hand = H(deck[board_count], deck[board_count + 1]);

    std::array<Card, kMaxBoardCards> permuted = {};
    const auto cards = BoardCards(board);
    std::copy(cards.begin(), cards.end(), permuted.begin());
    if (BoardCount(board) >= 3) {
      std::swap(permuted[0], permuted[2]);
    }
    const Board reversed = Runout(
        absl::Span<const Card>(permuted.data(), BoardCount(board)));
    const BoardFeatures features = BoardFeaturesFor(board);
    CHECK(BoardFeaturesFor(reversed) == features);
    CHECK(BoardTextureBucket(street, features) ==
          BoardTextureBucket(street, BoardFeaturesFor(reversed)));
    CHECK(Handcrafted36Bucket(hand, street, features) <
          kCoarsePrivateStreetObservationCount);

    std::array<S, 4> suits = {
        S::Hearts, S::Diamonds, S::Clubs, S::Spades};
    std::shuffle(suits.begin(), suits.end(), rng);
    const Board renamed_board = Rename(board, suits);
    CHECK(BoardTextureBucket(street, features) ==
          BoardTextureBucket(street, BoardFeaturesFor(renamed_board)));

    const auto sampled_result =
        SampleStreetCards(street, board, ComboMask(hand), rng);
    REQUIRE(sampled_result.ok());
    const auto& sampled = *sampled_result;
    CHECK(sampled.size() == static_cast<size_t>(CardsForNextStreet(street)));
    CardMask blocked = BoardMask(board) | ComboMask(hand);
    for (Card card : sampled) {
      CHECK((blocked & CardBit(card)) == 0);
      blocked |= CardBit(card);
    }
  }

  CardMask blocked = 0;
  for (int i = 0; i < kDeckCardCount - 2; ++i)
    blocked |= CardBit(kDeck[static_cast<size_t>(i)]);
  CHECK_FALSE(SampleStreetCards(StreetKind::Preflop,
                                Board{PreflopBoard{}}, blocked, rng).ok());
}

TEST_CASE("exact card observations are invariant under suit renaming") {
  const Board board = B({C(14, S::Hearts), C(13, S::Hearts),
                         C(7, S::Clubs), C(10, S::Diamonds)});
  const ComboId hand = H(C(12, S::Hearts), C(11, S::Spades));
  const CanonicalCardObservation expected =
      CanonicalizeObservation(hand, board);

  std::array<S, 4> suits = {
      S::Hearts, S::Diamonds, S::Clubs, S::Spades};
  do {
    const Board renamed_board = Rename(board, suits);
    CHECK(CanonicalPublicObservation(renamed_board) ==
          expected.public_observation);
    CHECK(CanonicalizeObservation(Rename(hand, suits), renamed_board) ==
          expected);
  } while (std::next_permutation(suits.begin(), suits.end()));

  const ComboId other_hand = H(C(9, S::Diamonds), C(8, S::Spades));
  CHECK(CanonicalizeObservation(other_hand, board).public_observation ==
        expected.public_observation);
}

TEST_CASE("handcrafted 36 mappings remain stable") {
  CHECK(Handcrafted36Bucket(
            H(C(14, S::Hearts), C(14, S::Spades)),
            StreetKind::Preflop, BoardFeatures{}) == 0);
  CHECK(Handcrafted36Bucket(
            H(C(14, S::Hearts), C(13, S::Hearts)),
            StreetKind::Preflop, BoardFeatures{}) == 12);
  CHECK(Handcrafted36Bucket(
            H(C(7, S::Hearts), C(2, S::Spades)),
            StreetKind::Preflop, BoardFeatures{}) == 35);

  const Board flop = B({C(2, S::Hearts), C(7, S::Hearts),
                        C(12, S::Hearts)});
  const ComboId hand = H(C(14, S::Hearts), C(13, S::Spades));
  CHECK(Handcrafted36Bucket(hand, StreetKind::Flop,
                            BoardFeaturesFor(flop)) == 6);

  const CardAbstractionConfig current{
      PublicCardMode::Texture,
      PrivateAbstractionKind::Handcrafted36,
      RecallMode::CurrentBucketOnly};
  const CardAbstraction abstraction(current);
  const PublicPosition position =
      PublicPosition::Root(abstraction, StreetKind::Flop, flop);
  CHECK(ObservePrivate(abstraction, hand, position) ==
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
  CHECK(CanonicalizeObservation(flush_draw, draw_board) !=
        CanonicalizeObservation(no_flush_draw, draw_board));

  const Board jack_then_ten =
      B({C(14, S::Hearts), C(13, S::Diamonds), C(12, S::Clubs),
         C(11, S::Spades), C(10, S::Hearts)});
  const Board ten_then_jack =
      B({C(14, S::Hearts), C(13, S::Diamonds), C(12, S::Clubs),
         C(10, S::Hearts), C(11, S::Spades)});
  CHECK(BoardMask(jack_then_ten) == BoardMask(ten_then_jack));
  CHECK(CanonicalPublicObservation(jack_then_ten) !=
        CanonicalPublicObservation(ten_then_jack));
}

TEST_CASE("canonical observation counts match holdem suit isomorphisms") {
  std::set<uint64_t> private_observations;
  const Board preflop = PreflopBoard{};
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      const ComboId hand = H(kDeck[first], kDeck[second]);
      private_observations.insert(
          CanonicalizeObservation(hand, preflop).private_observation.value());
    }
  }
  CHECK(private_observations.size() == 169);

  std::set<uint64_t> public_observations;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      for (size_t third = second + 1; third < kDeck.size(); ++third) {
        const FlopBoard flop = DealFlop(
            PreflopBoard{}, {kDeck[first], kDeck[second], kDeck[third]});
        public_observations.insert(
            CanonicalPublicObservation(Board{flop}).value());
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
  const FlopBoard flop = DealFlop(
      PreflopBoard{},
      {C(2, S::Hearts), C(7, S::Hearts), C(12, S::Clubs)});
  const TurnBoard turn = DealTurn(flop, C(9, S::Diamonds));
  const RiverBoard river = DealRiver(turn, C(4, S::Spades));

  for (const CardAbstractionConfig& config : configs) {
    CAPTURE(static_cast<int>(config.public_mode));
    CAPTURE(static_cast<int>(config.private_kind));
    const CardAbstraction abstraction(config);
    PublicPosition position = PublicPosition::Root(
        abstraction, StreetKind::Preflop, Board{PreflopBoard{}});
    PrivateObservationId private_id =
        ObservePrivate(abstraction, hand, position);

    for (const auto& [street, board] : {
             std::pair{StreetKind::Flop, Board{flop}},
             std::pair{StreetKind::Turn, Board{turn}},
             std::pair{StreetKind::River, Board{river}},
         }) {
      position = position.after_chance(abstraction, street, board);
      private_id = ObservePrivate(abstraction, hand, position);
      CHECK(position.observation() ==
            ObservePublic(abstraction, board));
      CHECK(private_id == ObservePrivate(abstraction, hand, position));
    }

    const std::array<S, 4> renamed_suits = {
        S::Clubs, S::Spades, S::Hearts, S::Diamonds};
    const Board renamed_board = Rename(Board{river}, renamed_suits);
    const ComboId renamed_hand = Rename(hand, renamed_suits);
    const PublicPosition renamed = PublicPosition::Root(
        abstraction, StreetKind::River, renamed_board);
    CHECK(renamed.observation() == position.observation());
    CHECK(ObservePrivate(abstraction, renamed_hand, renamed) == private_id);
  }
}

TEST_CASE("coarse public exact private keeps relative flush information") {
  const CardAbstractionConfig config{
      PublicCardMode::Texture,
      PrivateAbstractionKind::ExactCanonical};
  const CardAbstraction abstraction(config);
  const Board board = B({C(2, S::Hearts), C(7, S::Hearts),
                         C(12, S::Clubs)});
  const ComboId hand = H(C(14, S::Hearts), C(13, S::Spades));
  const PublicPosition position =
      PublicPosition::Root(abstraction, StreetKind::Flop, board);

  const std::array<S, 4> suits = {
      S::Diamonds, S::Clubs, S::Spades, S::Hearts};
  const Board renamed_board = Rename(board, suits);
  const ComboId renamed_hand = Rename(hand, suits);
  const PublicPosition renamed =
      PublicPosition::Root(abstraction, StreetKind::Flop, renamed_board);

  const InfoSetKey key{HistoryId(7), position.observation(),
                       ObservePrivate(abstraction, hand, position)};
  const InfoSetKey renamed_key{
      HistoryId(7), renamed.observation(),
      ObservePrivate(abstraction, renamed_hand, renamed)};
  CHECK(key == renamed_key);
}

}  // namespace
}  // namespace poker
