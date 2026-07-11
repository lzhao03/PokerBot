#include "src/card_abstraction.h"
#include "src/hand_evaluator.h"
#include "src/poker.h"
#include "tools/hand_evaluator_table_builder.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <random>
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
  return C(RankFromCardId(card),
           suits[static_cast<size_t>(SuitIndex(SuitFromCardId(card)))]);
}

Board Rename(const Board& board, const std::array<S, 4>& suits) {
  std::array<Card, kMaxBoardCards> cards = {};
  const auto board_cards = BoardCards(board);
  for (size_t i = 0; i < board_cards.size(); ++i) {
    cards[i] = Rename(board_cards[i], suits);
  }
  return Runout(absl::Span<const Card>(cards.data(), BoardCount(board)));
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
      C(14, S::kHearts), C(5, S::kDiamonds), C(4, S::kClubs),
      C(3, S::kSpades), C(2, S::kHearts)});
  CHECK(wheel.rank == HandRank::STRAIGHT);
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
    const BoardFeatures features = board_features(board);
    CHECK(board_features(reversed) == features);
    CHECK(board_texture_bucket(street, features) ==
          board_texture_bucket(street, board_features(reversed)));
    const PrivateStreetObservation private_observation =
        observe_private_street(hand, street, features);
    CHECK(private_observation.value == hand.index());

    std::array<S, 4> suits = {
        S::kHearts, S::kDiamonds, S::kClubs, S::kSpades};
    std::shuffle(suits.begin(), suits.end(), rng);
    const Board renamed_board = Rename(board, suits);
    CHECK(board_texture_bucket(street, features) ==
          board_texture_bucket(street, board_features(renamed_board)));

    const auto sampled = SampleStreetCards(street, board, ComboMask(hand), rng);
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
  CHECK_THROWS_AS(SampleStreetCards(StreetKind::kPreflop,
                                    Board{PreflopBoard{}}, blocked, rng),
                  std::runtime_error);
}

}  // namespace
}  // namespace poker
