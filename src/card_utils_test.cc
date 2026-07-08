#include "src/card_utils.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

bool ContainsCard(const std::vector<CardId>& cards, CardId card) {
  return std::find(cards.begin(), cards.end(), card) != cards.end();
}

template <typename CardContainer>
void CheckUniqueCards(const CardContainer& cards) {
  for (size_t i = 0; i < cards.size(); ++i) {
    for (size_t j = i + 1; j < cards.size(); ++j) {
      CAPTURE(i);
      CAPTURE(j);
      CHECK(cards[i] != cards[j]);
    }
  }
}

TEST_CASE("deck contains every card once") {
  std::vector<CardId> deck = BuildDeck();
  CHECK(deck.size() == kDeckCardCount);
  CheckUniqueCards(deck);
  CHECK(ContainsCard(deck, MakeCardId(14, SuitKind::kSpades)));
}

TEST_CASE("street helpers return expected card counts") {
  CHECK(CardsForNextStreet(StreetKind::kPreflop) == 3);
  CHECK(CardsForNextStreet(StreetKind::kFlop) == 1);
  CHECK(CardsForNextStreet(StreetKind::kTurn) == 1);
  CHECK(CardsForNextStreet(StreetKind::kRiver) == 0);

  CHECK(StreetAfterChance(StreetKind::kPreflop) == StreetKind::kFlop);
  CHECK(StreetAfterChance(StreetKind::kFlop) == StreetKind::kTurn);
  CHECK(StreetAfterChance(StreetKind::kTurn) == StreetKind::kRiver);
  CHECK(StreetAfterChance(StreetKind::kRiver) == StreetKind::kRiver);

  CHECK(BoardCardsForStreet(StreetKind::kPreflop) == 0);
  CHECK(BoardCardsForStreet(StreetKind::kFlop) == 3);
  CHECK(BoardCardsForStreet(StreetKind::kTurn) == 4);
  CHECK(BoardCardsForStreet(StreetKind::kRiver) == 5);
}

TEST_CASE("sampled turn card avoids private and board cards") {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(12, SuitKind::kHearts));
  known_private_cards |= CardBit(MakeCardId(11, SuitKind::kHearts));

  CompactPublicState state;
  state.street = StreetKind::kFlop;
  AddBoardCard(state, MakeCardId(2, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(3, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(4, SuitKind::kHearts));

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(state, known_private_cards, rng);
  REQUIRE(sampled.size() == 1);

  const CardMask known_cards = known_private_cards | state.board_mask;
  CHECK((known_cards & CardBit(sampled[0])) == 0);
}

TEST_CASE("sampled river card avoids private and board cards") {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));

  CompactPublicState state;
  state.street = StreetKind::kTurn;
  AddBoardCard(state, MakeCardId(2, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(3, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(4, SuitKind::kHearts));
  AddBoardCard(state, MakeCardId(5, SuitKind::kSpades));

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(state, known_private_cards, rng);
  REQUIRE(sampled.size() == 1);

  const CardMask known_cards = known_private_cards | state.board_mask;
  CHECK((known_cards & CardBit(sampled[0])) == 0);
}

TEST_CASE("primitive street sampling avoids private and board cards") {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));

  CardMask board_mask = 0;
  board_mask |= CardBit(MakeCardId(2, SuitKind::kClubs));
  board_mask |= CardBit(MakeCardId(3, SuitKind::kDiamonds));
  board_mask |= CardBit(MakeCardId(4, SuitKind::kHearts));
  board_mask |= CardBit(MakeCardId(5, SuitKind::kSpades));

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(
      StreetKind::kTurn, 4, board_mask, known_private_cards, rng);
  REQUIRE(sampled.size() == 1);

  const CardMask known_cards = known_private_cards | board_mask;
  CHECK((known_cards & CardBit(sampled[0])) == 0);
}

TEST_CASE("one-card sampling returns the only unblocked card") {
  std::vector<CardId> deck = BuildDeck();
  const CardId legal_card = deck.back();
  CardMask known_private_cards = 0;
  for (CardId card : deck) {
    if (card != legal_card) {
      known_private_cards |= CardBit(card);
    }
  }

  CompactPublicState state;
  state.street = StreetKind::kFlop;

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(state, known_private_cards, rng);
  REQUIRE(sampled.size() == 1);
  CHECK(sampled[0] == legal_card);
}

TEST_CASE("sampled flop cards are unique and avoid private cards") {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(12, SuitKind::kHearts));
  known_private_cards |= CardBit(MakeCardId(11, SuitKind::kHearts));

  CompactPublicState state;
  state.street = StreetKind::kPreflop;

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(state, known_private_cards, rng);
  REQUIRE(sampled.size() == 3);
  CheckUniqueCards(sampled);

  for (CardId card : sampled) {
    CHECK((known_private_cards & CardBit(card)) == 0);
  }
}

TEST_CASE("sampling throws when the deck is too small") {
  std::vector<CardId> deck = BuildDeck();
  CardMask known_private_cards = 0;
  for (int i = 0; i < 50; ++i) {
    known_private_cards |= CardBit(deck[i]);
  }

  CompactPublicState state;
  state.street = StreetKind::kPreflop;

  std::mt19937 rng(12345);
  CHECK_THROWS_AS(SampleStreetCards(state, known_private_cards, rng),
                  std::runtime_error);
}

}  // namespace
}  // namespace poker
