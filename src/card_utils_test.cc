#include "src/card_utils.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool ContainsCard(const std::vector<CardId>& cards, CardId card) {
  return std::find(cards.begin(), cards.end(), card) != cards.end();
}

void ExpectUniqueCards(const std::vector<CardId>& cards, const char* message) {
  for (size_t i = 0; i < cards.size(); ++i) {
    for (size_t j = i + 1; j < cards.size(); ++j) {
      Expect(cards[i] != cards[j], message);
    }
  }
}

void CheckBuildDeckHasUniqueCards() {
  std::vector<CardId> deck = BuildDeck();
  Expect(deck.size() == kDeckCardCount, "deck should contain 52 cards");
  ExpectUniqueCards(deck, "deck should contain unique cards");
  Expect(ContainsCard(deck, MakeCardId(14, SuitKind::kSpades)),
         "deck should contain ace of spades");
}

void CheckCardsForNextStreet() {
  Expect(CardsForNextStreet(StreetKind::kPreflop) == 3,
         "preflop should deal three flop cards");
  Expect(CardsForNextStreet(StreetKind::kFlop) == 1,
         "flop should deal one turn card");
  Expect(CardsForNextStreet(StreetKind::kTurn) == 1,
         "turn should deal one river card");
  Expect(CardsForNextStreet(StreetKind::kRiver) == 0,
         "river should deal no cards");
}

void CheckSampledStreetCardsAvoidKnownCards() {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(12, SuitKind::kHearts));
  known_private_cards |= CardBit(MakeCardId(11, SuitKind::kHearts));

  GameState state;
  state.street = StreetKind::kFlop;
  AddBoardCard(state, MakeCardId(2, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(3, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(4, SuitKind::kHearts));

  std::mt19937 rng(12345);
  std::vector<CardId> sampled = SampleStreetCards(state, known_private_cards, rng);
  Expect(sampled.size() == 1, "flop samples one turn card");

  CardMask known_cards = known_private_cards | state.board_mask;
  Expect((known_cards & CardBit(sampled[0])) == 0,
         "sampled cards should avoid known cards");
}

void CheckSampledFlopCardsAreUniqueAndAvoidKnownCards() {
  CardMask known_private_cards = 0;
  known_private_cards |= CardBit(MakeCardId(14, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(13, SuitKind::kSpades));
  known_private_cards |= CardBit(MakeCardId(12, SuitKind::kHearts));
  known_private_cards |= CardBit(MakeCardId(11, SuitKind::kHearts));

  GameState state;
  state.street = StreetKind::kPreflop;

  std::mt19937 rng(12345);
  std::vector<CardId> sampled = SampleStreetCards(state, known_private_cards, rng);
  Expect(sampled.size() == 3, "preflop samples three flop cards");
  ExpectUniqueCards(sampled, "sampled flop cards should be unique");

  for (CardId card : sampled) {
    Expect((known_private_cards & CardBit(card)) == 0,
           "sampled flop should avoid known cards");
  }
}

void CheckSamplingThrowsWhenDeckIsTooSmall() {
  std::vector<CardId> deck = BuildDeck();
  CardMask known_private_cards = 0;
  for (int i = 0; i < 50; ++i) {
    known_private_cards |= CardBit(deck[i]);
  }

  GameState state;
  state.street = StreetKind::kPreflop;

  std::mt19937 rng(12345);
  bool threw = false;
  try {
    SampleStreetCards(state, known_private_cards, rng);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "sampling should throw when too few cards remain");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckBuildDeckHasUniqueCards();
  poker::CheckCardsForNextStreet();
  poker::CheckSampledStreetCardsAvoidKnownCards();
  poker::CheckSampledFlopCardsAreUniqueAndAvoidKnownCards();
  poker::CheckSamplingThrowsWhenDeckIsTooSmall();
  return 0;
}
