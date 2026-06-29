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

bool ContainsCard(const std::vector<Card>& cards, const Card& card) {
  return std::any_of(cards.begin(), cards.end(), [&](const Card& candidate) {
    return SameCard(candidate, card);
  });
}

void AddCard(BoardState* state, const Card& card) {
  *state->add_cards() = card;
}

void CheckDealtHandsAreDisjoint() {
  std::vector<Card> deck = BuildDeck();
  Hand player_a = DealHand(&deck);
  Hand player_b = DealHand(&deck);

  Expect(deck.size() == 48, "dealing two hands removes four cards");
  for (const Card& left : player_a.cards()) {
    for (const Card& right : player_b.cards()) {
      Expect(!SameCard(left, right), "dealt player hands should be disjoint");
    }
  }
}

void CheckSampledStreetCardsAvoidKnownCards() {
  Hand player_a;
  *player_a.add_cards() = MakeCard(14, Suit::SPADES);
  *player_a.add_cards() = MakeCard(13, Suit::SPADES);

  Hand player_b;
  *player_b.add_cards() = MakeCard(12, Suit::HEARTS);
  *player_b.add_cards() = MakeCard(11, Suit::HEARTS);

  BoardState state;
  state.set_street(Street::FLOP);
  AddCard(&state, MakeCard(2, Suit::CLUBS));
  AddCard(&state, MakeCard(3, Suit::DIAMONDS));
  AddCard(&state, MakeCard(4, Suit::HEARTS));

  std::mt19937 rng(12345);
  std::vector<Card> sampled = SampleStreetCards(state, player_a, player_b, &rng);
  Expect(sampled.size() == 1, "flop samples one turn card");

  std::vector<Card> known;
  for (const Card& card : player_a.cards()) known.push_back(card);
  for (const Card& card : player_b.cards()) known.push_back(card);
  for (const Card& card : state.cards()) known.push_back(card);

  Expect(!ContainsCard(known, sampled[0]), "sampled cards should avoid known cards");
}

void CheckSamplingThrowsWhenDeckIsTooSmall() {
  std::vector<Card> deck = BuildDeck();
  Hand player_a;
  *player_a.add_cards() = deck[0];
  *player_a.add_cards() = deck[1];

  Hand player_b;
  *player_b.add_cards() = deck[2];
  *player_b.add_cards() = deck[3];

  BoardState state;
  state.set_street(Street::PREFLOP);
  for (int i = 4; i < 50; ++i) {
    AddCard(&state, deck[i]);
  }

  std::mt19937 rng(12345);
  bool threw = false;
  try {
    SampleStreetCards(state, player_a, player_b, &rng);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "sampling should throw when too few cards remain");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckDealtHandsAreDisjoint();
  poker::CheckSampledStreetCardsAvoidKnownCards();
  poker::CheckSamplingThrowsWhenDeckIsTooSmall();
  return 0;
}
