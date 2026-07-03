#include "src/card_utils.h"

#include <algorithm>
#include <stdexcept>

namespace poker {

std::vector<Card> BuildDeck() {
  std::vector<Card> deck;
  deck.reserve(52);
  for (int suit = Suit::HEARTS; suit <= Suit::SPADES; ++suit) {
    for (int rank = 2; rank <= 14; ++rank) {
      deck.push_back(MakeCard(rank, static_cast<Suit>(suit)));
    }
  }
  return deck;
}

namespace {

void RemoveCard(std::vector<Card>& deck, const Card& card) {
  auto it = std::find_if(deck.begin(), deck.end(), [&](const Card& deck_card) {
    return SameCard(deck_card, card);
  });
  if (it != deck.end()) {
    deck.erase(it);
  }
}

void RemoveHand(std::vector<Card>& deck, const Hand& hand) {
  for (const Card& card : hand.cards()) {
    RemoveCard(deck, card);
  }
}

void RemoveBoard(std::vector<Card>& deck, const BoardState& state) {
  for (const Card& card : state.cards()) {
    RemoveCard(deck, card);
  }
}

}  // namespace

Hand DealHand(std::vector<Card>& deck) {
  if (deck.size() < 2) {
    throw std::runtime_error("Not enough cards to deal a hand");
  }

  Hand hand;
  *hand.add_cards() = deck.back();
  deck.pop_back();
  *hand.add_cards() = deck.back();
  deck.pop_back();
  return hand;
}

int CardsForNextStreet(Street street) {
  switch (street) {
    case Street::PREFLOP:
      return 3;
    case Street::FLOP:
    case Street::TURN:
      return 1;
    case Street::RIVER:
      return 0;
    default:
      return 0;
  }
}

std::vector<Card> SampleStreetCards(const BoardState& state,
                                    const Hand& player_a_hand,
                                    const Hand& player_b_hand,
                                    std::mt19937& rng) {
  int count = CardsForNextStreet(state.street());
  std::vector<Card> deck = BuildDeck();
  RemoveHand(deck, player_a_hand);
  RemoveHand(deck, player_b_hand);
  RemoveBoard(deck, state);

  if (deck.size() < static_cast<size_t>(count)) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  std::shuffle(deck.begin(), deck.end(), rng);
  return std::vector<Card>(deck.end() - count, deck.end());
}

}  // namespace poker
