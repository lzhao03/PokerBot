#include "src/card_utils.h"

#include <cstdint>
#include <random>
#include <stdexcept>

namespace poker {

constexpr int kRanksPerSuit = 13;
constexpr int kDeckSize = 52;

std::vector<Card> BuildDeck() {
  std::vector<Card> deck;
  deck.reserve(kDeckSize);
  for (int suit = Suit::HEARTS; suit <= Suit::SPADES; ++suit) {
    for (int rank = 2; rank <= 14; ++rank) {
      deck.push_back(MakeCard(rank, static_cast<Suit>(suit)));
    }
  }
  return deck;
}

namespace {

int CardId(const Card& card) {
  const int rank_index = card.rank() - 2;
  const int suit_index = static_cast<int>(card.suit()) - 1;
  if (rank_index < 0 || rank_index >= kRanksPerSuit || suit_index < 0 ||
      suit_index >= 4) {
    return -1;
  }
  return suit_index * kRanksPerSuit + rank_index;
}

int64_t CardIdBit(int card_id) {
  return int64_t{1} << card_id;
}

Card CardFromId(int card_id) {
  return MakeCard(card_id % kRanksPerSuit + 2,
                  static_cast<Suit>(card_id / kRanksPerSuit +
                                    static_cast<int>(Suit::HEARTS)));
}

void AddCardToMask(const Card& card, int64_t& mask) {
  const int card_id = CardId(card);
  if (card_id >= 0) {
    mask |= CardIdBit(card_id);
  }
}

int64_t KnownCardMask(const BoardState& state,
                      const Hand& player_a_hand,
                      const Hand& player_b_hand) {
  int64_t mask = 0;
  for (const Card& card : player_a_hand.cards()) {
    AddCardToMask(card, mask);
  }
  for (const Card& card : player_b_hand.cards()) {
    AddCardToMask(card, mask);
  }
  for (const Card& card : state.cards()) {
    AddCardToMask(card, mask);
  }
  return mask;
}

int AvailableCards(int64_t known_mask) {
  int count = 0;
  for (int card_id = 0; card_id < kDeckSize; ++card_id) {
    if ((known_mask & CardIdBit(card_id)) == 0) {
      ++count;
    }
  }
  return count;
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
  const int count = CardsForNextStreet(state.street());
  if (count <= 0) {
    return {};
  }

  int64_t blocked_mask = KnownCardMask(state, player_a_hand, player_b_hand);
  if (AvailableCards(blocked_mask) < count) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  std::uniform_int_distribution<int> card_distribution(0, kDeckSize - 1);
  std::vector<Card> sampled;
  sampled.reserve(count);
  while (static_cast<int>(sampled.size()) < count) {
    const int card_id = card_distribution(rng);
    const int64_t card_bit = CardIdBit(card_id);
    if ((blocked_mask & card_bit) != 0) {
      continue;
    }
    blocked_mask |= card_bit;
    sampled.push_back(CardFromId(card_id));
  }
  return sampled;
}

}  // namespace poker
