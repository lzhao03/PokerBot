#include "src/hand_evaluator.h"

#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

Hand MakeHand(int first_rank, Suit first_suit, int second_rank,
              Suit second_suit) {
  Hand hand;
  *hand.add_cards() = MakeCard(first_rank, first_suit);
  *hand.add_cards() = MakeCard(second_rank, second_suit);
  return hand;
}

BoardState Board(std::vector<Card> cards) {
  BoardState state;
  for (const Card& card : cards) {
    *state.add_cards() = card;
  }
  return state;
}

void CheckFiveCardEvaluation() {
  Hand royal_flush;
  *royal_flush.add_cards() = MakeCard(10, Suit::HEARTS);
  *royal_flush.add_cards() = MakeCard(11, Suit::HEARTS);
  *royal_flush.add_cards() = MakeCard(12, Suit::HEARTS);
  *royal_flush.add_cards() = MakeCard(13, Suit::HEARTS);
  *royal_flush.add_cards() = MakeCard(14, Suit::HEARTS);

  HandEvaluator evaluator;
  HandEvaluation evaluation = evaluator.evaluate(royal_flush);
  Expect(evaluation.rank == HandRank::ROYAL_FLUSH,
         "five-card royal flush should rank as royal flush");
}

void CheckSevenCardBestHand() {
  Hand hand = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  BoardState board = Board({
      MakeCard(14, Suit::DIAMONDS),
      MakeCard(13, Suit::HEARTS),
      MakeCard(13, Suit::CLUBS),
      MakeCard(2, Suit::SPADES),
      MakeCard(7, Suit::HEARTS),
  });

  HandEvaluator evaluator;
  HandEvaluation evaluation = evaluator.evaluate_hand(hand, board);
  Expect(evaluation.rank == HandRank::FULL_HOUSE,
         "seven-card evaluation should choose full house");
  Expect(evaluation.kickers.size() == 2, "full house should have two kickers");
  Expect(evaluation.kickers[0] == 14 && evaluation.kickers[1] == 13,
         "full house should be aces full of kings");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckFiveCardEvaluation();
  poker::CheckSevenCardBestHand();
  return 0;
}
