#include "src/hand_evaluator.h"
#include "src/poker.h"

#include <array>
#include <bit>

#include "doctest/doctest.h"

namespace poker {
namespace {

Card C(int rank, Suit suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
}

TEST_CASE("board types preserve reveal order and card invariants") {
  const Card ace = C(14, Suit::kSpades);
  const Card king = C(13, Suit::kSpades);
  const Card queen = C(12, Suit::kSpades);
  const Card jack = C(11, Suit::kHearts);
  const Card ten = C(10, Suit::kDiamonds);

  const FlopBoard flop = DealFlop(PreflopBoard{}, {ace, king, queen});
  const FlopBoard permuted = DealFlop(PreflopBoard{}, {queen, ace, king});
  CHECK(flop == permuted);

  const TurnBoard first_turn = DealTurn(flop, jack);
  const TurnBoard different_turn = DealTurn(
      DealFlop(PreflopBoard{}, {ace, king, jack}), queen);
  CHECK(first_turn.mask() == different_turn.mask());
  CHECK_FALSE(first_turn == different_turn);

  const RiverBoard river = DealRiver(first_turn, ten);
  const RiverBoard different_river = DealRiver(DealTurn(flop, ten), jack);
  CHECK(river.mask() == different_river.mask());
  CHECK_FALSE(river == different_river);
  CHECK(std::popcount(river.mask()) == kMaxBoardCards);

  CHECK_FALSE(MakeFlop({ace, king, ace}).ok());
  CHECK_FALSE(MakeTurn(flop, ace).ok());
  CHECK_FALSE(MakeRiver(first_turn, jack).ok());
}

TEST_CASE("hand evaluator recognizes a royal flush") {
  const std::array<Card, 5> cards = {
      C(10, Suit::kHearts), C(11, Suit::kHearts),
      C(12, Suit::kHearts), C(13, Suit::kHearts),
      C(14, Suit::kHearts),
  };

  CHECK(EvaluateFiveCards(cards).rank == HandRank::ROYAL_FLUSH);
}

}  // namespace
}  // namespace poker
