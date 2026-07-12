#include "src/poker.h"

#include <array>
#include <bit>

#include "doctest/doctest.h"

namespace poker {
namespace {

Card C(int rank, Suit suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
}

TEST_CASE("board preserves reveal order and card invariants") {
  const Card ace = C(14, Suit::Spades);
  const Card king = C(13, Suit::Spades);
  const Card queen = C(12, Suit::Spades);
  const Card jack = C(11, Suit::Hearts);
  const Card ten = C(10, Suit::Diamonds);

  const Board flop = DealCards(
      Board{}, std::array<Card, 3>{ace, king, queen});
  const Board permuted = DealCards(
      Board{}, std::array<Card, 3>{queen, ace, king});
  CHECK(flop == permuted);

  const Board first_turn = DealCards(flop, std::array<Card, 1>{jack});
  const Board different_turn = DealCards(
      DealCards(Board{}, std::array<Card, 3>{ace, king, jack}),
      std::array<Card, 1>{queen});
  CHECK(first_turn.mask() == different_turn.mask());
  CHECK_FALSE(first_turn == different_turn);

  const Board river = DealCards(first_turn, std::array<Card, 1>{ten});
  const Board different_river = DealCards(
      DealCards(flop, std::array<Card, 1>{ten}),
      std::array<Card, 1>{jack});
  CHECK(river.mask() == different_river.mask());
  CHECK_FALSE(river == different_river);
  CHECK(std::popcount(river.mask()) == kMaxBoardCards);

  const auto made = MakeBoard(
      std::array<Card, 5>{queen, ace, king, jack, ten});
  REQUIRE(made.ok());
  CHECK(*made == river);

  CHECK_FALSE(MakeBoard(std::array<Card, 2>{ace, king}).ok());
  CHECK_FALSE(MakeBoard(std::array<Card, 3>{ace, king, ace}).ok());
  CHECK_FALSE(
      MakeBoard(std::array<Card, 4>{ace, king, queen, ace}).ok());
}

}  // namespace
}  // namespace poker
