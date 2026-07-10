#include "doctest/doctest.h"
#include "src/game_state.h"
#include "src/hand_evaluator.h"

#include <array>
#include <bit>
#include <stdexcept>

namespace poker {
namespace {

TEST_CASE("board rejects duplicates and keeps its mask consistent") {
  Board board;
  const CardId first = MakeCardId(2, SuitKind::kHearts);
  board.add(first);
  board.add(MakeCardId(7, SuitKind::kDiamonds));
  board.add(MakeCardId(12, SuitKind::kClubs));

  CHECK(board.count == 3);
  CHECK(std::popcount(board.mask) == board.count);
  CHECK(board.contains(first));
  CHECK_THROWS_AS(board.add(first), std::invalid_argument);
}

TEST_CASE("hand evaluator recognizes a royal flush") {
  const std::array<CardId, 5> cards = {
      MakeCardId(10, SuitKind::kHearts),
      MakeCardId(11, SuitKind::kHearts),
      MakeCardId(12, SuitKind::kHearts),
      MakeCardId(13, SuitKind::kHearts),
      MakeCardId(14, SuitKind::kHearts),
  };

  CHECK(HandEvaluator().evaluate(cards).rank == HandRank::ROYAL_FLUSH);
}

}  // namespace
}  // namespace poker
