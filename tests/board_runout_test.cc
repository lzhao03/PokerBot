#include "doctest/doctest.h"
#include "src/poker.h"
#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace poker {
namespace {

CardId Card(int rank, SuitKind suit) {
  return MakeCardId(rank, suit);
}

BoardRunout Flop(std::array<CardId, 3> cards) {
  BoardRunout runout = BoardRunout::Preflop();
  runout.deal_flop(cards);
  return runout;
}

TEST_CASE("flop order canonicalizes while later streets remain observable") {
  const CardId ace = Card(14, SuitKind::kSpades);
  const CardId king = Card(13, SuitKind::kSpades);
  const CardId queen = Card(12, SuitKind::kSpades);
  const CardId jack = Card(11, SuitKind::kHearts);
  const CardId ten = Card(10, SuitKind::kDiamonds);

  BoardRunout first = Flop({ace, king, queen});
  BoardRunout permuted = Flop({queen, ace, king});
  CHECK(first == permuted);

  first.deal_turn(jack);
  BoardRunout different_turn = Flop({ace, king, jack});
  different_turn.deal_turn(queen);
  CHECK(first.mask() == different_turn.mask());
  CHECK_FALSE(first == different_turn);

  first.deal_river(ten);
  BoardRunout different_river = Flop({ace, king, queen});
  different_river.deal_turn(ten);
  different_river.deal_river(jack);
  CHECK(first.mask() == different_river.mask());
  CHECK_FALSE(first == different_river);
}

TEST_CASE("runout dealing enforces street and card invariants") {
  const CardId first = Card(2, SuitKind::kHearts);
  const CardId second = Card(7, SuitKind::kDiamonds);
  const CardId third = Card(12, SuitKind::kClubs);
  BoardRunout runout = Flop({third, first, second});

  CHECK(runout.count() == 3);
  CHECK(std::popcount(runout.mask()) == runout.count());
  CHECK(std::is_sorted(runout.cards().begin(), runout.cards().end()));
  CHECK(runout.contains(first));
  CHECK_THROWS_AS(runout.deal_turn(first), std::invalid_argument);

  BoardRunout preflop = BoardRunout::Preflop();
  const std::array<CardId, 2> short_flop = {first, second};
  CHECK_THROWS_AS(preflop.deal_flop(short_flop), std::invalid_argument);
  CHECK_THROWS_AS(preflop.deal_turn(first), std::logic_error);
  BoardRunout no_turn = Flop({first, second, third});
  CHECK_THROWS_AS(no_turn.deal_river(Card(9, SuitKind::kSpades)),
                  std::logic_error);
  const std::array<CardId, 3> duplicate_flop = {first, second, first};
  CHECK_THROWS_AS(preflop.deal_flop(duplicate_flop),
                  std::invalid_argument);
}

TEST_CASE("hand evaluator recognizes a royal flush") {
  const std::array<CardId, 5> cards = {
      Card(10, SuitKind::kHearts),
      Card(11, SuitKind::kHearts),
      Card(12, SuitKind::kHearts),
      Card(13, SuitKind::kHearts),
      Card(14, SuitKind::kHearts),
  };

  CHECK(EvaluateFiveCards(cards).rank == HandRank::ROYAL_FLUSH);
}

}  // namespace
}  // namespace poker
