#include "src/hand_evaluator.h"
#include "src/hand_evaluator_table_builder.h"
#include "src/hand_evaluator_tables.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace poker {
namespace {

CardId TestCard(int rank, SuitKind suit) {
  return MakeCardId(rank, suit);
}

ComboId TestCombo(int first_rank,
                  SuitKind first_suit,
                  int second_rank,
                  SuitKind second_suit) {
  return CardsToComboId(TestCard(first_rank, first_suit),
                        TestCard(second_rank, second_suit));
}

ComboId TestCombo(CardId first, CardId second) {
  return CardsToComboId(first, second);
}

CompactPublicState TestBoard(std::initializer_list<CardId> cards) {
  CompactPublicState board;
  for (CardId card : cards) {
    AddBoardCard(board, card);
  }
  return board;
}

HandEvaluation ToHandEvaluation(
    const hand_evaluator_generation::EvaluationScore& score) {
  HandEvaluation evaluation;
  evaluation.rank = score.rank;
  evaluation.kickers = score.kickers;
  evaluation.kicker_count = score.kicker_count;
  return evaluation;
}

HandEvaluation ReferenceBestHandFromFiveCardEvaluation(
    ComboId hand, const CompactPublicState& board) {
  std::array<CardId, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (uint8_t i = 0; i < board.board_count; ++i) {
    cards[count] = board.board_cards[i];
    ++count;
  }

  if (count < 5) {
    throw std::invalid_argument("Need at least five cards");
  }

  HandEvaluation best;
  std::array<CardId, 5> subset = {};
  for (size_t a = 0; a + 4 < count; ++a) {
    subset[0] = cards[a];
    for (size_t b = a + 1; b + 3 < count; ++b) {
      subset[1] = cards[b];
      for (size_t c = b + 1; c + 2 < count; ++c) {
        subset[2] = cards[c];
        for (size_t d = c + 1; d + 1 < count; ++d) {
          subset[3] = cards[d];
          for (size_t e = d + 1; e < count; ++e) {
            subset[4] = cards[e];
            const HandEvaluation current = ToHandEvaluation(
                hand_evaluator_generation::EvaluateFiveCardScore(subset));
            if (best < current) {
              best = current;
            }
          }
        }
      }
    }
  }
  return best;
}

int ReferenceCompare(ComboId first,
                     ComboId second,
                     const CompactPublicState& board) {
  const HandEvaluation first_eval =
      ReferenceBestHandFromFiveCardEvaluation(first, board);
  const HandEvaluation second_eval =
      ReferenceBestHandFromFiveCardEvaluation(second, board);
  if (first_eval > second_eval) {
    return 1;
  }
  if (first_eval < second_eval) {
    return -1;
  }
  return 0;
}

void CheckCompareMatchesReference(ComboId first,
                                  ComboId second,
                                  const CompactPublicState& board,
                                  const char* message) {
  CAPTURE(message);
  HandEvaluator evaluator;
  CHECK(evaluator.compare_hands(first, second, board) ==
        ReferenceCompare(first, second, board));
}

TEST_CASE("generated Cactus tables match reference builder") {
  const auto reference = hand_evaluator_generation::BuildCactusTables();
  CHECK(reference.flushes == hand_evaluator_tables::kCactusFlushes);
  CHECK(reference.unique5 == hand_evaluator_tables::kCactusUnique5);
  REQUIRE(reference.products.size() ==
          hand_evaluator_tables::kCactusProducts.size());
  for (size_t i = 0; i < reference.products.size(); ++i) {
    CAPTURE(i);
    CHECK(reference.products[i] == hand_evaluator_tables::kCactusProducts[i]);
  }
  for (size_t i = 0; i < reference.scores.size(); ++i) {
    CAPTURE(i);
    const auto& expected = reference.scores[i];
    const auto& actual = hand_evaluator_tables::kCactusScores[i];
    CHECK(expected.rank == actual.rank);
    CHECK(expected.kickers == actual.kickers);
    CHECK(expected.kicker_count == actual.kicker_count);
  }
}

TEST_CASE("wheel straight is scored as five high") {
  HandEvaluator evaluator;
  const std::array<CardId, 5> wheel = {
      TestCard(14, SuitKind::kHearts),
      TestCard(5, SuitKind::kDiamonds),
      TestCard(4, SuitKind::kClubs),
      TestCard(3, SuitKind::kSpades),
      TestCard(2, SuitKind::kHearts),
  };
  const std::array<CardId, 5> six_high = {
      TestCard(6, SuitKind::kHearts),
      TestCard(5, SuitKind::kDiamonds),
      TestCard(4, SuitKind::kClubs),
      TestCard(3, SuitKind::kSpades),
      TestCard(2, SuitKind::kHearts),
  };

  HandEvaluation wheel_eval = evaluator.evaluate(wheel);
  CHECK(wheel_eval.rank == HandRank::STRAIGHT);
  CHECK(wheel_eval.kicker_count == 1);
  CHECK(wheel_eval.kickers[0] == 5);
  CHECK(evaluator.evaluate(six_high) > wheel_eval);

  const std::array<CardId, 5> wheel_flush = {
      TestCard(14, SuitKind::kHearts),
      TestCard(5, SuitKind::kHearts),
      TestCard(4, SuitKind::kHearts),
      TestCard(3, SuitKind::kHearts),
      TestCard(2, SuitKind::kHearts),
  };
  HandEvaluation wheel_flush_eval = evaluator.evaluate(wheel_flush);
  CHECK(wheel_flush_eval.rank == HandRank::STRAIGHT_FLUSH);
  CHECK(wheel_flush_eval.kicker_count == 1);
  CHECK(wheel_flush_eval.kickers[0] == 5);

  const CompactPublicState straight_board =
      TestBoard({TestCard(14, SuitKind::kClubs),
                 TestCard(5, SuitKind::kDiamonds),
                 TestCard(4, SuitKind::kHearts),
                 TestCard(3, SuitKind::kSpades),
                 TestCard(9, SuitKind::kClubs)});
  const ComboId wheel_combo =
      TestCombo(2, SuitKind::kHearts, 13, SuitKind::kClubs);
  const ComboId six_high_combo =
      TestCombo(6, SuitKind::kHearts, 2, SuitKind::kClubs);
  CHECK(evaluator.compare_hands(six_high_combo, wheel_combo,
                                straight_board) > 0);
}

TEST_CASE("representative Cactus comparisons match reference evaluator") {
  struct CompareCase {
    ComboId first;
    ComboId second;
    CompactPublicState board;
    const char* label;
  };

  const CompareCase cases[] = {
      {TestCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
       TestCombo(9, SuitKind::kHearts, 8, SuitKind::kHearts),
       TestBoard({TestCard(10, SuitKind::kHearts),
                  TestCard(11, SuitKind::kHearts),
                  TestCard(12, SuitKind::kHearts),
                  TestCard(2, SuitKind::kClubs),
                  TestCard(3, SuitKind::kDiamonds)}),
       "straight flush"},
      {TestCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades),
       TestCombo(13, SuitKind::kHearts, 13, SuitKind::kDiamonds),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(14, SuitKind::kDiamonds),
                  TestCard(14, SuitKind::kClubs),
                  TestCard(2, SuitKind::kClubs),
                  TestCard(7, SuitKind::kDiamonds)}),
       "quads"},
      {TestCombo(14, SuitKind::kClubs, 13, SuitKind::kDiamonds),
       TestCombo(13, SuitKind::kHearts, 13, SuitKind::kSpades),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(14, SuitKind::kDiamonds),
                  TestCard(13, SuitKind::kClubs),
                  TestCard(2, SuitKind::kSpades),
                  TestCard(7, SuitKind::kHearts)}),
       "full house"},
      {TestCombo(13, SuitKind::kHearts, 9, SuitKind::kHearts),
       TestCombo(11, SuitKind::kHearts, 10, SuitKind::kHearts),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(2, SuitKind::kHearts),
                  TestCard(7, SuitKind::kHearts),
                  TestCard(12, SuitKind::kDiamonds),
                  TestCard(3, SuitKind::kClubs)}),
       "flush"},
      {TestCombo(14, SuitKind::kHearts, 5, SuitKind::kDiamonds),
       TestCombo(5, SuitKind::kSpades, 6, SuitKind::kDiamonds),
       TestBoard({TestCard(2, SuitKind::kClubs),
                  TestCard(3, SuitKind::kDiamonds),
                  TestCard(4, SuitKind::kHearts),
                  TestCard(9, SuitKind::kSpades),
                  TestCard(13, SuitKind::kClubs)}),
       "wheel straight"},
      {TestCombo(9, SuitKind::kSpades, 14, SuitKind::kHearts),
       TestCombo(9, SuitKind::kClubs, 12, SuitKind::kHearts),
       TestBoard({TestCard(9, SuitKind::kHearts),
                  TestCard(9, SuitKind::kDiamonds),
                  TestCard(2, SuitKind::kClubs),
                  TestCard(5, SuitKind::kSpades),
                  TestCard(13, SuitKind::kClubs)}),
       "trips kicker"},
      {TestCombo(13, SuitKind::kClubs, 12, SuitKind::kHearts),
       TestCombo(13, SuitKind::kDiamonds, 11, SuitKind::kHearts),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(14, SuitKind::kDiamonds),
                  TestCard(13, SuitKind::kSpades),
                  TestCard(2, SuitKind::kClubs),
                  TestCard(7, SuitKind::kDiamonds)}),
       "two pair kicker"},
      {TestCombo(14, SuitKind::kClubs, 12, SuitKind::kHearts),
       TestCombo(14, SuitKind::kSpades, 11, SuitKind::kHearts),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(13, SuitKind::kDiamonds),
                  TestCard(7, SuitKind::kSpades),
                  TestCard(4, SuitKind::kClubs),
                  TestCard(2, SuitKind::kDiamonds)}),
       "pair kicker"},
      {TestCombo(12, SuitKind::kClubs, 9, SuitKind::kHearts),
       TestCombo(11, SuitKind::kClubs, 10, SuitKind::kHearts),
       TestBoard({TestCard(14, SuitKind::kHearts),
                  TestCard(13, SuitKind::kDiamonds),
                  TestCard(7, SuitKind::kSpades),
                  TestCard(4, SuitKind::kClubs),
                  TestCard(2, SuitKind::kDiamonds)}),
       "high card"},
  };

  for (const CompareCase& test_case : cases) {
    CheckCompareMatchesReference(test_case.first, test_case.second,
                                 test_case.board, test_case.label);
  }
}

TEST_CASE("random Cactus comparisons match reference evaluator") {
  std::array<CardId, kDeckCardCount> deck = {};
  for (int i = 0; i < kDeckCardCount; ++i) {
    deck[static_cast<size_t>(i)] = static_cast<CardId>(i);
  }

  std::mt19937 rng(12345);
  for (int i = 0; i < 1000; ++i) {
    CAPTURE(i);
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId first = TestCombo(deck[0], deck[1]);
    const ComboId second = TestCombo(deck[2], deck[3]);
    const CompactPublicState board =
        TestBoard({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CheckCompareMatchesReference(
        first, second, board,
        "random Cactus-Kev comparison should match reference");
  }
}

}  // namespace
}  // namespace poker
