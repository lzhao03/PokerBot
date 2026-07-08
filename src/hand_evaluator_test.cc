#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

CardId TestCard(int rank, SuitKind suit) {
  return MakeCardId(rank, suit);
}

ComboId TestCombo(int first_rank, SuitKind first_suit, int second_rank,
                  SuitKind second_suit) {
  return CardsToComboId(TestCard(first_rank, first_suit),
                        TestCard(second_rank, second_suit));
}

ComboId TestCombo(CardId first, CardId second) {
  return CardsToComboId(first, second);
}

GameState TestBoard(std::initializer_list<CardId> cards) {
  GameState board;
  for (CardId card : cards) {
    AddBoardCard(board, card);
  }
  return board;
}

HandEvaluation ReferenceBestHandFromFiveCardEvaluation(
    const HandEvaluator& evaluator, ComboId hand, const GameState& board) {
  std::array<CardId, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (CardId card : board.board_cards) {
    cards[count] = card;
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
            const HandEvaluation current = evaluator.evaluate(subset);
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

int ReferenceCompare(const HandEvaluator& evaluator,
                     ComboId first,
                     ComboId second,
                     const GameState& board) {
  const HandEvaluation first_eval =
      ReferenceBestHandFromFiveCardEvaluation(evaluator, first, board);
  const HandEvaluation second_eval =
      ReferenceBestHandFromFiveCardEvaluation(evaluator, second, board);
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
                                  const GameState& board,
                                  const char* message) {
  HandEvaluator evaluator;
  Expect(evaluator.compare_hands(first, second, board) ==
             ReferenceCompare(evaluator, first, second, board),
         message);
}

void CheckFiveCardEvaluation() {
  const std::array<CardId, 5> royal_flush = {
      TestCard(10, SuitKind::kHearts),
      TestCard(11, SuitKind::kHearts),
      TestCard(12, SuitKind::kHearts),
      TestCard(13, SuitKind::kHearts),
      TestCard(14, SuitKind::kHearts),
  };

  HandEvaluator evaluator;
  HandEvaluation evaluation = evaluator.evaluate(royal_flush);
  Expect(evaluation.rank == HandRank::ROYAL_FLUSH,
         "five-card royal flush should rank as royal flush");
}

void CheckWheelStraightIsFiveHigh() {
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
  Expect(wheel_eval.rank == HandRank::STRAIGHT,
         "wheel should rank as straight");
  Expect(wheel_eval.kicker_count == 1 && wheel_eval.kickers[0] == 5,
         "wheel straight should be five-high");
  Expect(evaluator.evaluate(six_high) > wheel_eval,
         "six-high straight should beat wheel straight");

  const std::array<CardId, 5> wheel_flush = {
      TestCard(14, SuitKind::kHearts),
      TestCard(5, SuitKind::kHearts),
      TestCard(4, SuitKind::kHearts),
      TestCard(3, SuitKind::kHearts),
      TestCard(2, SuitKind::kHearts),
  };
  HandEvaluation wheel_flush_eval = evaluator.evaluate(wheel_flush);
  Expect(wheel_flush_eval.rank == HandRank::STRAIGHT_FLUSH,
         "suited wheel should rank as straight flush");
  Expect(wheel_flush_eval.kicker_count == 1 &&
             wheel_flush_eval.kickers[0] == 5,
         "wheel straight flush should be five-high");
}

void CheckSevenCardBestHand() {
  ComboId hand = TestCombo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
  GameState board;
  AddBoardCard(board, TestCard(14, SuitKind::kDiamonds));
  AddBoardCard(board, TestCard(13, SuitKind::kHearts));
  AddBoardCard(board, TestCard(13, SuitKind::kClubs));
  AddBoardCard(board, TestCard(2, SuitKind::kSpades));
  AddBoardCard(board, TestCard(7, SuitKind::kHearts));

  HandEvaluator evaluator;
  HandEvaluation evaluation = evaluator.evaluate_hand(hand, board);
  Expect(evaluation == ReferenceBestHandFromFiveCardEvaluation(evaluator, hand,
                                                              board),
         "seven-card evaluation should match five-card reference");
  Expect(evaluation.rank == HandRank::FULL_HOUSE,
         "seven-card evaluation should choose full house");
  Expect(evaluation.kicker_count == 2, "full house should have two kickers");
  Expect(evaluation.kickers[0] == 14 && evaluation.kickers[1] == 13,
         "full house should be aces full of kings");
}

void CheckCactusWheelOrdering() {
  HandEvaluator evaluator;
  const GameState straight_board =
      TestBoard({TestCard(14, SuitKind::kClubs),
                 TestCard(5, SuitKind::kDiamonds),
                 TestCard(4, SuitKind::kHearts),
                 TestCard(3, SuitKind::kSpades),
                 TestCard(9, SuitKind::kClubs)});
  const ComboId wheel =
      TestCombo(2, SuitKind::kHearts, 13, SuitKind::kClubs);
  const ComboId six_high =
      TestCombo(6, SuitKind::kHearts, 2, SuitKind::kClubs);
  Expect(evaluator.compare_hands(six_high, wheel, straight_board) > 0,
         "Cactus compare should score wheel lower than six-high straight");
  const HandEvaluation wheel_eval = evaluator.evaluate_hand(wheel,
                                                            straight_board);
  Expect(wheel_eval.rank == HandRank::STRAIGHT &&
             wheel_eval.kicker_count == 1 && wheel_eval.kickers[0] == 5,
         "seven-card wheel should evaluate as five-high straight");
}

void CheckCompactAndGameStateComparisonMatch() {
  ComboId aces = TestCombo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
  ComboId kings = TestCombo(13, SuitKind::kHearts, 13, SuitKind::kSpades);

  GameState game_state;
  AddBoardCard(game_state, TestCard(2, SuitKind::kDiamonds));
  AddBoardCard(game_state, TestCard(7, SuitKind::kClubs));
  AddBoardCard(game_state, TestCard(9, SuitKind::kHearts));
  AddBoardCard(game_state, TestCard(11, SuitKind::kSpades));
  AddBoardCard(game_state, TestCard(12, SuitKind::kClubs));

  CompactPublicState compact_state;
  for (CardId card : game_state.board_cards) {
    AddBoardCard(compact_state, card);
  }

  HandEvaluator evaluator;
  Expect(evaluator.compare_hands(aces, kings, game_state) ==
             evaluator.compare_hands(aces, kings, compact_state),
         "compact and GameState showdown comparison should match");
  Expect(evaluator.compare_hands(aces, kings, game_state) ==
             evaluator.compare_hands(aces, kings, compact_state.board_cards,
                                     compact_state.board_count),
         "explicit-board showdown comparison should match GameState");
}

void CheckCactusCompareRepresentativeParity() {
  CheckCompareMatchesReference(
      TestCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
      TestCombo(9, SuitKind::kHearts, 8, SuitKind::kHearts),
      TestBoard({TestCard(10, SuitKind::kHearts),
                 TestCard(11, SuitKind::kHearts),
                 TestCard(12, SuitKind::kHearts),
                 TestCard(2, SuitKind::kClubs),
                 TestCard(3, SuitKind::kDiamonds)}),
      "royal and straight flush comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades),
      TestCombo(13, SuitKind::kHearts, 13, SuitKind::kDiamonds),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(14, SuitKind::kDiamonds),
                 TestCard(14, SuitKind::kClubs),
                 TestCard(2, SuitKind::kClubs),
                 TestCard(7, SuitKind::kDiamonds)}),
      "quads comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(14, SuitKind::kClubs, 13, SuitKind::kDiamonds),
      TestCombo(13, SuitKind::kHearts, 13, SuitKind::kSpades),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(14, SuitKind::kDiamonds),
                 TestCard(13, SuitKind::kClubs),
                 TestCard(2, SuitKind::kSpades),
                 TestCard(7, SuitKind::kHearts)}),
      "full house comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(13, SuitKind::kHearts, 9, SuitKind::kHearts),
      TestCombo(11, SuitKind::kHearts, 10, SuitKind::kHearts),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(2, SuitKind::kHearts),
                 TestCard(7, SuitKind::kHearts),
                 TestCard(12, SuitKind::kDiamonds),
                 TestCard(3, SuitKind::kClubs)}),
      "flush comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(14, SuitKind::kHearts, 5, SuitKind::kDiamonds),
      TestCombo(5, SuitKind::kSpades, 6, SuitKind::kDiamonds),
      TestBoard({TestCard(2, SuitKind::kClubs),
                 TestCard(3, SuitKind::kDiamonds),
                 TestCard(4, SuitKind::kHearts),
                 TestCard(9, SuitKind::kSpades),
                 TestCard(13, SuitKind::kClubs)}),
      "wheel straight comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(9, SuitKind::kSpades, 14, SuitKind::kHearts),
      TestCombo(9, SuitKind::kClubs, 12, SuitKind::kHearts),
      TestBoard({TestCard(9, SuitKind::kHearts),
                 TestCard(9, SuitKind::kDiamonds),
                 TestCard(2, SuitKind::kClubs),
                 TestCard(5, SuitKind::kSpades),
                 TestCard(13, SuitKind::kClubs)}),
      "trips kicker comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(13, SuitKind::kClubs, 12, SuitKind::kHearts),
      TestCombo(13, SuitKind::kDiamonds, 11, SuitKind::kHearts),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(14, SuitKind::kDiamonds),
                 TestCard(13, SuitKind::kSpades),
                 TestCard(2, SuitKind::kClubs),
                 TestCard(7, SuitKind::kDiamonds)}),
      "two pair kicker comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(14, SuitKind::kClubs, 12, SuitKind::kHearts),
      TestCombo(14, SuitKind::kSpades, 11, SuitKind::kHearts),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(13, SuitKind::kDiamonds),
                 TestCard(7, SuitKind::kSpades),
                 TestCard(4, SuitKind::kClubs),
                 TestCard(2, SuitKind::kDiamonds)}),
      "pair kicker comparison should match reference");

  CheckCompareMatchesReference(
      TestCombo(12, SuitKind::kClubs, 9, SuitKind::kHearts),
      TestCombo(11, SuitKind::kClubs, 10, SuitKind::kHearts),
      TestBoard({TestCard(14, SuitKind::kHearts),
                 TestCard(13, SuitKind::kDiamonds),
                 TestCard(7, SuitKind::kSpades),
                 TestCard(4, SuitKind::kClubs),
                 TestCard(2, SuitKind::kDiamonds)}),
      "high card comparison should match reference");
}

void CheckCactusCompareRandomParity() {
  std::array<CardId, kDeckCardCount> deck = {};
  for (int i = 0; i < kDeckCardCount; ++i) {
    deck[static_cast<size_t>(i)] = static_cast<CardId>(i);
  }

  std::mt19937 rng(12345);
  for (int i = 0; i < 1000; ++i) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId first = TestCombo(deck[0], deck[1]);
    const ComboId second = TestCombo(deck[2], deck[3]);
    const GameState board =
        TestBoard({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CheckCompareMatchesReference(
        first, second, board,
        "random Cactus-Kev comparison should match reference");
  }
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckFiveCardEvaluation();
  poker::CheckWheelStraightIsFiveHigh();
  poker::CheckSevenCardBestHand();
  poker::CheckCactusWheelOrdering();
  poker::CheckCompactAndGameStateComparisonMatch();
  poker::CheckCactusCompareRepresentativeParity();
  poker::CheckCactusCompareRandomParity();
  return 0;
}
