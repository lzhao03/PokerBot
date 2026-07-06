#include "src/hand_evaluator.h"

#include <array>
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
  Expect(evaluation.rank == HandRank::FULL_HOUSE,
         "seven-card evaluation should choose full house");
  Expect(evaluation.kicker_count == 2, "full house should have two kickers");
  Expect(evaluation.kickers[0] == 14 && evaluation.kickers[1] == 13,
         "full house should be aces full of kings");
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
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckFiveCardEvaluation();
  poker::CheckSevenCardBestHand();
  poker::CheckCompactAndGameStateComparisonMatch();
  return 0;
}
