#include "src/card_abstraction.h"

#include "doctest/doctest.h"
#include "src/coarse_chance_transitions.h"

namespace poker {
namespace {

Board TestBoard(StreetKind street,
                CardId first,
                CardId second,
                CardId third,
                CardId fourth = 0) {
  Board board;
  board.add(first);
  board.add(second);
  board.add(third);
  if (street == StreetKind::kTurn || street == StreetKind::kRiver) {
    board.add(fourth);
  }
  return board;
}

ComboId ExactCombo(int first_rank,
                   SuitKind first_suit,
                   int second_rank,
                   SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

StreetKind NextStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return StreetKind::kFlop;
    case StreetKind::kFlop:
      return StreetKind::kTurn;
    case StreetKind::kTurn:
      return StreetKind::kRiver;
    case StreetKind::kRiver:
      return StreetKind::kRiver;
  }
}

template <size_t N>
void CheckTextureTransitions(
    StreetKind street,
    const std::array<CoarseChanceTransition, N>& transitions) {
  for (const CoarseChanceTransition& transition : transitions) {
    Board parent;
    for (uint8_t i = 0; i < transition.parent_count; ++i) {
      parent.add(transition.parent_cards[static_cast<size_t>(i)]);
    }
    CHECK(board_texture_bucket(street, board_features(parent)) ==
          transition.parent_bucket);

    Board child = parent;
    for (uint8_t i = 0; i < transition.card_count; ++i) {
      child.add(transition.cards[static_cast<size_t>(i)]);
    }
    CHECK(board_texture_bucket(NextStreet(street),
                                      board_features(child)) ==
          transition.child_bucket);
  }
}

TEST_CASE("exact public buckets use the board mask") {
  const Board first = TestBoard(
      StreetKind::kFlop, MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs));
  const Board reordered = TestBoard(
      StreetKind::kFlop, MakeCardId(11, SuitKind::kClubs),
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds));

  CHECK(exact_board_bucket(first) == first.mask);
  CHECK(exact_board_bucket(first) == exact_board_bucket(reordered));
}

TEST_CASE("texture public buckets group by board texture") {
  const Board first = TestBoard(
      StreetKind::kFlop, MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs));
  const Board same_texture = TestBoard(
      StreetKind::kFlop, MakeCardId(3, SuitKind::kHearts),
      MakeCardId(8, SuitKind::kDiamonds),
      MakeCardId(12, SuitKind::kClubs));
  const Board paired = TestBoard(
      StreetKind::kFlop, MakeCardId(2, SuitKind::kHearts),
      MakeCardId(2, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs));
  const Board monotone = TestBoard(
      StreetKind::kFlop, MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kHearts),
      MakeCardId(11, SuitKind::kHearts));
  const Board turn = TestBoard(
      StreetKind::kTurn, MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(11, SuitKind::kClubs),
      MakeCardId(14, SuitKind::kSpades));

  CHECK(board_texture_bucket(StreetKind::kFlop,
                                    board_features(first)) ==
        board_texture_bucket(StreetKind::kFlop,
                                    board_features(same_texture)));
  CHECK(board_texture_bucket(StreetKind::kFlop,
                                    board_features(first)) !=
        board_texture_bucket(StreetKind::kFlop,
                                    board_features(paired)));
  CHECK(board_texture_bucket(StreetKind::kFlop,
                                    board_features(first)) !=
        board_texture_bucket(StreetKind::kFlop,
                                    board_features(monotone)));
  CHECK(board_texture_bucket(StreetKind::kFlop,
                                    board_features(first)) !=
        board_texture_bucket(StreetKind::kTurn,
                                    board_features(turn)));
}

TEST_CASE("exact private buckets use exact combo ids") {
  const ComboId aces =
      ExactCombo(14, SuitKind::kSpades, 14, SuitKind::kHearts);

  CHECK(exact_private_bucket(aces) == aces);
  const uint32_t expected_count = kCoarsePrivateBuckets ? 36 : kComboCount;
  CHECK(private_bucket_count(StreetKind::kPreflop) == expected_count);
}

TEST_CASE("coarse private buckets merge equivalent combos") {
  const ComboId ace_king_spades =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades);
  const ComboId ace_king_hearts =
      ExactCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId ace_king_offsuit =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kHearts);

  CHECK(ace_king_spades != ace_king_hearts);
  const Board board;
  const BoardFeatures features = board_features(board);
  CHECK(coarse_private_bucket(ace_king_spades, StreetKind::kPreflop,
                              features) ==
        coarse_private_bucket(ace_king_hearts, StreetKind::kPreflop,
                              features));
  CHECK(coarse_private_bucket(ace_king_spades, StreetKind::kPreflop,
                              features) !=
        coarse_private_bucket(ace_king_offsuit, StreetKind::kPreflop,
                              features));
}

TEST_CASE("coarse private bucket ids are local to street") {
  Board board;
  board.add(MakeCardId(14, SuitKind::kDiamonds));
  board.add(MakeCardId(2, SuitKind::kClubs));
  board.add(MakeCardId(7, SuitKind::kSpades));

  const ComboId hand =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades);
  CHECK(coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(board)) < 36);

  board.add(MakeCardId(9, SuitKind::kHearts));
  CHECK(coarse_private_bucket(hand, StreetKind::kTurn,
                              board_features(board)) < 36);

  board.add(MakeCardId(3, SuitKind::kDiamonds));
  CHECK(coarse_private_bucket(hand, StreetKind::kRiver,
                              board_features(board)) < 36);
}

TEST_CASE("coarse private buckets still use exact board within public bucket") {
  const ComboId hand =
      ExactCombo(12, SuitKind::kHearts, 9, SuitKind::kHearts);

  Board paired_with_hand;
  paired_with_hand.add(MakeCardId(12, SuitKind::kSpades));
  paired_with_hand.add(MakeCardId(2, SuitKind::kClubs));
  paired_with_hand.add(MakeCardId(7, SuitKind::kDiamonds));

  Board unpaired_for_hand;
  unpaired_for_hand.add(MakeCardId(11, SuitKind::kSpades));
  unpaired_for_hand.add(MakeCardId(3, SuitKind::kClubs));
  unpaired_for_hand.add(MakeCardId(8, SuitKind::kDiamonds));

  CHECK(board_texture_bucket(
            StreetKind::kFlop, board_features(paired_with_hand)) ==
        board_texture_bucket(
            StreetKind::kFlop, board_features(unpaired_for_hand)));
  CHECK(coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(paired_with_hand)) !=
        coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(unpaired_for_hand)));
}

TEST_CASE("generated texture transition tables match their representatives") {
  CheckTextureTransitions(StreetKind::kPreflop, kPreflopTextureTransitions);
  CheckTextureTransitions(StreetKind::kFlop, kFlopTextureTransitions);
  CheckTextureTransitions(StreetKind::kTurn, kTurnTextureTransitions);
}

}  // namespace
}  // namespace poker
