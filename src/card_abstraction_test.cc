#include "src/card_abstraction.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

CompactPublicState PublicState(StreetKind street,
                               CardId first,
                               CardId second,
                               CardId third,
                               CardId fourth = 0) {
  CompactPublicState state;
  state.street = street;
  AddBoardCard(state, first);
  AddBoardCard(state, second);
  AddBoardCard(state, third);
  if (street == StreetKind::kTurn || street == StreetKind::kRiver) {
    AddBoardCard(state, fourth);
  }
  return state;
}

CompactPublicState PreflopState() {
  CompactPublicState state;
  state.street = StreetKind::kPreflop;
  return state;
}

ComboId ExactCombo(int first_rank,
                   SuitKind first_suit,
                   int second_rank,
                   SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

TEST_CASE("exact public buckets use the board mask") {
  const CompactPublicState first =
      PublicState(StreetKind::kFlop,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs));
  const CompactPublicState reordered =
      PublicState(StreetKind::kFlop,
                  MakeCardId(11, SuitKind::kClubs),
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds));

  CHECK(exact_public_bucket(BoardFromCompact(first)) == first.board_mask);
  CHECK(exact_public_bucket(BoardFromCompact(first)) ==
        exact_public_bucket(BoardFromCompact(reordered)));
}

TEST_CASE("texture public buckets group by board texture") {
  const CompactPublicState first_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs));
  const CompactPublicState same_texture_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(3, SuitKind::kHearts),
                  MakeCardId(8, SuitKind::kDiamonds),
                  MakeCardId(12, SuitKind::kClubs));
  const CompactPublicState paired_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(2, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs));
  const CompactPublicState monotone_flop =
      PublicState(StreetKind::kFlop,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kHearts),
                  MakeCardId(11, SuitKind::kHearts));
  const CompactPublicState turn =
      PublicState(StreetKind::kTurn,
                  MakeCardId(2, SuitKind::kHearts),
                  MakeCardId(7, SuitKind::kDiamonds),
                  MakeCardId(11, SuitKind::kClubs),
                  MakeCardId(14, SuitKind::kSpades));

  const Board first = BoardFromCompact(first_flop);
  const Board same_texture = BoardFromCompact(same_texture_flop);
  const Board paired = BoardFromCompact(paired_flop);
  const Board monotone = BoardFromCompact(monotone_flop);
  const Board turn_board = BoardFromCompact(turn);
  CHECK(board_texture_public_bucket(first_flop.street,
                                    board_features(first)) ==
        board_texture_public_bucket(same_texture_flop.street,
                                    board_features(same_texture)));
  CHECK(board_texture_public_bucket(first_flop.street,
                                    board_features(first)) !=
        board_texture_public_bucket(paired_flop.street,
                                    board_features(paired)));
  CHECK(board_texture_public_bucket(first_flop.street,
                                    board_features(first)) !=
        board_texture_public_bucket(monotone_flop.street,
                                    board_features(monotone)));
  CHECK(board_texture_public_bucket(first_flop.street,
                                    board_features(first)) !=
        board_texture_public_bucket(turn.street, board_features(turn_board)));
}

TEST_CASE("exact private buckets use exact combo ids") {
  const CompactPublicState state = PreflopState();
  const ComboId aces =
      ExactCombo(14, SuitKind::kSpades, 14, SuitKind::kHearts);

  CHECK(exact_private_bucket(aces) == aces);
  const uint32_t expected_count = kCoarsePrivateBuckets ? 36 : kComboCount;
  CHECK(private_bucket_count(state.street) == expected_count);
}

TEST_CASE("coarse private buckets merge equivalent combos") {
  const CompactPublicState state = PreflopState();
  const ComboId ace_king_spades =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades);
  const ComboId ace_king_hearts =
      ExactCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId ace_king_offsuit =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kHearts);

  CHECK(ace_king_spades != ace_king_hearts);
  const Board board = BoardFromCompact(state);
  const BoardFeatures features = board_features(board);
  CHECK(coarse_private_bucket(ace_king_spades, state.street, features) ==
        coarse_private_bucket(ace_king_hearts, state.street, features));
  CHECK(coarse_private_bucket(ace_king_spades, state.street, features) !=
        coarse_private_bucket(ace_king_offsuit, state.street, features));
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

  CHECK(board_texture_public_bucket(
            StreetKind::kFlop, board_features(paired_with_hand)) ==
        board_texture_public_bucket(
            StreetKind::kFlop, board_features(unpaired_for_hand)));
  CHECK(coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(paired_with_hand)) !=
        coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(unpaired_for_hand)));
}

}  // namespace
}  // namespace poker
