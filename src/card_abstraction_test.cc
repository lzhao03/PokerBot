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
  ExactPublicCardBuckets buckets;
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

  CHECK(buckets.bucket(first.street, BoardFromCompact(first)) ==
        first.board_mask);
  CHECK(buckets.bucket(first.street, BoardFromCompact(first)) ==
        buckets.bucket(reordered.street, BoardFromCompact(reordered)));
}

TEST_CASE("texture public buckets group by board texture") {
  BoardTexturePublicCardBuckets buckets;
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

  CHECK(buckets.bucket(first_flop.street, BoardFromCompact(first_flop)) ==
        buckets.bucket(same_texture_flop.street,
                       BoardFromCompact(same_texture_flop)));
  CHECK(buckets.bucket(first_flop.street, BoardFromCompact(first_flop)) !=
        buckets.bucket(paired_flop.street, BoardFromCompact(paired_flop)));
  CHECK(buckets.bucket(first_flop.street, BoardFromCompact(first_flop)) !=
        buckets.bucket(monotone_flop.street,
                       BoardFromCompact(monotone_flop)));
  CHECK(buckets.bucket(first_flop.street, BoardFromCompact(first_flop)) !=
        buckets.bucket(turn.street, BoardFromCompact(turn)));
}

TEST_CASE("exact private buckets use exact combo ids") {
  ExactPrivateBuckets buckets;
  const CompactPublicState state = PreflopState();
  const ComboId aces =
      ExactCombo(14, SuitKind::kSpades, 14, SuitKind::kHearts);

  CHECK(buckets.bucket(aces, state.street, BoardFromCompact(state)) == aces);
  CHECK(buckets.bucket_count(state.street, BoardFromCompact(state)) ==
        kComboCount);
}

TEST_CASE("coarse private buckets merge equivalent combos") {
  CoarsePrivateBuckets buckets;
  const CompactPublicState state = PreflopState();
  const ComboId ace_king_spades =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades);
  const ComboId ace_king_hearts =
      ExactCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId ace_king_offsuit =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kHearts);

  CHECK(ace_king_spades != ace_king_hearts);
  const Board board = BoardFromCompact(state);
  CHECK(buckets.bucket(ace_king_spades, state.street, board) ==
        buckets.bucket(ace_king_hearts, state.street, board));
  CHECK(buckets.bucket(ace_king_spades, state.street, board) !=
        buckets.bucket(ace_king_offsuit, state.street, board));
  CHECK(buckets.bucket_count(state.street, board) < kComboCount);
}

}  // namespace
}  // namespace poker
