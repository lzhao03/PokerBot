#include "src/card_abstraction.h"

#include <cstdlib>
#include <iostream>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

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

void CheckExactPublicBucketsUseBoardMask() {
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

  Expect(buckets.bucket(first) == first.board_mask,
         "exact public buckets should use the board mask");
  Expect(buckets.bucket(first) == buckets.bucket(reordered),
         "exact public buckets should be order independent");
}

void CheckTextureBucketsUseBoardTexture() {
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

  Expect(buckets.bucket(first_flop) == buckets.bucket(same_texture_flop),
         "texture buckets should merge similar disconnected rainbow flops");
  Expect(buckets.bucket(first_flop) != buckets.bucket(paired_flop),
         "texture buckets should split paired flops");
  Expect(buckets.bucket(first_flop) != buckets.bucket(monotone_flop),
         "texture buckets should split monotone flops");
  Expect(buckets.bucket(first_flop) != buckets.bucket(turn),
         "texture buckets should include the street");
}

void CheckExactPrivateBucketsUseComboId() {
  ExactPrivateBuckets buckets;
  const CompactPublicState state = PreflopState();
  const ComboId aces =
      ExactCombo(14, SuitKind::kSpades, 14, SuitKind::kHearts);

  Expect(buckets.bucket(aces, state) == aces,
         "exact private buckets should use exact combo ids");
  Expect(buckets.bucket_count(state) == kComboCount,
         "exact private bucket count should equal exact combo count");
}

#if POKER_COARSE_PUBLIC_BUCKETS
void CheckCoarsePrivateBucketsMergeCombos() {
  CoarsePrivateBuckets buckets;
  const CompactPublicState state = PreflopState();
  const ComboId ace_king_spades =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kSpades);
  const ComboId ace_king_hearts =
      ExactCombo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId ace_king_offsuit =
      ExactCombo(14, SuitKind::kSpades, 13, SuitKind::kHearts);

  Expect(ace_king_spades != ace_king_hearts,
         "fixture should use distinct exact combos");
  Expect(buckets.bucket(ace_king_spades, state) ==
             buckets.bucket(ace_king_hearts, state),
         "coarse private buckets should merge equivalent suited combos");
  Expect(buckets.bucket(ace_king_spades, state) !=
             buckets.bucket(ace_king_offsuit, state),
         "coarse private buckets should keep suitedness shape");
  Expect(buckets.bucket_count(state) < kComboCount,
         "coarse private bucket count should be smaller than exact combos");
}
#endif

}  // namespace
}  // namespace poker

int main() {
  poker::CheckExactPublicBucketsUseBoardMask();
  poker::CheckTextureBucketsUseBoardTexture();
  poker::CheckExactPrivateBucketsUseComboId();
#if POKER_COARSE_PUBLIC_BUCKETS
  poker::CheckCoarsePrivateBucketsMergeCombos();
#endif
  return 0;
}
