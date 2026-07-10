#include "src/card_abstraction.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "src/card_utils.h"
#include "src/coarse_chance_transitions.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <utility>

namespace poker {
namespace {

struct GeneratedCase {
  StreetKind street = StreetKind::kPreflop;
  Board board;
  ComboId hand = 0;
};

CardId SelectAvailableCard(CardMask blocked, uint16_t choice) {
  std::array<CardId, kDeckCardCount> available = {};
  size_t count = 0;
  for (int id = 0; id < kDeckCardCount; ++id) {
    const CardId card = static_cast<CardId>(id);
    if ((blocked & CardBit(card)) == 0) {
      available[count++] = card;
    }
  }
  RC_ASSERT(count > 0);
  return available[choice % count];
}

GeneratedCase GenerateCase() {
  const int street_index = *rc::gen::inRange(0, 4).as("street");
  const auto choices =
      *rc::gen::arbitrary<std::array<uint16_t, 7>>().as("cards");

  GeneratedCase generated;
  generated.street = static_cast<StreetKind>(street_index);
  CardMask blocked = 0;
  size_t cursor = 0;
  const int board_count = BoardCardsForStreet(generated.street);
  for (int i = 0; i < board_count; ++i) {
    const CardId card = SelectAvailableCard(blocked, choices[cursor++]);
    generated.board.add(card);
    blocked |= CardBit(card);
  }

  const CardId first = SelectAvailableCard(blocked, choices[cursor++]);
  blocked |= CardBit(first);
  const CardId second = SelectAvailableCard(blocked, choices[cursor]);
  generated.hand = CardsToComboId(first, second);
  return generated;
}

Board PermuteBoard(const Board& board,
                   const std::array<uint16_t, kMaxBoardCards>& choices) {
  std::array<CardId, kMaxBoardCards> cards = board.cards;
  size_t cursor = 0;
  for (size_t remaining = board.count; remaining > 1; --remaining) {
    std::swap(cards[remaining - 1], cards[choices[cursor++] % remaining]);
  }

  Board permuted;
  for (uint8_t i = 0; i < board.count; ++i) {
    permuted.add(cards[static_cast<size_t>(i)]);
  }
  return permuted;
}

std::array<SuitKind, 4> SuitPermutation(
    const std::array<uint16_t, 4>& choices) {
  std::array<SuitKind, 4> suits = {
      SuitKind::kHearts,
      SuitKind::kDiamonds,
      SuitKind::kClubs,
      SuitKind::kSpades,
  };
  size_t cursor = 0;
  for (size_t remaining = suits.size(); remaining > 1; --remaining) {
    std::swap(suits[remaining - 1], suits[choices[cursor++] % remaining]);
  }
  return suits;
}

CardId RenameSuit(CardId card, const std::array<SuitKind, 4>& permutation) {
  return MakeCardId(
      RankFromCardId(card),
      permutation[static_cast<size_t>(SuitIndex(SuitFromCardId(card)))]);
}

Board RenameSuits(const Board& board,
                  const std::array<SuitKind, 4>& permutation) {
  Board renamed;
  for (CardId card : board.span()) {
    renamed.add(RenameSuit(card, permutation));
  }
  return renamed;
}

ComboId RenameSuits(ComboId hand,
                    const std::array<SuitKind, 4>& permutation) {
  const ComboInfo& combo = GetComboInfo(hand);
  return CardsToComboId(RenameSuit(combo.card0, permutation),
                        RenameSuit(combo.card1, permutation));
}

Board BoardOf(std::initializer_list<CardId> cards) {
  Board board;
  for (CardId card : cards) {
    board.add(card);
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
    CHECK(board_texture_bucket(NextStreet(street), board_features(child)) ==
          transition.child_bucket);
  }
}

TEST_CASE("board abstraction ignores card order") {
  const bool passed = rc::check("board order invariance", [] {
    const GeneratedCase generated = GenerateCase();
    const auto choices =
        *rc::gen::arbitrary<std::array<uint16_t, kMaxBoardCards>>()
             .as("order");
    const Board permuted = PermuteBoard(generated.board, choices);
    const BoardFeatures features = board_features(generated.board);
    const BoardFeatures permuted_features = board_features(permuted);

    RC_ASSERT(features == permuted_features);
    RC_ASSERT(exact_board_bucket(generated.board) == generated.board.mask);
    RC_ASSERT(exact_board_bucket(generated.board) ==
              exact_board_bucket(permuted));
    RC_ASSERT(board_texture_bucket(generated.street, features) ==
              board_texture_bucket(generated.street, permuted_features));
  });
  CHECK(passed);
}

TEST_CASE("coarse buckets ignore global suit renaming") {
  const bool passed = rc::check("suit invariance", [] {
    const GeneratedCase generated = GenerateCase();
    const auto choices =
        *rc::gen::arbitrary<std::array<uint16_t, 4>>().as("suits");
    const auto permutation = SuitPermutation(choices);
    const Board renamed_board = RenameSuits(generated.board, permutation);
    const ComboId renamed_hand = RenameSuits(generated.hand, permutation);
    const BoardFeatures features = board_features(generated.board);
    const BoardFeatures renamed_features = board_features(renamed_board);

    RC_ASSERT(board_texture_bucket(generated.street, features) ==
              board_texture_bucket(generated.street, renamed_features));
    RC_ASSERT(coarse_private_bucket(generated.hand, generated.street,
                                    features) ==
              coarse_private_bucket(renamed_hand, generated.street,
                                    renamed_features));
  });
  CHECK(passed);
}

TEST_CASE("coarse private buckets stay in their local range") {
  const bool passed = rc::check("private bucket range", [] {
    const GeneratedCase generated = GenerateCase();
    RC_ASSERT(coarse_private_bucket(generated.hand, generated.street,
                                    board_features(generated.board)) < 36);
  });
  CHECK(passed);
}

TEST_CASE("texture buckets classify paired monotone and connected boards") {
  const Board rainbow = BoardOf({MakeCardId(2, SuitKind::kHearts),
                                 MakeCardId(7, SuitKind::kDiamonds),
                                 MakeCardId(11, SuitKind::kClubs)});
  const Board paired = BoardOf({MakeCardId(11, SuitKind::kHearts),
                                MakeCardId(11, SuitKind::kDiamonds),
                                MakeCardId(2, SuitKind::kClubs)});
  const Board monotone = BoardOf({MakeCardId(2, SuitKind::kHearts),
                                  MakeCardId(7, SuitKind::kHearts),
                                  MakeCardId(11, SuitKind::kHearts)});
  const Board connected = BoardOf({MakeCardId(9, SuitKind::kHearts),
                                   MakeCardId(10, SuitKind::kDiamonds),
                                   MakeCardId(11, SuitKind::kClubs)});
  const BoardBucketId baseline =
      board_texture_bucket(StreetKind::kFlop, board_features(rainbow));

  CHECK(board_features(paired).max_rank_count == 2);
  CHECK(board_features(monotone).max_suit_count == 3);
  CHECK(straight_density(board_features(connected).rank_mask) == 3);
  CHECK(board_texture_bucket(StreetKind::kFlop, board_features(paired)) !=
        baseline);
  CHECK(board_texture_bucket(StreetKind::kFlop, board_features(monotone)) !=
        baseline);
  CHECK(board_texture_bucket(StreetKind::kFlop, board_features(connected)) !=
        baseline);
}

TEST_CASE("exact private buckets use exact combo ids") {
  const ComboId aces =
      ExactCombo(14, SuitKind::kSpades, 14, SuitKind::kHearts);

  CHECK(exact_private_bucket(aces) == aces);
  const uint32_t expected_count = kCoarsePrivateBuckets ? 36 : kComboCount;
  CHECK(private_bucket_count(StreetKind::kPreflop) == expected_count);
}

TEST_CASE("coarse private buckets still use exact board within public bucket") {
  const ComboId hand =
      ExactCombo(12, SuitKind::kHearts, 9, SuitKind::kHearts);
  const Board paired_with_hand =
      BoardOf({MakeCardId(12, SuitKind::kSpades),
               MakeCardId(2, SuitKind::kClubs),
               MakeCardId(7, SuitKind::kDiamonds)});
  const Board unpaired_for_hand =
      BoardOf({MakeCardId(11, SuitKind::kSpades),
               MakeCardId(3, SuitKind::kClubs),
               MakeCardId(8, SuitKind::kDiamonds)});

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
