#include "src/card_abstraction.h"
#include "src/card_utils.h"
#include "src/coarse_chance_transitions.h"
#include "src/combo.h"
#include "src/hand_evaluator.h"
#include "src/hand_evaluator_table_builder.h"
#include "src/hand_evaluator_tables.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <utility>

namespace poker {
namespace {

CardId Card(int rank, SuitKind suit) {
  return MakeCardId(rank, suit);
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(Card(first_rank, first_suit),
                        Card(second_rank, second_suit));
}

ComboId Combo(CardId first, CardId second) {
  return CardsToComboId(first, second);
}

Board BoardOf(std::initializer_list<CardId> cards) {
  Board board;
  for (CardId card : cards) {
    board.add(card);
  }
  return board;
}

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

struct GeneratedCase {
  StreetKind street = StreetKind::kPreflop;
  Board board;
  ComboId hand = 0;
};

GeneratedCase GenerateCase() {
  GeneratedCase generated;
  generated.street = static_cast<StreetKind>(
      *rc::gen::inRange(0, 4).as("street"));
  const auto choices =
      *rc::gen::arbitrary<std::array<uint16_t, 7>>().as("cards");

  CardMask blocked = 0;
  size_t cursor = 0;
  for (int i = 0; i < BoardCardsForStreet(generated.street); ++i) {
    const CardId card = SelectAvailableCard(blocked, choices[cursor++]);
    generated.board.add(card);
    blocked |= CardBit(card);
  }

  const CardId first = SelectAvailableCard(blocked, choices[cursor++]);
  blocked |= CardBit(first);
  generated.hand =
      Combo(first, SelectAvailableCard(blocked, choices[cursor]));
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
    permuted.add(cards[i]);
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

CardId RenameSuit(CardId card,
                  const std::array<SuitKind, 4>& permutation) {
  return Card(RankFromCardId(card),
              permutation[static_cast<size_t>(
                  SuitIndex(SuitFromCardId(card)))]);
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
  return Combo(RenameSuit(combo.card0, permutation),
               RenameSuit(combo.card1, permutation));
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
      parent.add(transition.parent_cards[i]);
    }
    CHECK(board_texture_bucket(street, board_features(parent)) ==
          transition.parent_bucket);

    Board child = parent;
    for (uint8_t i = 0; i < transition.card_count; ++i) {
      child.add(transition.cards[i]);
    }
    CHECK(board_texture_bucket(NextStreet(street), board_features(child)) ==
          transition.child_bucket);
  }
}

HandEvaluation ToHandEvaluation(
    const hand_evaluator_generation::EvaluationScore& score) {
  HandEvaluation evaluation;
  evaluation.rank = score.rank;
  evaluation.kickers = score.kickers;
  evaluation.kicker_count = score.kicker_count;
  return evaluation;
}

HandEvaluation ReferenceBestHand(ComboId hand, const Board& board) {
  std::array<CardId, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (CardId card : board.span()) {
    cards[count++] = card;
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
                     const Board& board) {
  const HandEvaluation first_eval = ReferenceBestHand(first, board);
  const HandEvaluation second_eval = ReferenceBestHand(second, board);
  return first_eval > second_eval ? 1 : first_eval < second_eval ? -1 : 0;
}

void CheckCompare(ComboId first,
                  ComboId second,
                  const Board& board,
                  const char* label) {
  CAPTURE(label);
  HandEvaluator evaluator;
  CHECK(evaluator.compare_hands(first, second, board) ==
        ReferenceCompare(first, second, board));
}

TEST_CASE("combo encoding is an exhaustive canonical bijection") {
  ComboId expected = 0;
  for (int first = 0; first < kDeckCardCount; ++first) {
    for (int second = first + 1; second < kDeckCardCount; ++second) {
      const CardId a = static_cast<CardId>(first);
      const CardId b = static_cast<CardId>(second);
      CAPTURE(first);
      CAPTURE(second);

      CHECK(Combo(a, b) == expected);
      CHECK(Combo(b, a) == expected);
      const ComboInfo& info = GetComboInfo(expected);
      CHECK(info.card0 == a);
      CHECK(info.card1 == b);
      CHECK(info.mask == (CardBit(a) | CardBit(b)));
      ++expected;
    }
  }

  CHECK(expected == kComboCount);
  CHECK(!MaybeCardsToComboId(0, 0).has_value());
  CHECK(!MaybeCardsToComboId(static_cast<CardId>(kDeckCardCount), 0)
             .has_value());
}

TEST_CASE("street sampling returns unique unblocked cards") {
  const bool passed = rc::check("valid card sampling", [] {
    const StreetKind street = static_cast<StreetKind>(
        *rc::gen::inRange(0, 4).as("street"));
    const int private_count =
        *rc::gen::inRange(0, 5).as("private count");
    const auto choices =
        *rc::gen::arbitrary<std::array<uint16_t, 9>>().as("cards");
    const uint32_t seed = *rc::gen::arbitrary<uint32_t>().as("seed");

    Board board;
    CardMask blocked = 0;
    size_t cursor = 0;
    for (int i = 0; i < BoardCardsForStreet(street); ++i) {
      const CardId card = SelectAvailableCard(blocked, choices[cursor++]);
      board.add(card);
      blocked |= CardBit(card);
    }

    CardMask private_cards = 0;
    for (int i = 0; i < private_count; ++i) {
      const CardId card = SelectAvailableCard(blocked, choices[cursor++]);
      private_cards |= CardBit(card);
      blocked |= CardBit(card);
    }

    std::mt19937 rng(seed);
    const auto sampled =
        SampleStreetCards(street, board, private_cards, rng);
    RC_ASSERT(sampled.size() ==
              static_cast<size_t>(CardsForNextStreet(street)));
    CardMask seen = 0;
    for (CardId card : sampled) {
      RC_ASSERT((blocked & CardBit(card)) == 0);
      RC_ASSERT((seen & CardBit(card)) == 0);
      seen |= CardBit(card);
    }
  });
  CHECK(passed);
}

TEST_CASE("street sampling handles boundary card counts") {
  const CardId legal_card = Card(2, SuitKind::kClubs);
  CardMask blocked = 0;
  for (int id = 0; id < kDeckCardCount; ++id) {
    const CardId card = static_cast<CardId>(id);
    if (card != legal_card) {
      blocked |= CardBit(card);
    }
  }

  std::mt19937 rng(12345);
  const auto sampled =
      SampleStreetCards(StreetKind::kFlop, Board{}, blocked, rng);
  REQUIRE(sampled.size() == 1);
  CHECK(sampled[0] == legal_card);

  blocked = 0;
  for (int id = 0; id < kDeckCardCount - 2; ++id) {
    blocked |= CardBit(static_cast<CardId>(id));
  }
  CHECK_THROWS_AS(
      SampleStreetCards(StreetKind::kPreflop, Board{}, blocked, rng),
      std::runtime_error);
}

TEST_CASE("card abstraction is order- and suit-invariant") {
  const bool passed = rc::check("card abstraction invariants", [] {
    const GeneratedCase generated = GenerateCase();
    const BoardFeatures features = board_features(generated.board);

    const auto order =
        *rc::gen::arbitrary<std::array<uint16_t, kMaxBoardCards>>()
             .as("order");
    const Board permuted = PermuteBoard(generated.board, order);
    const BoardFeatures permuted_features = board_features(permuted);
    RC_ASSERT(features == permuted_features);
    RC_ASSERT(exact_board_bucket(generated.board) == generated.board.mask);
    RC_ASSERT(exact_board_bucket(generated.board) ==
              exact_board_bucket(permuted));
    RC_ASSERT(board_texture_bucket(generated.street, features) ==
              board_texture_bucket(generated.street, permuted_features));

    const auto suit_choices =
        *rc::gen::arbitrary<std::array<uint16_t, 4>>().as("suits");
    const auto suits = SuitPermutation(suit_choices);
    const Board renamed_board = RenameSuits(generated.board, suits);
    const ComboId renamed_hand = RenameSuits(generated.hand, suits);
    const BoardFeatures renamed_features = board_features(renamed_board);
    RC_ASSERT(board_texture_bucket(generated.street, features) ==
              board_texture_bucket(generated.street, renamed_features));
    RC_ASSERT(coarse_private_bucket(generated.hand, generated.street,
                                    features) ==
              coarse_private_bucket(renamed_hand, generated.street,
                                    renamed_features));
    RC_ASSERT(coarse_private_bucket(generated.hand, generated.street,
                                    features) < 36);
  });
  CHECK(passed);
}

TEST_CASE("texture and private buckets preserve intended distinctions") {
  const Board rainbow = BoardOf({Card(2, SuitKind::kHearts),
                                 Card(7, SuitKind::kDiamonds),
                                 Card(11, SuitKind::kClubs)});
  const Board paired = BoardOf({Card(11, SuitKind::kHearts),
                                Card(11, SuitKind::kDiamonds),
                                Card(2, SuitKind::kClubs)});
  const Board monotone = BoardOf({Card(2, SuitKind::kHearts),
                                  Card(7, SuitKind::kHearts),
                                  Card(11, SuitKind::kHearts)});
  const Board connected = BoardOf({Card(9, SuitKind::kHearts),
                                   Card(10, SuitKind::kDiamonds),
                                   Card(11, SuitKind::kClubs)});
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

  const ComboId hand =
      Combo(12, SuitKind::kHearts, 9, SuitKind::kHearts);
  const Board paired_with_hand = BoardOf({Card(12, SuitKind::kSpades),
                                          Card(2, SuitKind::kClubs),
                                          Card(7, SuitKind::kDiamonds)});
  const Board unpaired_for_hand = BoardOf({Card(11, SuitKind::kSpades),
                                           Card(3, SuitKind::kClubs),
                                           Card(8, SuitKind::kDiamonds)});
  CHECK(board_texture_bucket(StreetKind::kFlop,
                             board_features(paired_with_hand)) ==
        board_texture_bucket(StreetKind::kFlop,
                             board_features(unpaired_for_hand)));
  CHECK(coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(paired_with_hand)) !=
        coarse_private_bucket(hand, StreetKind::kFlop,
                              board_features(unpaired_for_hand)));

  const ComboId aces =
      Combo(14, SuitKind::kSpades, 14, SuitKind::kHearts);
  CHECK(exact_private_bucket(aces) == aces);
  CHECK(private_bucket_count(StreetKind::kPreflop) ==
        (kCoarsePrivateBuckets ? 36u : static_cast<uint32_t>(kComboCount)));
}

TEST_CASE("generated texture transitions match their representatives") {
  CheckTextureTransitions(StreetKind::kPreflop,
                          kPreflopTextureTransitions);
  CheckTextureTransitions(StreetKind::kFlop, kFlopTextureTransitions);
  CheckTextureTransitions(StreetKind::kTurn, kTurnTextureTransitions);
}

TEST_CASE("generated evaluator tables match the reference builder [slow]") {
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

TEST_CASE("wheel straights are five-high") {
  HandEvaluator evaluator;
  const std::array<CardId, 5> wheel = {
      Card(14, SuitKind::kHearts), Card(5, SuitKind::kDiamonds),
      Card(4, SuitKind::kClubs), Card(3, SuitKind::kSpades),
      Card(2, SuitKind::kHearts),
  };
  const std::array<CardId, 5> six_high = {
      Card(6, SuitKind::kHearts), Card(5, SuitKind::kDiamonds),
      Card(4, SuitKind::kClubs), Card(3, SuitKind::kSpades),
      Card(2, SuitKind::kHearts),
  };

  const HandEvaluation wheel_eval = evaluator.evaluate(wheel);
  CHECK(wheel_eval.rank == HandRank::STRAIGHT);
  CHECK(wheel_eval.kicker_count == 1);
  CHECK(wheel_eval.kickers[0] == 5);
  CHECK(evaluator.evaluate(six_high) > wheel_eval);

  const std::array<CardId, 5> wheel_flush = {
      Card(14, SuitKind::kHearts), Card(5, SuitKind::kHearts),
      Card(4, SuitKind::kHearts), Card(3, SuitKind::kHearts),
      Card(2, SuitKind::kHearts),
  };
  const HandEvaluation wheel_flush_eval = evaluator.evaluate(wheel_flush);
  CHECK(wheel_flush_eval.rank == HandRank::STRAIGHT_FLUSH);
  CHECK(wheel_flush_eval.kickers[0] == 5);
}

TEST_CASE("seven-card comparisons match an independent reference") {
  struct CompareCase {
    ComboId first;
    ComboId second;
    Board board;
    const char* label;
  };
  const CompareCase rare_cases[] = {
      {Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
       Combo(9, SuitKind::kHearts, 8, SuitKind::kHearts),
       BoardOf({Card(10, SuitKind::kHearts), Card(11, SuitKind::kHearts),
                Card(12, SuitKind::kHearts), Card(2, SuitKind::kClubs),
                Card(3, SuitKind::kDiamonds)}),
       "straight flush"},
      {Combo(14, SuitKind::kSpades, 13, SuitKind::kSpades),
       Combo(13, SuitKind::kHearts, 13, SuitKind::kDiamonds),
       BoardOf({Card(14, SuitKind::kHearts), Card(14, SuitKind::kDiamonds),
                Card(14, SuitKind::kClubs), Card(2, SuitKind::kClubs),
                Card(7, SuitKind::kDiamonds)}),
       "quads"},
      {Combo(13, SuitKind::kHearts, 9, SuitKind::kHearts),
       Combo(11, SuitKind::kHearts, 10, SuitKind::kHearts),
       BoardOf({Card(14, SuitKind::kHearts), Card(2, SuitKind::kHearts),
                Card(7, SuitKind::kHearts), Card(12, SuitKind::kDiamonds),
                Card(3, SuitKind::kClubs)}),
       "flush kicker"},
      {Combo(13, SuitKind::kClubs, 12, SuitKind::kHearts),
       Combo(13, SuitKind::kDiamonds, 11, SuitKind::kHearts),
       BoardOf({Card(14, SuitKind::kHearts), Card(14, SuitKind::kDiamonds),
                Card(13, SuitKind::kSpades), Card(2, SuitKind::kClubs),
                Card(7, SuitKind::kDiamonds)}),
       "two-pair kicker"},
  };
  for (const CompareCase& test : rare_cases) {
    CheckCompare(test.first, test.second, test.board, test.label);
  }

  std::array<CardId, kDeckCardCount> deck = {};
  for (int i = 0; i < kDeckCardCount; ++i) {
    deck[i] = static_cast<CardId>(i);
  }
  std::mt19937 rng(12345);
  for (int i = 0; i < 1000; ++i) {
    CAPTURE(i);
    std::shuffle(deck.begin(), deck.end(), rng);
    CheckCompare(Combo(deck[0], deck[1]), Combo(deck[2], deck[3]),
                 BoardOf({deck[4], deck[5], deck[6], deck[7], deck[8]}),
                 "deterministic random comparison");
  }
}

}  // namespace
}  // namespace poker
