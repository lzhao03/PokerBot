#include "src/card_abstraction.h"
#include "src/hand_evaluator.h"
#include "src/poker.h"
#include "tools/hand_evaluator_table_builder.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <random>
#include <stdexcept>

namespace poker {
namespace {

using S = SuitKind;

Card C(int rank, S suit) { return MakeCardId(rank, suit); }
ComboId H(Card a, Card b) { return CardsToComboId(a, b); }

BoardRunout Runout(absl::Span<const Card> cards) {
  BoardRunout runout = BoardRunout::Preflop();
  if (cards.empty()) {
    return runout;
  }
  runout.deal_flop(absl::Span<const Card>(cards.data(), 3));
  if (cards.size() >= 4) {
    runout.deal_turn(cards[3]);
  }
  if (cards.size() == 5) {
    runout.deal_river(cards[4]);
  }
  return runout;
}

BoardRunout B(std::initializer_list<Card> cards) {
  return Runout(absl::Span<const Card>(cards.begin(), cards.size()));
}

HandEvaluation Score5(const std::array<Card, 5>& cards) {
  const auto score = hand_evaluator_generation::EvaluateFiveCardScore(cards);
  HandEvaluation out;
  out.rank = score.rank;
  out.kickers = score.kickers;
  out.kicker_count = score.kicker_count;
  return out;
}

HandEvaluation ReferenceBest(ComboId hand, const BoardRunout& board) {
  std::array<Card, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (Card card : board.cards()) cards[count++] = card;

  HandEvaluation best;
  for (size_t a = 0; a + 4 < count; ++a)
    for (size_t b = a + 1; b + 3 < count; ++b)
      for (size_t c = b + 1; c + 2 < count; ++c)
        for (size_t d = c + 1; d + 1 < count; ++d)
          for (size_t e = d + 1; e < count; ++e)
            best = std::max(best, Score5({cards[a], cards[b], cards[c],
                                          cards[d], cards[e]}));
  return best;
}

int ReferenceCompare(ComboId a, ComboId b, const BoardRunout& board) {
  const HandEvaluation ea = ReferenceBest(a, board);
  const HandEvaluation eb = ReferenceBest(b, board);
  return ea > eb ? 1 : ea < eb ? -1 : 0;
}

Card Rename(Card card, const std::array<S, 4>& suits) {
  return C(RankFromCardId(card),
           suits[static_cast<size_t>(SuitIndex(SuitFromCardId(card)))]);
}

BoardRunout Rename(const BoardRunout& board, const std::array<S, 4>& suits) {
  std::array<Card, kMaxBoardCards> cards = {};
  for (size_t i = 0; i < board.cards().size(); ++i) {
    cards[i] = Rename(board.cards()[i], suits);
  }
  return Runout(absl::Span<const Card>(cards.data(), board.count()));
}

TEST_CASE("combo ids are an exhaustive canonical bijection") {
  ComboId expected = 0;
  for (int a = 0; a < kDeckCardCount; ++a) {
    for (int b = a + 1; b < kDeckCardCount; ++b, ++expected) {
      const Card ca = kDeck[static_cast<size_t>(a)];
      const Card cb = kDeck[static_cast<size_t>(b)];
      CHECK(H(ca, cb) == expected);
      CHECK(H(cb, ca) == expected);
      const ComboInfo& info = GetComboInfo(expected);
      CHECK(info.card0 == ca);
      CHECK(info.card1 == cb);
      CHECK(info.mask == (CardBit(ca) | CardBit(cb)));
    }
  }
  CHECK(expected == kComboCount);
  CHECK_FALSE(MaybeCardsToComboId(kDeck[0], kDeck[0]).has_value());
}

TEST_CASE("hand evaluator matches an independent five-card oracle") {
  const auto wheel = EvaluateFiveCards(std::array<Card, 5>{
      C(14, S::kHearts), C(5, S::kDiamonds), C(4, S::kClubs),
      C(3, S::kSpades), C(2, S::kHearts)});
  CHECK(wheel.rank == HandRank::STRAIGHT);
  CHECK(wheel.kickers[0] == 5);

  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 400; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId a = H(deck[0], deck[1]);
    const ComboId b = H(deck[2], deck[3]);
    const BoardRunout board = B({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CAPTURE(trial);
    CHECK(CompareHands(a, b, board) ==
          ReferenceCompare(a, b, board));
  }
}

TEST_CASE("sampling and card abstractions preserve identity") {
  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(7);

  for (int trial = 0; trial < 150; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const StreetKind street = static_cast<StreetKind>(trial % 3);
    const int board_count = BoardCardsForStreet(street);
    const BoardRunout board = Runout(
        absl::Span<const Card>(deck.data(), board_count));
    const ComboId hand = H(deck[board_count], deck[board_count + 1]);

    std::array<Card, kMaxBoardCards> permuted = {};
    std::copy(board.cards().begin(), board.cards().end(), permuted.begin());
    if (board.count() >= 3) {
      std::swap(permuted[0], permuted[2]);
    }
    const BoardRunout reversed = Runout(
        absl::Span<const Card>(permuted.data(), board.count()));
    const BoardFeatures features = board_features(board);
    CHECK(board_features(reversed) == features);
    CHECK(board_texture_bucket(street, features) ==
          board_texture_bucket(street, board_features(reversed)));
    const PrivateStreetObservation private_observation =
        observe_private_street(hand, street, features);
    CHECK(private_observation.value == hand);

    std::array<S, 4> suits = {
        S::kHearts, S::kDiamonds, S::kClubs, S::kSpades};
    std::shuffle(suits.begin(), suits.end(), rng);
    const BoardRunout renamed_board = Rename(board, suits);
    CHECK(board_texture_bucket(street, features) ==
          board_texture_bucket(street, board_features(renamed_board)));

    const auto sampled = SampleStreetCards(street, board, ComboMask(hand), rng);
    CHECK(sampled.size() == static_cast<size_t>(CardsForNextStreet(street)));
    CardMask blocked = board.mask() | ComboMask(hand);
    for (Card card : sampled) {
      CHECK((blocked & CardBit(card)) == 0);
      blocked |= CardBit(card);
    }
  }

  CardMask blocked = 0;
  for (int i = 0; i < kDeckCardCount - 2; ++i)
    blocked |= CardBit(kDeck[static_cast<size_t>(i)]);
  CHECK_THROWS_AS(SampleStreetCards(StreetKind::kPreflop,
                                    BoardRunout::Preflop(), blocked, rng),
                  std::runtime_error);
}

}  // namespace
}  // namespace poker
