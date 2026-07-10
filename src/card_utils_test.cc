#include "src/card_utils.h"

#include "doctest/doctest.h"

#include <array>
#include <random>
#include <stdexcept>

namespace poker {
namespace {

void CheckSample(CardMask blocked_cards,
                 absl::Span<const CardId> cards,
                 size_t expected_count) {
  REQUIRE(cards.size() == expected_count);
  CardMask seen = 0;
  for (CardId card : cards) {
    CHECK((blocked_cards & CardBit(card)) == 0);
    CHECK((seen & CardBit(card)) == 0);
    seen |= CardBit(card);
  }
}

TEST_CASE("street sampling returns unique unblocked cards") {
  const CardMask known_private_cards =
      CardBit(MakeCardId(14, SuitKind::kSpades)) |
      CardBit(MakeCardId(13, SuitKind::kSpades)) |
      CardBit(MakeCardId(12, SuitKind::kHearts)) |
      CardBit(MakeCardId(11, SuitKind::kHearts));

  struct Scenario {
    StreetKind street;
    std::array<CardId, kMaxBoardCards> board;
    uint8_t board_count = 0;
    size_t expected_count = 0;
  };
  const std::array<Scenario, 4> scenarios{{
      {StreetKind::kPreflop, {}, 0, 3},
      {StreetKind::kFlop,
       {MakeCardId(2, SuitKind::kClubs),
        MakeCardId(3, SuitKind::kDiamonds),
        MakeCardId(4, SuitKind::kHearts)},
       3,
       1},
      {StreetKind::kTurn,
       {MakeCardId(2, SuitKind::kClubs),
        MakeCardId(3, SuitKind::kDiamonds),
        MakeCardId(4, SuitKind::kHearts),
        MakeCardId(5, SuitKind::kSpades)},
       4,
       1},
      {StreetKind::kRiver,
       {MakeCardId(2, SuitKind::kClubs),
        MakeCardId(3, SuitKind::kDiamonds),
        MakeCardId(4, SuitKind::kHearts),
        MakeCardId(5, SuitKind::kSpades),
        MakeCardId(6, SuitKind::kClubs)},
       5,
       0},
  }};

  std::mt19937 rng(12345);
  for (const Scenario& scenario : scenarios) {
    Board board;
    for (uint8_t i = 0; i < scenario.board_count; ++i) {
      board.add(scenario.board[i]);
    }

    CheckSample(known_private_cards | board.mask,
                SampleStreetCards(scenario.street, board,
                                  known_private_cards, rng),
                scenario.expected_count);
  }
}

TEST_CASE("one-card sampling returns the only unblocked card") {
  const CardId legal_card = MakeCardId(2, SuitKind::kClubs);
  CardMask blocked_cards = 0;
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    const CardId card = static_cast<CardId>(card_id);
    if (card != legal_card) {
      blocked_cards |= CardBit(card);
    }
  }

  std::mt19937 rng(12345);
  const auto sampled = SampleStreetCards(
      StreetKind::kFlop, Board{}, blocked_cards, rng);
  REQUIRE(sampled.size() == 1);
  CHECK(sampled[0] == legal_card);
}

TEST_CASE("sampling throws when too few cards remain") {
  CardMask blocked_cards = 0;
  for (int card_id = 0; card_id < kDeckCardCount - 2; ++card_id) {
    blocked_cards |= CardBit(static_cast<CardId>(card_id));
  }

  std::mt19937 rng(12345);
  CHECK_THROWS_AS(SampleStreetCards(StreetKind::kPreflop, Board{},
                                    blocked_cards, rng),
                  std::runtime_error);
}

}  // namespace
}  // namespace poker
