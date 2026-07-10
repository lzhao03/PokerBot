#include "src/card_utils.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace poker {
namespace {

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

TEST_CASE("street sampling returns unique unblocked cards") {
  const bool passed = rc::check("valid card sampling", [] {
    const int street_index = *rc::gen::inRange(0, 4).as("street");
    const int private_count = *rc::gen::inRange(0, 5).as("private count");
    const auto choices =
        *rc::gen::arbitrary<std::array<uint16_t, 9>>().as("cards");
    const uint32_t seed = *rc::gen::arbitrary<uint32_t>().as("seed");
    const StreetKind street = static_cast<StreetKind>(street_index);

    Board board;
    CardMask blocked = 0;
    size_t cursor = 0;
    const int board_count = BoardCardsForStreet(street);
    for (int i = 0; i < board_count; ++i) {
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
