#include "src/card_canonicalization.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace poker {
namespace {

using SuitPermutation = std::array<Suit, 4>;

constexpr std::array<SuitPermutation, 24> BuildSuitPermutations() {
  std::array<SuitPermutation, 24> permutations = {};
  SuitPermutation current = {
      Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades};
  size_t index = 0;
  do {
    permutations[index++] = current;
  } while (std::next_permutation(current.begin(), current.end()));
  return permutations;
}

inline constexpr auto kSuitPermutations = BuildSuitPermutations();

Card PermuteCard(Card card, const SuitPermutation& permutation) noexcept {
  return Card(card.rank(), permutation[static_cast<size_t>(card.suit())]);
}

Board PermuteBoard(const Board& board,
                   const SuitPermutation& permutation) noexcept {
  const auto cards = BoardCards(board);
  if (cards.empty()) {
    return PreflopBoard{};
  }

  std::array<Card, kMaxBoardCards> mapped = {};
  for (size_t index = 0; index < cards.size(); ++index) {
    mapped[index] = PermuteCard(cards[index], permutation);
  }
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {mapped[0], mapped[1], mapped[2]});
  if (cards.size() == 3) {
    return flop;
  }
  const TurnBoard turn = DealTurn(flop, mapped[3]);
  return cards.size() == 4 ? Board{turn}
                           : Board{DealRiver(turn, mapped[4])};
}

uint64_t EncodeBoard(const Board& board) noexcept {
  uint64_t encoded = BoardCount(board);
  size_t shift = 3;
  for (Card card : BoardCards(board)) {
    encoded |= static_cast<uint64_t>(card.index()) << shift;
    shift += 6;
  }
  return encoded;
}

ComboId PermuteCombo(ComboId hand,
                     const SuitPermutation& permutation) noexcept {
  const ComboInfo& combo = GetComboInfo(hand);
  return CardsToComboId(PermuteCard(combo.card0, permutation),
                        PermuteCard(combo.card1, permutation));
}

}  // namespace

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (const SuitPermutation& permutation : kSuitPermutations) {
    best = std::min(best, EncodeBoard(PermuteBoard(board, permutation)));
  }
  return PublicObservationId(best);
}

CanonicalCardObservation CanonicalizeObservation(
    ComboId hand,
    const Board& board) noexcept {
  uint64_t best_board = std::numeric_limits<uint64_t>::max();
  uint16_t best_hand = std::numeric_limits<uint16_t>::max();
  for (const SuitPermutation& permutation : kSuitPermutations) {
    const Board mapped_board = PermuteBoard(board, permutation);
    const uint64_t board_id = EncodeBoard(mapped_board);
    const ComboId mapped_hand = PermuteCombo(hand, permutation);
    const uint16_t hand_id = static_cast<uint16_t>(mapped_hand.index());
    if (board_id < best_board ||
        (board_id == best_board && hand_id < best_hand)) {
      best_board = board_id;
      best_hand = hand_id;
    }
  }
  return {PublicObservationId(best_board), PrivateObservationId(best_hand)};
}

}  // namespace poker
