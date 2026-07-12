#include "src/card_canonicalization.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

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
  return Card(card.rank(), permutation[std::to_underlying(card.suit())]);
}

Board PermuteBoard(const Board& board,
                   const SuitPermutation& permutation) noexcept {
  const auto cards = board.cards();
  if (cards.empty()) {
    return {};
  }

  std::array<Card, kMaxBoardCards> mapped = {};
  for (size_t index = 0; index < cards.size(); ++index) {
    mapped[index] = PermuteCard(cards[index], permutation);
  }
  Board result = DealCards(
      Board{}, absl::Span<const Card>(mapped.data(), 3));
  for (size_t index = 3; index < cards.size(); ++index) {
    result = DealCards(
        result, absl::Span<const Card>(mapped.data() + index, 1));
  }
  return result;
}

uint64_t EncodeBoard(const Board& board) noexcept {
  uint64_t encoded = board.count();
  size_t shift = 3;
  for (Card card : board.cards()) {
    encoded |= static_cast<uint64_t>(card.index()) << shift;
    shift += 6;
  }
  return encoded;
}

ComboId PermuteCombo(ComboId hand,
                     const SuitPermutation& permutation) noexcept {
  const auto cards = hand.cards();
  return CardsToComboId(PermuteCard(cards[0], permutation),
                        PermuteCard(cards[1], permutation));
}

}  // namespace

PublicObservationId CanonicalPublicObservation(const Board& board) noexcept {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (const SuitPermutation& permutation : kSuitPermutations) {
    best = std::min(best, EncodeBoard(PermuteBoard(board, permutation)));
  }
  return PublicObservationId(best);
}

PrivateObservationId CanonicalPrivateObservation(
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
  return PrivateObservationId(best_hand);
}

}  // namespace poker
