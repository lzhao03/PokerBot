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

uint64_t EncodeBoard(const Board& board,
                     const SuitPermutation& permutation) noexcept {
  const auto cards = board.cards();
  std::array<Card, kMaxBoardCards> mapped = {};
  for (size_t index = 0; index < cards.size(); ++index) {
    mapped[index] = PermuteCard(cards[index], permutation);
  }
  std::sort(mapped.begin(),
            mapped.begin() + std::min<size_t>(3, cards.size()));
  uint64_t encoded = cards.size();
  size_t shift = 3;
  for (size_t index = 0; index < cards.size(); ++index) {
    encoded |= static_cast<uint64_t>(mapped[index].index()) << shift;
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
    best = std::min(best, EncodeBoard(board, permutation));
  }
  return PublicObservationId(best);
}

PrivateObservationId CanonicalPrivateObservation(
    ComboId hand,
    const Board& board) noexcept {
  std::pair<uint64_t, uint16_t> best{
      std::numeric_limits<uint64_t>::max(),
      std::numeric_limits<uint16_t>::max()};
  for (const SuitPermutation& permutation : kSuitPermutations) {
    const uint64_t board_id = EncodeBoard(board, permutation);
    const ComboId mapped_hand = PermuteCombo(hand, permutation);
    const uint16_t hand_id = static_cast<uint16_t>(mapped_hand.index());
    best = std::min(best, std::pair{board_id, hand_id});
  }
  return PrivateObservationId(best.second);
}

}  // namespace poker
