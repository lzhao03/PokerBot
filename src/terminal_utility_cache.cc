#include "src/terminal_utility_cache.h"

#include <algorithm>
#include <cmath>

namespace poker {
namespace {

int EncodeCard(const Card& card) {
  return (card.rank() * 8) + static_cast<int>(card.suit());
}

template <size_t N, typename Cards>
std::array<int, N> EncodedSortedCards(const Cards& cards) {
  std::array<int, N> encoded;
  encoded.fill(-1);
  size_t count = 0;
  for (const Card& card : cards) {
    if (count == N) {
      break;
    }
    encoded[count] = EncodeCard(card);
    ++count;
  }
  std::sort(encoded.begin(), encoded.begin() + count);
  return encoded;
}

int ContributionAt(const BoardState& state, int player) {
  return state.player_contribution_size() > player
             ? static_cast<int>(std::llround(state.player_contribution(player)))
             : 0;
}

void HashCombine(size_t& seed, int value) {
  seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <size_t N>
void HashArray(size_t& seed, const std::array<int, N>& values) {
  for (int value : values) {
    HashCombine(seed, value);
  }
}

}  // namespace

TerminalUtilityCache::Stats TerminalUtilityCache::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {hits_, misses_, static_cast<int64_t>(values_.size())};
}

bool TerminalUtilityCache::Key::operator==(const Key& other) const {
  return street == other.street && pot == other.pot &&
         player_a_contribution == other.player_a_contribution &&
         player_b_contribution == other.player_b_contribution &&
         board_size == other.board_size &&
         player_a_cards == other.player_a_cards &&
         player_b_cards == other.player_b_cards &&
         board_cards == other.board_cards;
}

size_t TerminalUtilityCache::KeyHash::operator()(const Key& key) const {
  size_t seed = 0;
  HashCombine(seed, key.street);
  HashCombine(seed, key.pot);
  HashCombine(seed, key.player_a_contribution);
  HashCombine(seed, key.player_b_contribution);
  HashCombine(seed, key.board_size);
  HashArray(seed, key.player_a_cards);
  HashArray(seed, key.player_b_cards);
  HashArray(seed, key.board_cards);
  return seed;
}

TerminalUtilityCache::Key TerminalUtilityCache::key_for(
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand) {
  Key key;
  key.street = static_cast<int>(state.street());
  key.pot = state.pot();
  key.player_a_contribution = ContributionAt(state, 0);
  key.player_b_contribution = ContributionAt(state, 1);
  key.board_size = state.cards_size();
  key.player_a_cards = EncodedSortedCards<2>(player_a_hand.cards());
  key.player_b_cards = EncodedSortedCards<2>(player_b_hand.cards());
  key.board_cards = EncodedSortedCards<5>(state.cards());
  return key;
}

}  // namespace poker
