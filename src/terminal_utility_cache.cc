#include "src/terminal_utility_cache.h"

#include <algorithm>

namespace poker {
namespace {

template <size_t N>
std::array<int, N> EncodedSortedCards(
    const std::array<CardId, N>& cards,
    size_t count) {
  std::array<int, N> encoded;
  encoded.fill(-1);
  for (size_t i = 0; i < count && i < N; ++i) {
    encoded[i] = EncodedCard(cards[i]);
  }
  std::sort(encoded.begin(), encoded.begin() + std::min(count, N));
  return encoded;
}

std::array<int, 2> EncodedComboCards(ComboId combo_id) {
  const ComboInfo& combo = GetComboInfo(combo_id);
  std::array<CardId, 2> cards = {combo.card0, combo.card1};
  return EncodedSortedCards(cards, cards.size());
}

std::array<int, 5> EncodedBoardCards(const GameState& state) {
  std::array<CardId, 5> cards = {};
  for (size_t i = 0; i < state.board_cards.size() && i < cards.size(); ++i) {
    cards[i] = state.board_cards[i];
  }
  return EncodedSortedCards(cards, state.board_cards.size());
}

std::array<int, 5> EncodedBoardCards(const CompactPublicState& state) {
  std::array<CardId, 5> cards = {};
  for (size_t i = 0; i < state.board_count && i < cards.size(); ++i) {
    cards[i] = state.board_cards[i];
  }
  return EncodedSortedCards(cards, state.board_count);
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
    const GameState& state,
    ComboId player_a_hand,
    ComboId player_b_hand) {
  Key key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.player_a_contribution = state.player_contribution[0];
  key.player_b_contribution = state.player_contribution[1];
  key.board_size = static_cast<int>(state.board_cards.size());
  key.player_a_cards = EncodedComboCards(player_a_hand);
  key.player_b_cards = EncodedComboCards(player_b_hand);
  key.board_cards = EncodedBoardCards(state);
  return key;
}

TerminalUtilityCache::Key TerminalUtilityCache::key_for(
    const CompactPublicState& state,
    ComboId player_a_hand,
    ComboId player_b_hand) {
  Key key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.player_a_contribution = state.player_contribution[0];
  key.player_b_contribution = state.player_contribution[1];
  key.board_size = state.board_count;
  key.player_a_cards = EncodedComboCards(player_a_hand);
  key.player_b_cards = EncodedComboCards(player_b_hand);
  key.board_cards = EncodedBoardCards(state);
  return key;
}

}  // namespace poker
