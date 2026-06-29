#include "simulator.h"

#include <algorithm>
#include <unordered_map>

namespace poker {

namespace {
void SetStack(BoardState* s, int player, int value) {
  if (player == 0) {
    s->set_stack_a(value);
  } else {
    s->set_stack_b(value);
  }
}

int GetStack(const BoardState& s, int player) {
  return (player == 0) ? s.stack_a() : s.stack_b();
}

int Opp(int p) { return 1 - p; }
}

bool NoLimitHoldemSimulator::StartNewHand(const Config& cfg, BoardState* state,
                                          Hand* p0_hole, Hand* p1_hole) {
  if (!state || !p0_hole || !p1_hole) return false;

  // Reset state
  state->Clear();
  state->set_stack_a(cfg.starting_stack);
  state->set_stack_b(cfg.starting_stack);
  state->set_pot(0);
  state->mutable_cards()->Clear();
  state->mutable_history()->mutable_actions()->Clear();
  state->set_street(Street::PREFLOP);
  state->set_all_in(false);
  state->set_folded_player(-1);
  state->mutable_player_contribution()->Clear();
  state->add_player_contribution(0.0);
  state->add_player_contribution(0.0);
  state->set_player_to_act(0);  // SB acts first preflop in heads-up

  BuildFreshDeck();
  ShuffleDeck();
  DealHoles(p0_hole, p1_hole);

  // Post blinds: player 0 = SB, player 1 = BB
  int sb = std::min(cfg.small_blind, GetStack(*state, 0));
  int bb = std::min(cfg.big_blind, GetStack(*state, 1));
  SetStack(state, 0, GetStack(*state, 0) - sb);
  SetStack(state, 1, GetStack(*state, 1) - bb);
  state->set_pot(state->pot() + sb + bb);
  (*state->mutable_player_contribution())[0] += sb;
  (*state->mutable_player_contribution())[1] += bb;

  // Record blind actions (as BET for simplicity)
  Action a_sb; a_sb.set_action(ActionType::BET); a_sb.set_amount(sb);
  Action a_bb; a_bb.set_action(ActionType::BET); a_bb.set_amount(bb);
  a_sb.set_player(0);
  a_bb.set_player(1);
  *state->mutable_history()->add_actions() = a_sb;
  *state->mutable_history()->add_actions() = a_bb;

  // If either blind is all-in (short stack), mark all_in
  if (GetStack(*state, 0) == 0 || GetStack(*state, 1) == 0) {
    state->set_all_in(true);
  }

  return true;
}

void NoLimitHoldemSimulator::BuildFreshDeck() {
  deck_.clear();
  deck_.reserve(52);
  for (int suit = Suit::HEARTS; suit <= Suit::SPADES; ++suit) {
    for (int rank = 1; rank <= 13; ++rank) {  // 1=Ace ... 13=King
      Card c; c.set_rank(rank); c.set_suit(static_cast<Suit>(suit));
      deck_.push_back(c);
    }
  }
  deck_index_ = 0;
}

void NoLimitHoldemSimulator::ShuffleDeck() { std::shuffle(deck_.begin(), deck_.end(), rng_); }

void NoLimitHoldemSimulator::DealHoles(Hand* p0_hole, Hand* p1_hole) {
  p0_hole->mutable_cards()->Clear();
  p1_hole->mutable_cards()->Clear();
  // Deal two cards each alternating
  for (int i = 0; i < 2; ++i) {
    *p0_hole->add_cards() = deck_[deck_index_++];
    *p1_hole->add_cards() = deck_[deck_index_++];
  }
}

void NoLimitHoldemSimulator::DealCommunity(BoardState* state, int n) {
  for (int i = 0; i < n; ++i) {
    *state->add_cards() = deck_[deck_index_++];
  }
}

int NoLimitHoldemSimulator::ContributionSize(const BoardState& state, int player) const {
  if (state.player_contribution_size() <= player) return 0;
  return static_cast<int>(state.player_contribution(player));
}

int NoLimitHoldemSimulator::OutstandingToCall(const BoardState& state, int player) const {
  int me = ContributionSize(state, player);
  int opp = ContributionSize(state, Opp(player));
  return std::max(0, opp - me);
}

bool NoLimitHoldemSimulator::BettingRoundOver(const BoardState& state) const {
  if (state.all_in() || state.folded_player() != -1) return true;
  if (state.history().actions_size() == 0) return false;
  const Action& last = state.history().actions(state.history().actions_size() - 1);
  int p = state.player_to_act();
  int to_call_me = OutstandingToCall(state, p);
  int to_call_opp = OutstandingToCall(state, Opp(p));
  // Round over if no bets outstanding for both and last action was CHECK or CALL
  if (to_call_me == 0 && to_call_opp == 0) {
    if (last.action() == ActionType::CHECK || last.action() == ActionType::CALL) return true;
  }
  return false;
}

bool NoLimitHoldemSimulator::ApplyAction(const Action& action, BoardState* state) {
  if (!state || IsTerminal(*state)) return false;
  int p = state->player_to_act();
  int opp = Opp(p);

  Action applied = action;  // we'll clamp/adjust amount
  applied.set_player(p);

  switch (action.action()) {
    case ActionType::FOLD: {
      state->set_folded_player(p);
      *state->mutable_history()->add_actions() = applied;
      return true;  // terminal via fold
    }
    case ActionType::CHECK: {
      if (OutstandingToCall(*state, p) != 0) return false;  // illegal check facing bet
      applied.set_amount(0);
      *state->mutable_history()->add_actions() = applied;
      state->set_player_to_act(opp);
      return true;
    }
    case ActionType::CALL: {
      int to_call = OutstandingToCall(*state, p);
      if (to_call == 0) return false;  // nothing to call
      int pay = std::min(to_call, GetStack(*state, p));
      // Update contributions, stacks, pot
      (*state->mutable_player_contribution())[p] += pay;
      SetStack(state, p, GetStack(*state, p) - pay);
      state->set_pot(state->pot() + pay);
      applied.set_amount(pay);
      *state->mutable_history()->add_actions() = applied;
      if (GetStack(*state, p) == 0) state->set_all_in(true);
      state->set_player_to_act(opp);
      return true;
    }
    case ActionType::BET:
    case ActionType::RAISE:
    case ActionType::ALL_IN: {
      int stack = GetStack(*state, p);
      if (stack <= 0) return false;
      int to_call = OutstandingToCall(*state, p);
      int commit = 0;
      if (action.action() == ActionType::ALL_IN) {
        commit = stack;  // push all remaining
      } else {
        // For minimal simulator, accept requested amount, but ensure it exceeds call amount by at least 1 when raising.
        commit = std::min(stack, static_cast<int>(action.amount()));
      }
      if (commit < to_call + (to_call > 0 ? 1 : 1)) {
        // Must at least increase the price by 1 chip over current if betting/raising.
        commit = std::min(stack, to_call + 1);
      }
      // Apply commit
      (*state->mutable_player_contribution())[p] += commit;
      SetStack(state, p, GetStack(*state, p) - commit);
      state->set_pot(state->pot() + commit);
      applied.set_amount(commit);
      *state->mutable_history()->add_actions() = applied;
      if (GetStack(*state, p) == 0) state->set_all_in(true);
      state->set_player_to_act(opp);
      return true;
    }
    default:
      return false;
  }
}

bool NoLimitHoldemSimulator::AdvanceIfReady(BoardState* state) {
  if (!state) return false;
  if (IsTerminal(*state)) return false;

  if (!BettingRoundOver(*state)) return false;

  // If we are on river and betting is over, hand is terminal (showdown or fold already handled).
  if (state->street() == Street::RIVER) {
    return true;  // nothing to deal; terminal achieved
  }

  // Advance street and deal community cards
  switch (state->street()) {
    case Street::PREFLOP:
      state->set_street(Street::FLOP);
      DealCommunity(state, 3);
      break;
    case Street::FLOP:
      state->set_street(Street::TURN);
      DealCommunity(state, 1);
      break;
    case Street::TURN:
      state->set_street(Street::RIVER);
      DealCommunity(state, 1);
      break;
    default:
      break;
  }

  // Reset action context for next street: the next player to act is SB? In heads-up, first postflop actor is player 1 (BB)
  state->set_player_to_act(1);
  // No need to reset contributions since we store total; betting logic relies on equality only.

  return true;
}

bool NoLimitHoldemSimulator::IsTerminal(const BoardState& state) const {
  if (state.folded_player() != -1) return true;
  if (state.all_in()) {
    // If someone is all-in and all bets matched, we can treat as terminal (auto-runout would be external).
    int to_call_p0 = OutstandingToCall(state, 0);
    int to_call_p1 = OutstandingToCall(state, 1);
    if (to_call_p0 == 0 && to_call_p1 == 0 && state.street() == Street::RIVER && BettingRoundOver(state)) {
      return true;
    }
    // Otherwise, simulator can still advance streets automatically via AdvanceIfReady.
  }
  // Terminal if river betting over
  if (state.street() == Street::RIVER && BettingRoundOver(state)) return true;
  return false;
}

NoLimitHoldemSimulator::Result NoLimitHoldemSimulator::Showdown(
    const BoardState& state, const Hand& p0_hole, const Hand& p1_hole) const {
  Result res;
  res.terminal = IsTerminal(state);
  if (!res.terminal) return res;

  if (state.folded_player() != -1) {
    res.winner = Opp(state.folded_player());
    res.split = false;
    return res;
  }

  // Evaluate both hands
  HandValue v0 = EvaluateBestHand(p0_hole, state);
  HandValue v1 = EvaluateBestHand(p1_hole, state);
  if (v0 > v1) {
    res.winner = 0;
  } else if (v1 > v0) {
    res.winner = 1;
  } else {
    res.winner = -1;
    res.split = true;
  }
  return res;
}

NoLimitHoldemSimulator::HandValue NoLimitHoldemSimulator::EvaluateBestHand(
    const Hand& hole, const BoardState& board) const {
  std::vector<Card> all_cards = CombineCards(hole, board);
  if (all_cards.size() < 5) {
    // Not enough cards for evaluation - return high card
    HandValue hv;
    hv.rank = HIGH_CARD;
    hv.kickers.fill(0);
    return hv;
  }
  
  // Try all combinations of 5 cards and find the best
  HandValue best;
  best.rank = HIGH_CARD;
  best.kickers.fill(0);
  
  std::vector<bool> selector(all_cards.size());
  std::fill(selector.begin(), selector.begin() + 5, true);
  
  do {
    std::vector<Card> hand;
    for (size_t i = 0; i < all_cards.size(); ++i) {
      if (selector[i]) hand.push_back(all_cards[i]);
    }
    HandValue current = EvaluateHand(hand);
    if (current > best) best = current;
  } while (std::prev_permutation(selector.begin(), selector.end()));
  
  return best;
}

NoLimitHoldemSimulator::HandValue NoLimitHoldemSimulator::EvaluateHand(
    const std::vector<Card>& cards) const {
  if (cards.size() != 5) {
    HandValue hv;
    hv.rank = HIGH_CARD;
    hv.kickers.fill(0);
    return hv;
  }
  
  HandValue result;
  
  // Convert to ranks and suits
  std::vector<int> ranks;
  std::vector<int> suits;
  for (const Card& c : cards) {
    ranks.push_back(c.rank());
    suits.push_back(static_cast<int>(c.suit()));
  }
  
  // Sort ranks in descending order for easier evaluation
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());
  
  // Count rank frequencies
  std::unordered_map<int, int> rank_counts;
  for (int r : ranks) rank_counts[r]++;
  
  // Check for flush
  bool is_flush = true;
  for (size_t i = 1; i < suits.size(); ++i) {
    if (suits[i] != suits[0]) {
      is_flush = false;
      break;
    }
  }
  
  // Check for straight (including A-2-3-4-5 wheel)
  bool is_straight = false;
  if (ranks[0] == 14 && ranks[1] == 5 && ranks[2] == 4 && ranks[3] == 3 && ranks[4] == 2) {
    // A-2-3-4-5 wheel straight
    is_straight = true;
    ranks = {5, 4, 3, 2, 1};  // Treat ace as 1 for wheel
  } else {
    // Regular straight check
    is_straight = true;
    for (size_t i = 1; i < ranks.size(); ++i) {
      if (ranks[i] != ranks[i-1] - 1) {
        is_straight = false;
        break;
      }
    }
  }
  
  // Determine hand rank and kickers
  if (is_straight && is_flush) {
    if (ranks[0] == 14) {
      result.rank = ROYAL_FLUSH;
    } else {
      result.rank = STRAIGHT_FLUSH;
    }
    result.kickers[0] = ranks[0];
    for (int i = 1; i < 5; ++i) result.kickers[i] = 0;
  } else if (rank_counts.size() == 2) {
    // Either four of a kind or full house
    std::vector<std::pair<int, int>> count_pairs;
    for (const auto& p : rank_counts) {
      count_pairs.push_back({p.second, p.first});
    }
    std::sort(count_pairs.rbegin(), count_pairs.rend());
    
    if (count_pairs[0].first == 4) {
      result.rank = FOUR_KIND;
      result.kickers[0] = count_pairs[0].second;  // quad rank
      result.kickers[1] = count_pairs[1].second;  // kicker
      for (int i = 2; i < 5; ++i) result.kickers[i] = 0;
    } else {
      result.rank = FULL_HOUSE;
      result.kickers[0] = count_pairs[0].second;  // trips rank
      result.kickers[1] = count_pairs[1].second;  // pair rank
      for (int i = 2; i < 5; ++i) result.kickers[i] = 0;
    }
  } else if (is_flush) {
    result.rank = FLUSH;
    for (int i = 0; i < 5; ++i) result.kickers[i] = ranks[i];
  } else if (is_straight) {
    result.rank = STRAIGHT;
    result.kickers[0] = ranks[0];
    for (int i = 1; i < 5; ++i) result.kickers[i] = 0;
  } else if (rank_counts.size() == 3) {
    // Either three of a kind or two pair
    std::vector<std::pair<int, int>> count_pairs;
    for (const auto& p : rank_counts) {
      count_pairs.push_back({p.second, p.first});
    }
    std::sort(count_pairs.rbegin(), count_pairs.rend());
    
    if (count_pairs[0].first == 3) {
      result.rank = THREE_KIND;
      result.kickers[0] = count_pairs[0].second;  // trips rank
      result.kickers[1] = count_pairs[1].second;  // first kicker
      result.kickers[2] = count_pairs[2].second;  // second kicker
      for (int i = 3; i < 5; ++i) result.kickers[i] = 0;
    } else {
      result.rank = TWO_PAIR;
      result.kickers[0] = std::max(count_pairs[0].second, count_pairs[1].second);
      result.kickers[1] = std::min(count_pairs[0].second, count_pairs[1].second);
      result.kickers[2] = count_pairs[2].second;  // kicker
      for (int i = 3; i < 5; ++i) result.kickers[i] = 0;
    }
  } else if (rank_counts.size() == 4) {
    // One pair
    result.rank = PAIR;
    int pair_rank = 0;
    std::vector<int> kickers;
    for (const auto& p : rank_counts) {
      if (p.second == 2) {
        pair_rank = p.first;
      } else {
        kickers.push_back(p.first);
      }
    }
    std::sort(kickers.rbegin(), kickers.rend());
    result.kickers[0] = pair_rank;
    for (size_t i = 0; i < kickers.size() && i < 3; ++i) {
      result.kickers[i + 1] = kickers[i];
    }
    for (int i = kickers.size() + 1; i < 5; ++i) result.kickers[i] = 0;
  } else {
    // High card
    result.rank = HIGH_CARD;
    for (int i = 0; i < 5; ++i) result.kickers[i] = ranks[i];
  }
  
  return result;
}

std::vector<Card> NoLimitHoldemSimulator::CombineCards(
    const Hand& hole, const BoardState& board) const {
  std::vector<Card> combined;
  
  // Add hole cards
  for (const Card& c : hole.cards()) {
    combined.push_back(c);
  }
  
  // Add board cards
  for (const Card& c : board.cards()) {
    combined.push_back(c);
  }
  
  return combined;
}

}  // namespace poker
