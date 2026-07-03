#include "simulator.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace poker {

void PrintCard(const Card& card) {
  constexpr std::array<std::string_view, 15> ranks = {
      "", "A", "2", "3", "4", "5", "6", "7",
      "8", "9", "T", "J", "Q", "K", "A"};
  constexpr std::array<std::string_view, 5> suits = {"?", "♥", "♦", "♣", "♠"};
  
  if (card.rank() >= 1 && card.rank() <= 13) {
    std::cout << ranks[card.rank()] << suits[static_cast<int>(card.suit())];
  } else {
    std::cout << "??";
  }
}

void PrintHand(const Hand& hand, const std::string& label) {
  std::cout << label << ": ";
  for (const Card& card : hand.cards()) {
    PrintCard(card);
    std::cout << " ";
  }
  std::cout << std::endl;
}

void PrintBoard(const BoardState& state) {
  std::cout << "Board: ";
  for (const Card& card : state.cards()) {
    PrintCard(card);
    std::cout << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Pot: " << state.pot() << " | ";
  std::cout << "P0 Stack: " << state.stack_a() << " | ";
  std::cout << "P1 Stack: " << state.stack_b() << " | ";
  std::cout << "Street: ";
  switch (state.street()) {
    case PREFLOP: std::cout << "Preflop"; break;
    case FLOP: std::cout << "Flop"; break;
    case TURN: std::cout << "Turn"; break;
    case RIVER: std::cout << "River"; break;
    default: std::cout << "Unknown"; break;
  }
  std::cout << " | To Act: P" << state.player_to_act() << std::endl;
}

void PrintAction(const Action& action, int player) {
  std::cout << "P" << player << " ";
  switch (action.action()) {
    case ActionType::FOLD: std::cout << "folds"; break;
    case ActionType::CHECK: std::cout << "checks"; break;
    case ActionType::CALL: std::cout << "calls " << action.amount(); break;
    case ActionType::BET: std::cout << "bets " << action.amount(); break;
    case ActionType::RAISE: std::cout << "raises to " << action.amount(); break;
    case ActionType::ALL_IN: std::cout << "goes all-in for " << action.amount(); break;
    default: std::cout << "unknown action"; break;
  }
  std::cout << std::endl;
}

void RunSimpleGame() {
  std::cout << "=== No-Limit Hold'em Simulator Test ===" << std::endl;
  
  NoLimitHoldemSimulator simulator(12345);  // Fixed seed for reproducible results
  NoLimitHoldemSimulator::Config config;
  config.starting_stack = 100;
  config.small_blind = 1;
  config.big_blind = 2;
  
  BoardState state;
  Hand p0_hole, p1_hole;
  
  // Start new hand
  if (!simulator.StartNewHand(config, state, p0_hole, p1_hole)) {
    std::cout << "Failed to start hand!" << std::endl;
    return;
  }
  
  std::cout << "\n--- Hand Started ---" << std::endl;
  PrintHand(p0_hole, "P0 Hole");
  PrintHand(p1_hole, "P1 Hole");
  PrintBoard(state);
  
  // Simulate some preflop action
  std::cout << "\n--- Preflop Action ---" << std::endl;
  
  // P0 (SB) calls
  Action call;
  call.set_action(ActionType::CALL);
  call.set_amount(1);  // Call the remaining 1 to match BB
  
  if (simulator.ApplyAction(call, state)) {
    PrintAction(call, 0);
    PrintBoard(state);
  }
  
  // P1 (BB) checks
  Action check;
  check.set_action(ActionType::CHECK);
  check.set_amount(0);
  
  if (simulator.ApplyAction(check, state)) {
    PrintAction(check, 1);
    PrintBoard(state);
  }
  
  // Advance to flop
  if (simulator.AdvanceIfReady(state)) {
    std::cout << "\n--- Flop ---" << std::endl;
    PrintBoard(state);
  }
  
  // P1 bets
  Action bet;
  bet.set_action(ActionType::BET);
  bet.set_amount(5);
  
  if (simulator.ApplyAction(bet, state)) {
    PrintAction(bet, 1);
    PrintBoard(state);
  }
  
  // P0 calls
  Action call_flop;
  call_flop.set_action(ActionType::CALL);
  call_flop.set_amount(5);
  
  if (simulator.ApplyAction(call_flop, state)) {
    PrintAction(call_flop, 0);
    PrintBoard(state);
  }
  
  // Advance to turn
  if (simulator.AdvanceIfReady(state)) {
    std::cout << "\n--- Turn ---" << std::endl;
    PrintBoard(state);
  }
  
  // Both check
  Action check_turn1;
  check_turn1.set_action(ActionType::CHECK);
  if (simulator.ApplyAction(check_turn1, state)) {
    PrintAction(check_turn1, 1);
  }
  
  Action check_turn2;
  check_turn2.set_action(ActionType::CHECK);
  if (simulator.ApplyAction(check_turn2, state)) {
    PrintAction(check_turn2, 0);
  }
  
  // Advance to river
  if (simulator.AdvanceIfReady(state)) {
    std::cout << "\n--- River ---" << std::endl;
    PrintBoard(state);
  }
  
  // Both check to showdown
  Action check_river1;
  check_river1.set_action(ActionType::CHECK);
  if (simulator.ApplyAction(check_river1, state)) {
    PrintAction(check_river1, 1);
  }
  
  Action check_river2;
  check_river2.set_action(ActionType::CHECK);
  if (simulator.ApplyAction(check_river2, state)) {
    PrintAction(check_river2, 0);
  }
  
  // Check if terminal and run showdown
  if (simulator.IsTerminal(state)) {
    std::cout << "\n--- Showdown ---" << std::endl;
    auto result = simulator.Showdown(state, p0_hole, p1_hole);
    
    PrintHand(p0_hole, "P0 shows");
    PrintHand(p1_hole, "P1 shows");
    
    if (result.split) {
      std::cout << "Split pot!" << std::endl;
    } else if (result.winner == 0) {
      std::cout << "P0 wins!" << std::endl;
    } else if (result.winner == 1) {
      std::cout << "P1 wins!" << std::endl;
    }
  }
  
  std::cout << "\n=== Test Complete ===" << std::endl;
}

}  // namespace poker

int main() {
  poker::RunSimpleGame();
  return 0;
}
