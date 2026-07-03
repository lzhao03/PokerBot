# No-Limit Hold'em Simulator

A minimal, isolated no-limit hold'em poker simulator with built-in hand evaluation.

## Features

- **Isolated Design**: Only depends on protobuf definitions from parent directory
- **Complete Game Logic**: Handles all poker actions, street progression, and showdowns
- **Built-in Hand Evaluation**: Self-contained hand ranking system (no external dependencies)
- **Configurable**: Adjustable stack sizes and blind levels

## Building

```bash
# Build the simulator library
bazel build //src/simulator:simulator

# Build and run the test
bazel build //src/simulator:simulator_test
./bazel-bin/src/simulator/simulator_test
```

## Usage

```cpp
#include "simulator.h"

// Create simulator with optional seed for reproducible results
NoLimitHoldemSimulator simulator(12345);

// Configure game parameters
NoLimitHoldemSimulator::Config config;
config.starting_stack = 100;
config.small_blind = 1;
config.big_blind = 2;

// Game state and hole cards
BoardState state;
Hand p0_hole, p1_hole;

// Start a new hand
simulator.StartNewHand(config, state, p0_hole, p1_hole);

// Apply actions
Action action;
action.set_action(ActionType::CALL);
action.set_amount(2);
simulator.ApplyAction(action, state);

// Advance streets when betting is complete
simulator.AdvanceIfReady(state);

// Check if hand is finished
if (simulator.IsTerminal(state)) {
    auto result = simulator.Showdown(state, p0_hole, p1_hole);
    // result.winner: 0, 1, or -1 (split)
    // result.split: true if pot is split
}
```

## API Reference

### NoLimitHoldemSimulator

#### Core Methods

- `StartNewHand(config, state, p0_hole, p1_hole)` - Initialize new hand with shuffled deck and posted blinds
- `ApplyAction(action, state)` - Process player action (fold, check, call, bet, raise, all-in)
- `AdvanceIfReady(state)` - Deal community cards and advance to next street if betting round complete
- `IsTerminal(state)` - Check if hand is finished (fold or river showdown)
- `Showdown(state, p0_hole, p1_hole)` - Determine winner using built-in hand evaluation

#### Configuration

```cpp
struct Config {
    int starting_stack = 100;   // chips per player
    int small_blind = 1;
    int big_blind = 2;
};
```

#### Result

```cpp
struct Result {
    bool terminal = false;
    int winner = -1;         // -1 if split, 0 or 1 for player
    bool split = false;      // true if pot split
};
```

## Hand Evaluation

The simulator includes a complete hand evaluation system that:

- Supports all standard poker hands (high card through royal flush)
- Handles tie-breaking with proper kicker comparison
- Works with any combination of hole cards + board cards
- Finds the best 5-card hand from available cards

## Example Output

```
=== No-Limit Hold'em Simulator Test ===

--- Hand Started ---
P0 Hole: 9♣ 6♣ 
P1 Hole: K♣ 5♥ 
Board: 
Pot: 3 | P0 Stack: 99 | P1 Stack: 98 | Street: Preflop | To Act: P0

--- Preflop Action ---
P0 calls 1
P1 checks

--- Flop ---
Board: 2♠ 8♠ 5♠ 
P1 bets 5
P0 calls 5

--- Showdown ---
P0 shows: 9♣ 6♣ 
P1 shows: K♣ 5♥ 
P1 wins!
```

## Dependencies

- Protocol Buffers (for game state representation)
- C++14 or later
- Bazel build system

The simulator is completely isolated from other components in the poker solver project.
