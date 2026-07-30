import assert from "node:assert/strict";
import type { LoggedAction, Player } from "./poker";
import {
  bbPer100,
  emptyStats,
  recordHand,
  stdDevPer100
} from "./stats";

const completedHand = (
  stacks: [number, number],
  actions: LoggedAction[]
) => ({
  winner: [0] as Player[],
  stacks,
  startingStacks: [200, 200] as [number, number],
  actions
});

let stats = recordHand(emptyStats(), completedHand(
  [202, 198],
  [
    { action: "raise", raiseTo: 4, player: 0, street: "preflop" },
    { action: "fold", player: 1, street: "preflop" }
  ]
));
assert.equal(stats.hands, 1);
assert.equal(stats.netBb, 1);
assert.equal(stats.vpipHands, 1);
assert.equal(stats.pfrHands, 1);
assert.equal(stats.streets.preflop.blind.raise, 1);

stats = recordHand(stats, completedHand(
  [204, 196],
  [
    { action: "raise", raiseTo: 4, player: 1, street: "preflop" },
    { action: "raise", raiseTo: 6, player: 0, street: "preflop" },
    { action: "fold", player: 1, street: "preflop" }
  ]
));
assert.equal(stats.threeBetOpportunities, 1);
assert.equal(stats.threeBets, 1);
assert.equal(stats.streets.preflop.raise.raise, 1);
assert.equal(bbPer100(stats), 150);
assert.equal(stdDevPer100(stats), Math.sqrt(50));

stats = recordHand(stats, completedHand(
  [200, 200],
  [
    { action: "call", player: 0, street: "preflop" },
    { action: "check", player: 1, street: "preflop" },
    { action: "raise", raiseTo: 2, player: 1, street: "flop" },
    { action: "call", player: 0, street: "flop" }
  ]
));
assert.equal(stats.streets.flop.bet.call, 1);

stats = recordHand(stats, completedHand(
  [200, 200],
  [
    { action: "call", player: 1, street: "preflop" },
    { action: "check", player: 0, street: "preflop" }
  ]
));
assert.equal(stats.streets.preflop.call.check, 1);

console.log("ok");
