import assert from "node:assert/strict";
import { act, bestHand, botAction, compareHands, legalActions, newHand, nextHand, type Card } from "./poker";
import { policyActions, policyHistoryNodeCount, privateObservation, publicObservation } from "./policy";
import { bbPer100, emptyStats, recordHand, variance } from "./stats";

const cards = (text: string): Card[] => text.split(" ") as Card[];

assert.equal(bestHand(cards("AS KS QS JS TS 2C 3D")).name, "straight flush");
assert.equal(bestHand(cards("AS AH AD KC KD 2S 3S")).name, "full house");
assert.ok(compareHands(cards("AS AH 9C 8D 7S 2H 3C"), cards("KS KH QC JD 9S 2D 3H")) > 0);

let game = newHand([100, 100], 0);
assert.equal(game.pot, 3);
assert.equal(game.toAct, 0);
assert.deepEqual(legalActions(game), {
  toCall: 1,
  canFold: true,
  canCheck: false,
  canCall: true,
  minRaiseTo: 4,
  maxRaiseTo: 100
});
assert.throws(() => act(game, "check"));
game = act(game, "call");
assert.equal(game.bets[0], 2);
assert.equal(legalActions(game).canCheck, true);
game = act(game, "check");
assert.equal(game.street, "flop");
assert.equal(game.board.length, 3);
assert.equal(legalActions(game).minRaiseTo, 2);

game = newHand([100, 100], 1);
assert.equal(botAction(game, () => 0.1), "fold");
assert.equal(botAction(game, () => 0.5), "call");
assert.equal(botAction(game, () => 0.9), "raise");

game = newHand([100, 100], 0);
game = act(game, "raise", 4);
assert.equal(game.currentBet, 4);
assert.equal(game.minRaise, 2);
assert.equal(legalActions(game).minRaiseTo, 6);
assert.throws(() => act(game, "raise", 5));

game = newHand([100, 5], 0);
game = act(game, "raise", 4);
assert.equal(legalActions(game).minRaiseTo, 5);
game = act(game, "raise", 5);
assert.equal(game.minRaise, 2);
assert.equal(legalActions(game).minRaiseTo, null);
game = act(game, "call");
assert.equal(game.showdown, true);
assert.equal(game.stacks[0] + game.stacks[1], 105);

game = newHand([100, 3], 0);
game = act(game, "raise", 4);
game = act(game, "call");
assert.deepEqual(game.bets, [3, 3]);

game = newHand([100, 2], 0);
assert.equal(legalActions(game).toCall, 1);
assert.equal(legalActions(game).minRaiseTo, null);
game = act(game, "call");
assert.equal(game.showdown, true);

game = newHand([100, 1], 0);
assert.equal(game.showdown, true);
assert.deepEqual(game.bets, [1, 1]);
assert.equal(game.stacks[0] + game.stacks[1], 101);

game = newHand([100, 100], 0);
for (let turns = 0; !game.winner; turns += 1) {
  assert.ok(turns < 20);
  game = act(game, legalActions(game).canCheck ? "check" : "call");
}
assert.equal(game.board.length, 5);
assert.equal(game.stacks[0] + game.stacks[1], 200);

game = newHand();
assert.deepEqual(game.stacks, [199, 198]);
assert.equal(game.pot, 3);
game = act(game, "fold");
game = nextHand(game);
assert.equal(game.stacks[0] + game.stacks[1] + game.pot, 400);
assert.deepEqual(game.startingStacks, [200, 200]);

game = newHand([200, 200], 0);
assert.equal(policyHistoryNodeCount, 136689);
assert.deepEqual(policyActions(game), [
  { action: "fold" },
  { action: "call" },
  { action: "raise", raiseTo: 4 },
  { action: "raise", raiseTo: 6 },
  { action: "raise", raiseTo: 200, allIn: true }
]);
assert.equal(privateObservation(cards("AH AS"), []), 1);
assert.equal(privateObservation(cards("AH KH"), []), 13);
assert.equal(privateObservation(cards("7H 2S"), []), 36);
assert.equal(privateObservation(cards("AH KS"), cards("2H 7H QH")), 7);
assert.equal(publicObservation(cards("AH 9H 4C 7D 2S")), 131330n);

let stats = emptyStats();
game = newHand([200, 200], 0);
game = act(game, "raise", 4);
game = act(game, "fold");
stats = recordHand(stats, game);
assert.equal(stats.hands, 1);
assert.equal(stats.netBb, 1);
assert.equal(stats.vpipHands, 1);
assert.equal(stats.pfrHands, 1);
assert.equal(stats.streets.preflop.blind.raise, 1);

game = newHand([200, 200], 1);
game = act(game, "raise", 4);
game = act(game, "raise", 6);
game = act(game, "fold");
stats = recordHand(stats, game);
assert.equal(stats.threeBetOpportunities, 1);
assert.equal(stats.threeBets, 1);
assert.equal(stats.streets.preflop.raise.raise, 1);
assert.equal(bbPer100(stats), 150);
assert.equal(variance(stats), 0.5);

game = newHand([200, 200], 0);
game = act(game, "call");
game = act(game, "check");
game = act(game, "raise", 2);
game = act(game, "call");
while (!game.winner) game = act(game, legalActions(game).canCheck ? "check" : "call");
stats = recordHand(stats, game);
assert.equal(stats.streets.flop.bet.call, 1);

game = newHand([200, 200], 1);
game = act(game, "call");
game = act(game, "check");
while (!game.winner) game = act(game, "check");
stats = recordHand(stats, game);
assert.equal(stats.streets.preflop.call.check, 1);

console.log("ok");
