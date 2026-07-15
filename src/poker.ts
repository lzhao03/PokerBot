export type Rank = "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "T" | "J" | "Q" | "K" | "A";
export type Suit = "S" | "H" | "D" | "C";
export type Card = `${Rank}${Suit}`;
export type Player = 0 | 1;
export type Action = "fold" | "check" | "call" | "raise";
export type Street = "preflop" | "flop" | "turn" | "river";

export interface LoggedAction {
  action: Action;
  raiseTo?: number;
  player: Player;
  street: Street;
}

type Pair<T> = [T, T];

export interface HandScore {
  category: number;
  values: number[];
  name: string;
}

export interface Game {
  deck: Card[];
  holes: Pair<Card[]>;
  board: Card[];
  startingStacks: Pair<number>;
  stacks: Pair<number>;
  bets: Pair<number>;
  pot: number;
  dealer: Player;
  toAct: Player;
  street: Street;
  currentBet: number;
  minRaise: number;
  acted: Pair<boolean>;
  message: string;
  winner: Player[] | null;
  showdown: boolean;
  actions: LoggedAction[];
}

export interface LegalActions {
  toCall: number;
  canFold: boolean;
  canCheck: boolean;
  canCall: boolean;
  minRaiseTo: number | null;
  maxRaiseTo: number;
}

const RANKS: Rank[] = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
const SUITS: Suit[] = ["S", "H", "D", "C"];
const SMALL_BLIND = 1;
const BIG_BLIND = 2;
const STARTING_STACK = BIG_BLIND * 100;
const HANDS = [
  "high card",
  "pair",
  "two pair",
  "three of a kind",
  "straight",
  "flush",
  "full house",
  "four of a kind",
  "straight flush"
];

const other = (player: Player): Player => (player === 0 ? 1 : 0);
const rank = (card: Card): number => RANKS.indexOf(card[0] as Rank) + 2;
const uniqueDesc = (ranks: number[]): number[] => [...new Set(ranks)].sort((a, b) => b - a);

function deck(): Card[] {
  return RANKS.flatMap((r) => SUITS.map((s) => `${r}${s}` as Card));
}

function shuffle(cards: Card[]): Card[] {
  const out = [...cards];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function straightHigh(ranks: number[]): number {
  const values = uniqueDesc(ranks);
  if (values.includes(14)) values.push(1);
  for (let i = 0; i <= values.length - 5; i += 1) {
    if (values[i] - values[i + 4] === 4) return values[i];
  }
  return 0;
}

const score = (category: number, values: number[]): HandScore => ({ category, values, name: HANDS[category] });

export function bestHand(cards: Card[]): HandScore {
  const ranks = cards.map(rank).sort((a, b) => b - a);
  const counts = new Map(ranks.map((r) => [r, ranks.filter((x) => x === r).length]));
  const group = (n: number): number[] => [...counts].filter(([, count]) => count >= n).map(([r]) => r).sort((a, b) => b - a);
  const kickers = (skip: number[], n: number): number[] => uniqueDesc(ranks).filter((r) => !skip.includes(r)).slice(0, n);
  const flushSuit = [...SUITS].find((s) => cards.filter((card) => card[1] === s).length >= 5);
  const flushRanks = flushSuit ? cards.filter((card) => card[1] === flushSuit).map(rank) : [];
  const straightFlush = straightHigh(flushRanks);
  const trips = group(3);
  const pairs = group(2);

  if (straightFlush) return score(8, [straightFlush]);
  if (group(4)[0]) return score(7, [group(4)[0], ...kickers(group(4), 1)]);
  if (trips[0] && (pairs.filter((r) => r !== trips[0])[0] || trips[1])) {
    return score(6, [trips[0], pairs.filter((r) => r !== trips[0])[0] || trips[1]]);
  }
  if (flushSuit) return score(5, uniqueDesc(flushRanks).slice(0, 5));
  if (straightHigh(ranks)) return score(4, [straightHigh(ranks)]);
  if (trips[0]) return score(3, [trips[0], ...kickers([trips[0]], 2)]);
  if (pairs[1]) return score(2, [pairs[0], pairs[1], ...kickers([pairs[0], pairs[1]], 1)]);
  if (pairs[0]) return score(1, [pairs[0], ...kickers([pairs[0]], 3)]);
  return score(0, uniqueDesc(ranks).slice(0, 5));
}

export function compareHands(a: Card[], b: Card[]): number {
  const left = bestHand(a);
  const right = bestHand(b);
  const values = [left.category, ...left.values].map((v, i) => v - [right.category, ...right.values][i]);
  return values.find(Boolean) || 0;
}

function clone(game: Game): Game {
  return {
    ...game,
    deck: [...game.deck],
    holes: [[...game.holes[0]], [...game.holes[1]]],
    board: [...game.board],
    startingStacks: [...game.startingStacks],
    stacks: [...game.stacks],
    bets: [...game.bets],
    acted: [...game.acted],
    actions: game.actions.map((action) => ({ ...action }))
  };
}

function pay(game: Game, player: Player, amount: number): void {
  const chips = Math.min(amount, game.stacks[player]);
  game.stacks[player] -= chips;
  game.bets[player] += chips;
  game.pot += chips;
}

function dealTo(game: Game, totalBoardCards: number): void {
  const count = totalBoardCards - game.board.length;
  game.board.push(...game.deck.slice(0, count));
  game.deck = game.deck.slice(count);
}

export function newHand(stacks: Pair<number> = [STARTING_STACK, STARTING_STACK], dealer: Player = 0): Game {
  const cards = shuffle(deck());
  const game: Game = {
    deck: cards.slice(4),
    holes: [cards.slice(0, 2), cards.slice(2, 4)],
    board: [],
    startingStacks: [...stacks],
    stacks: [...stacks],
    bets: [0, 0],
    pot: 0,
    dealer,
    toAct: dealer,
    street: "preflop",
    currentBet: 0,
    minRaise: BIG_BLIND,
    acted: [false, false],
    message: `Player ${dealer + 1} posts ${SMALL_BLIND}; Player ${other(dealer) + 1} posts ${BIG_BLIND}.`,
    winner: null,
    showdown: false,
    actions: []
  };
  pay(game, dealer, SMALL_BLIND);
  pay(game, other(dealer), BIG_BLIND);
  game.currentBet = Math.max(...game.bets);
  game.acted = [game.stacks[0] === 0, game.stacks[1] === 0];
  if (game.stacks[dealer] === 0 || (game.stacks[other(dealer)] === 0 && game.currentBet === game.bets[dealer])) {
    refundUncalled(game);
    return showdown(game);
  }
  return game;
}

export function nextHand(game: Game): Game {
  return newHand([STARTING_STACK, STARTING_STACK], other(game.dealer));
}

export function legalActions(game: Game): LegalActions {
  const player = game.toAct;
  const toCall = Math.max(0, game.currentBet - game.bets[player]);
  const maxRaiseTo = game.bets[player] + game.stacks[player];
  const canRaise = !game.winner && game.stacks[other(player)] > 0 && maxRaiseTo > game.currentBet;
  const fullRaiseTo = game.currentBet ? game.currentBet + game.minRaise : BIG_BLIND;
  return {
    toCall,
    canFold: !game.winner && toCall > 0,
    canCheck: !game.winner && toCall === 0,
    canCall: !game.winner && toCall > 0,
    minRaiseTo: canRaise ? Math.min(fullRaiseTo, maxRaiseTo) : null,
    maxRaiseTo
  };
}

export function botAction(game: Game, random: () => number = Math.random): Action {
  const legal = legalActions(game);
  const actions: Action[] = [];
  if (legal.canFold) actions.push("fold");
  if (legal.canCheck) actions.push("check");
  if (legal.canCall) actions.push("call");
  if (legal.minRaiseTo !== null) actions.push("raise");
  return actions[Math.floor(random() * actions.length)];
}

function refundUncalled(game: Game): void {
  if (game.bets[0] === game.bets[1]) return;
  const player = game.bets[0] > game.bets[1] ? 0 : 1;
  const refund = game.bets[player] - game.bets[other(player)];
  game.bets[player] -= refund;
  game.stacks[player] += refund;
  game.pot -= refund;
}

function showdown(game: Game): Game {
  dealTo(game, 5);
  game.showdown = true;
  const hands = game.holes.map((hand) => bestHand([...hand, ...game.board]));
  const result = compareHands([...game.holes[0], ...game.board], [...game.holes[1], ...game.board]);
  if (result === 0) {
    const half = Math.floor(game.pot / 2);
    game.stacks[0] += half + (game.dealer === 1 ? game.pot % 2 : 0);
    game.stacks[1] += half + (game.dealer === 0 ? game.pot % 2 : 0);
    game.message = `Split pot: both have ${hands[0].name}.`;
    game.winner = [0, 1];
  } else {
    const winner = result > 0 ? 0 : 1;
    game.stacks[winner] += game.pot;
    game.message = `Player ${winner + 1} wins with ${hands[winner].name}.`;
    game.winner = [winner];
  }
  game.pot = 0;
  return game;
}

function finishStreet(game: Game): Game {
  if (game.stacks.some((stack) => stack === 0)) {
    refundUncalled(game);
    return showdown(game);
  }
  if (game.street === "river") return showdown(game);
  const streets: Record<Exclude<Street, "river">, [Street, number]> = {
    preflop: ["flop", 3],
    flop: ["turn", 4],
    turn: ["river", 5]
  };
  const next = streets[game.street];
  game.street = next[0];
  dealTo(game, next[1]);
  game.bets = [0, 0];
  game.currentBet = 0;
  game.minRaise = BIG_BLIND;
  game.acted = [false, false];
  game.toAct = other(game.dealer);
  game.message = `${game.street[0].toUpperCase()}${game.street.slice(1)}. Player ${game.toAct + 1} acts.`;
  return game;
}

function roundDone(game: Game): boolean {
  return game.acted.every(Boolean) && (game.bets[0] === game.bets[1] || game.stacks.some((stack) => stack === 0));
}

export function act(game: Game, action: Action, raiseTo?: number): Game {
  if (game.winner) return game;
  const next = clone(game);
  const player = next.toAct;
  const opponent = other(player);
  const legal = legalActions(next);

  if (action === "fold") {
    if (!legal.canFold) throw new Error("Cannot fold when checking is available");
    next.stacks[opponent] += next.pot;
    next.message = `Player ${opponent + 1} wins by fold.`;
    next.pot = 0;
    next.winner = [opponent];
    next.actions.push({ action, player, street: next.street });
    return next;
  }

  if (action === "raise") {
    if (legal.minRaiseTo === null || raiseTo === undefined || raiseTo < legal.minRaiseTo || raiseTo > legal.maxRaiseTo) {
      throw new Error("Illegal raise amount");
    }
    const oldBet = next.currentBet;
    pay(next, player, raiseTo - next.bets[player]);
    const raiseSize = raiseTo - oldBet;
    if (raiseSize >= next.minRaise) next.minRaise = raiseSize;
    next.currentBet = raiseTo;
    next.acted[opponent] = false;
  } else if (action === "call") {
    if (!legal.canCall) throw new Error("Nothing to call");
    pay(next, player, legal.toCall);
  } else if (action === "check") {
    if (!legal.canCheck) throw new Error("Cannot check facing a bet");
  }

  next.acted[player] = true;
  next.actions.push({ action, player, street: next.street, ...(action === "raise" ? { raiseTo } : {}) });
  if (roundDone(next)) return finishStreet(next);
  next.toAct = opponent;
  next.message = `Player ${next.toAct + 1} acts.`;
  return next;
}
