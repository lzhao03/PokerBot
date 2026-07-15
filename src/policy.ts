import type { Action, Card, Game, LoggedAction } from "./poker";

const FINGERPRINT = "d134d372f54a6e10eee5ce9fc74c411da09281145be34c32267c8160862d1ca5";
const FRACTIONS = [0.25, 0.5, 1];

type Player = 0 | 1;
type Kind = "bet" | "fold" | "call" | "raise" | "check" | "all-in";
type Street = 0 | 1 | 2 | 3;

interface BettingData {
  stack: [number, number];
  total: [number, number];
  committed: [number, number];
  lastRaise: number;
  street: Street;
  pending: number;
}

interface DecisionState {
  type: "decision";
  data: BettingData;
  actor: Player;
}

interface ActionEdge {
  kind: Kind;
  target: number;
  child: number;
}

type State = DecisionState | { type: "chance"; data: BettingData } | { type: "terminal"; data: BettingData };
type Node = { type: "decision"; actions: ActionEdge[] } | { type: "chance"; child: number } | { type: "terminal" };

interface PolicyRow {
  offset: number;
  count: number;
}

export interface Policy {
  rows: Map<string, PolicyRow>;
  probabilities: Float32Array;
}

export interface PolicyMove {
  action: Action;
  raiseTo?: number;
  found: boolean;
}

const other = (player: Player): Player => (player === 0 ? 1 : 0);
const copyData = (data: BettingData): BettingData => ({
  ...data,
  stack: [...data.stack],
  total: [...data.total],
  committed: [...data.committed]
});
const wager = (data: BettingData): number => Math.max(...data.committed);
const toCall = (data: BettingData, player: Player): number => wager(data) - data.committed[player];
const allInTo = (data: BettingData, player: Player): number =>
  data.committed[player] + Math.min(data.stack[player], toCall(data, player) + data.stack[other(player)]);

function abstractActions(state: DecisionState): Omit<ActionEdge, "child">[] {
  const { data, actor } = state;
  const current = data.committed[actor];
  const highest = wager(data);
  const call = toCall(data, actor);
  const callTo = Math.min(highest, current + data.stack[actor]);
  const allIn = allInTo(data, actor);
  const minimum = (highest > 0 ? highest : current) + data.lastRaise;
  const actions: Omit<ActionEdge, "child">[] = call > 0
    ? [{ kind: "fold", target: 0 }, { kind: "call", target: callTo }]
    : [{ kind: "check", target: 0 }];
  const aggressive: Kind = highest > 0 ? "raise" : "bet";
  const potAfterCall = data.total[0] + data.total[1] + call;

  for (const fraction of FRACTIONS) {
    const target = highest + Math.ceil(fraction * potAfterCall);
    if (target >= minimum && target < allIn && actions.at(-1)?.target !== target) {
      actions.push({ kind: aggressive, target });
    }
  }
  if (allIn > callTo) actions.push({ kind: "all-in", target: allIn });
  return actions;
}

function roundOver(data: BettingData): boolean {
  const matched = data.committed[0] === data.committed[1];
  if (data.pending === 0 && matched) return true;
  if (data.stack[0] > 0 && data.stack[1] > 0) return false;
  if (data.stack[0] === 0 && data.stack[1] === 0) return true;
  const live: Player = data.stack[0] > 0 ? 0 : 1;
  return toCall(data, live) === 0;
}

function apply(state: DecisionState, action: Omit<ActionEdge, "child">): State {
  if (action.kind === "fold") return { type: "terminal", data: state.data };
  const data = copyData(state.data);
  const actor = state.actor;
  const opponent = other(actor);
  const highest = wager(data);
  const delta = action.target - data.committed[actor];
  if (!["check"].includes(action.kind)) {
    const paid = Math.min(delta, data.stack[actor]);
    data.stack[actor] -= paid;
    data.total[actor] += paid;
    data.committed[actor] += paid;
  }
  const aggressive = action.target > highest;
  const raise = data.committed[actor] - highest;
  if (aggressive && raise >= data.lastRaise) data.lastRaise = raise;
  data.pending = aggressive ? 1 << opponent : data.pending & ~(1 << actor);
  if (!roundOver(data)) return { type: "decision", data, actor: opponent };

  if (data.committed[0] !== data.committed[1]) {
    const larger: Player = data.committed[0] > data.committed[1] ? 0 : 1;
    const refund = data.committed[larger] - data.committed[other(larger)];
    data.committed[larger] -= refund;
    data.total[larger] -= refund;
    data.stack[larger] += refund;
  }
  return data.street === 3 ? { type: "terminal", data } : { type: "chance", data };
}

function afterChance(state: Extract<State, { type: "chance" }>): State {
  const data = copyData(state.data);
  data.street = (data.street + 1) as Street;
  data.committed = [0, 0];
  data.lastRaise = 2;
  data.pending = 3;
  if (roundOver(data)) return data.street === 3 ? { type: "terminal", data } : { type: "chance", data };
  return { type: "decision", data, actor: 1 };
}

function buildTree(): Node[] {
  const nodes: Node[] = [];
  const append = (state: State): number => {
    const id = nodes.length;
    nodes.push({ type: "terminal" });
    if (state.type === "decision") {
      const node: Extract<Node, { type: "decision" }> = { type: "decision", actions: [] };
      nodes[id] = node;
      node.actions = abstractActions(state).map((action) => ({ ...action, child: append(apply(state, action)) }));
    } else if (state.type === "chance") {
      nodes[id] = { type: "chance", child: append(afterChance(state)) };
    }
    return id;
  };
  append({
    type: "decision",
    actor: 0,
    data: { stack: [7, 6], total: [1, 2], committed: [1, 2], lastRaise: 2, street: 0, pending: 3 }
  });
  return nodes;
}

const TREE = buildTree();

function decisionFor(actions: LoggedAction[]): { history: number; actions: ActionEdge[] } {
  let history = 0;
  const skipChance = (): void => {
    while (TREE[history]?.type === "chance") history = (TREE[history] as Extract<Node, { type: "chance" }>).child;
  };
  for (const logged of actions) {
    skipChance();
    const node = TREE[history];
    if (node?.type !== "decision") throw new Error("Policy history reached a terminal state");
    const edge = node.actions.find((candidate) => {
      if (logged.action === "raise") return ["bet", "raise", "all-in"].includes(candidate.kind) && candidate.target === logged.raiseTo;
      return candidate.kind === logged.action;
    });
    if (!edge) throw new Error(`Action ${logged.action} is outside the policy abstraction`);
    history = edge.child;
  }
  skipChance();
  const node = TREE[history];
  if (node?.type !== "decision") throw new Error("Policy has no decision at this state");
  return { history, actions: node.actions };
}

const rank = (card: Card): number => "23456789TJQKA".indexOf(card[0]) + 2;
const suit = (card: Card): string => card[1];

function features(cards: Card[]): { ranks: number[]; suits: Map<string, number>; mask: number; maxRank: number } {
  const ranks = Array(13).fill(0) as number[];
  const suits = new Map<string, number>();
  let mask = 0;
  let maxRank = 0;
  for (const card of cards) {
    const value = rank(card);
    ranks[value - 2] += 1;
    suits.set(suit(card), (suits.get(suit(card)) ?? 0) + 1);
    mask |= 1 << (value - 2);
    maxRank = Math.max(maxRank, value);
  }
  return { ranks, suits, mask, maxRank };
}

function straightDensity(mask: number): number {
  let best = 0;
  for (let start = 0; start <= 8; start += 1) best = Math.max(best, bitCount((mask >> start) & 0x1f));
  return Math.max(best, bitCount(mask & ((1 << 12) | 0x0f)));
}

function bitCount(value: number): number {
  let count = 0;
  for (; value; value &= value - 1) count += 1;
  return count;
}

function boardBucket(cards: Card[]): number {
  const value = features(cards);
  const maxRankCount = Math.max(...value.ranks);
  const maxSuitCount = Math.max(0, ...value.suits.values());
  const paired = maxRankCount >= 3 ? 2 : maxRankCount === 2 ? 1 : 0;
  const suited = maxSuitCount >= 4 ? 3 : maxSuitCount >= 3 ? 2 : maxSuitCount === 2 ? 1 : 0;
  const density = straightDensity(value.mask);
  const straight = density >= 4 ? 2 : density >= 3 ? 1 : 0;
  const high = value.maxRank >= 14 ? 0 : value.maxRank >= 11 ? 1 : 2;
  return (((paired * 4 + suited) * 3 + straight) * 3) + high;
}

function publicObservation(board: Card[]): bigint {
  if (board.length === 0) return 0n;
  let value = BigInt(boardBucket(board.slice(0, 3)) + 1);
  if (board.length >= 4) value |= BigInt(boardBucket(board.slice(0, 4)) + 1) << 7n;
  if (board.length >= 5) value |= BigInt(boardBucket(board) + 1) << 14n;
  return value;
}

function privateObservation(hand: Card[], board: Card[]): bigint {
  const high = Math.max(...hand.map(rank));
  const low = Math.min(...hand.map(rank));
  const pair = high === low;
  const suited = suit(hand[0]) === suit(hand[1]);
  if (board.length === 0) {
    const shape = pair ? 0 : suited ? 1 : 2;
    const highGroup = high >= 14 ? 0 : high >= 12 ? 1 : high >= 9 ? 2 : 3;
    const lowGroup = low >= 10 ? 0 : low >= 7 ? 1 : 2;
    return BigInt(shape * 12 + highGroup * 3 + lowGroup + 1);
  }

  const value = features([...board, ...hand]);
  const pairs = value.ranks.filter((count) => count >= 2).length;
  const maxCount = Math.max(...value.ranks);
  const made = maxCount >= 3 ? 3 : pairs >= 2 ? 2 : pairs === 1 ? 1 : 0;
  const flushDraw = [...value.suits.values()].some((count) => count >= 4);
  const draw = flushDraw ? 2 : straightDensity(value.mask) >= 4 ? 1 : 0;
  const gap = high - low;
  const strength = pair || high === 14 || (high >= 13 && low >= 10)
    ? 0
    : (high >= 11 && low >= 8) || (suited && gap <= 2) ? 1 : 2;
  return BigInt(made * 9 + draw * 3 + strength + 1);
}

const key = (history: number, publicId: bigint, privateId: bigint): string => `${history}/${publicId}/${privateId}`;

export function parsePolicy(buffer: ArrayBuffer): Policy {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (bytes.length < 60 || String.fromCharCode(...bytes.slice(0, 8)) !== "PKPOLCY1" || view.getUint32(8, true) !== 1) {
    throw new Error("Invalid policy header");
  }
  const fingerprint = [...bytes.slice(12, 44)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  if (fingerprint !== FINGERPRINT) throw new Error("Policy does not match the 8-chip game model");
  const rowCount = Number(view.getBigUint64(44, true));
  const probabilityCount = Number(view.getBigUint64(52, true));
  const probabilityStart = 60 + rowCount * 29;
  if (!Number.isSafeInteger(rowCount) || !Number.isSafeInteger(probabilityCount) || probabilityStart + probabilityCount * 4 !== bytes.length) {
    throw new Error("Invalid policy size");
  }

  const rows = new Map<string, PolicyRow>();
  const spans: PolicyRow[] = [];
  let cursor = 60;
  for (let index = 0; index < rowCount; index += 1) {
    const history = view.getUint32(cursor, true);
    const publicId = view.getBigUint64(cursor + 4, true);
    const privateId = view.getBigUint64(cursor + 12, true);
    const offset = Number(view.getBigUint64(cursor + 20, true));
    const count = view.getUint8(cursor + 28);
    const node = TREE[history];
    if (node?.type !== "decision" || count !== node.actions.length || offset + count > probabilityCount) throw new Error("Invalid policy row");
    const rowKey = key(history, publicId, privateId);
    if (rows.has(rowKey)) throw new Error("Duplicate policy row");
    const row = { offset, count };
    rows.set(rowKey, row);
    spans.push(row);
    cursor += 29;
  }
  const probabilities = new Float32Array(probabilityCount);
  for (let index = 0; index < probabilityCount; index += 1) probabilities[index] = view.getFloat32(cursor + index * 4, true);
  spans.sort((left, right) => left.offset - right.offset);
  let expectedOffset = 0;
  for (const row of spans) {
    if (row.offset !== expectedOffset) throw new Error("Non-contiguous policy rows");
    let sum = 0;
    for (let index = row.offset; index < row.offset + row.count; index += 1) {
      const probability = probabilities[index];
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("Invalid policy probability");
      sum += probability;
    }
    if (Math.abs(sum - 1) > 1e-5) throw new Error("Policy row is not normalized");
    expectedOffset += row.count;
  }
  if (expectedOffset !== probabilityCount) throw new Error("Policy probability count does not match rows");
  return { rows, probabilities };
}

export async function loadPolicy(url = "/pokerbot.policy"): Promise<Policy> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load policy: ${response.status}`);
  return parsePolicy(await response.arrayBuffer());
}

export function policyMove(policy: Policy, game: Game, random: () => number = Math.random): PolicyMove {
  const decision = decisionFor(game.actions);
  const hand = game.holes[game.toAct];
  const row = policy.rows.get(key(decision.history, publicObservation(game.board), privateObservation(hand, game.board)));
  const found = row?.count === decision.actions.length;
  let roll = random();
  let chosen = decision.actions.length - 1;
  for (let index = 0; index < decision.actions.length; index += 1) {
    roll -= found ? policy.probabilities[row.offset + index] : 1 / decision.actions.length;
    if (roll <= 0) {
      chosen = index;
      break;
    }
  }
  const edge = decision.actions[chosen];
  if (["bet", "raise", "all-in"].includes(edge.kind)) return { action: "raise", raiseTo: edge.target, found };
  return { action: edge.kind as Action, found };
}

export function policyRaiseTo(game: Game): number | null {
  return decisionFor(game.actions).actions.find((edge) => ["bet", "raise", "all-in"].includes(edge.kind))?.target ?? null;
}

export const policyHistoryNodeCount = TREE.length;
