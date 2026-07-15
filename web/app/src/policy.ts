import type { Action, Card, Game, LoggedAction } from "./poker";

const MODEL_LOW = 0xa150904f;
const MODEL_HIGH = 0x5bf84653;
const FRACTIONS = [0.5, 1];
const MAX_ACTIONS = 8;

type Player = 0 | 1;
type Kind = "bet" | "fold" | "call" | "raise" | "check" | "all-in";
type Street = 0 | 1 | 2 | 3;

interface BettingData {
  stack: [number, number];
  total: [number, number];
  committed: [number, number];
  lastRaise: number;
  street: Street;
  actionsRemaining: number;
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

interface Decoder {
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
  _poker_allocate(size: number): number;
  _poker_free(pointer: number): void;
  _poker_load_policy(pointer: number, size: number): number;
  _poker_strategy(
    publicLow: number,
    publicHigh: number,
    history: number,
    privateObservation: number,
    actionCount: number,
    output: number
  ): number;
  _poker_model_low(): number;
  _poker_model_high(): number;
}

export interface Policy {
  decoder: Decoder;
  output: number;
}

export interface PolicyAction {
  action: Action;
  raiseTo?: number;
  allIn?: boolean;
}

export interface PolicyMove extends PolicyAction {
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
  if (data.actionsRemaining === 0 && matched) return true;
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
  data.actionsRemaining = aggressive ? 1 : data.actionsRemaining - 1;
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
  data.actionsRemaining = 2;
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
    data: { stack: [199, 198], total: [1, 2], committed: [1, 2], lastRaise: 2, street: 0, actionsRemaining: 2 }
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

export function publicObservation(board: Card[]): bigint {
  const buckets = [16, 16, 64];
  let observation = 0n;
  for (let index = 2; index < board.length; index += 1) {
    const bucket = Math.floor(boardBucket(board.slice(0, index + 1)) * buckets[index - 2] / 108);
    observation |= BigInt(bucket + 1) << BigInt((index - 2) * 7);
  }
  return observation;
}

export function privateObservation(hand: Card[], board: Card[]): number {
  const high = Math.max(...hand.map(rank));
  const low = Math.min(...hand.map(rank));
  const pair = high === low;
  const suited = suit(hand[0]) === suit(hand[1]);
  if (board.length === 0) {
    const shape = pair ? 0 : suited ? 1 : 2;
    const highGroup = high >= 14 ? 0 : high >= 12 ? 1 : high >= 9 ? 2 : 3;
    const lowGroup = low >= 10 ? 0 : low >= 7 ? 1 : 2;
    return shape * 12 + highGroup * 3 + lowGroup + 1;
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
  return made * 9 + draw * 3 + strength + 1;
}

export async function loadPolicy(url = "/pokerbot.policy"): Promise<Policy> {
  const decoderPromise: Promise<Decoder> = import("@poker/policy_decoder")
    .then(async (module) => (await module.default()) as Decoder);
  const [decoder, response] = await Promise.all([
    decoderPromise,
    fetch(url)
  ]);
  if (!response.ok) throw new Error(`Could not load policy: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const pointer = decoder._poker_allocate(bytes.length);
  if (!pointer) throw new Error("Could not allocate policy memory");
  try {
    decoder.HEAPU8.set(bytes, pointer);
    if (decoder._poker_load_policy(pointer, bytes.length) !== 1) throw new Error("Invalid compact policy");
  } finally {
    decoder._poker_free(pointer);
  }
  if ((decoder._poker_model_low() >>> 0) !== MODEL_LOW || (decoder._poker_model_high() >>> 0) !== MODEL_HIGH) {
    throw new Error("Policy does not match the 100BB game model");
  }
  const output = decoder._poker_allocate(MAX_ACTIONS * Float32Array.BYTES_PER_ELEMENT);
  if (!output) throw new Error("Could not allocate strategy memory");
  return { decoder, output };
}

const moveFor = (edge: ActionEdge): PolicyAction => ["bet", "raise", "all-in"].includes(edge.kind)
  ? { action: "raise", raiseTo: edge.target, ...(edge.kind === "all-in" ? { allIn: true } : {}) }
  : { action: edge.kind as Action };

export function policyActions(game: Game): PolicyAction[] {
  return decisionFor(game.actions).actions.map(moveFor);
}

export function policyMove(policy: Policy | null, game: Game, random: () => number = Math.random): PolicyMove {
  const decision = decisionFor(game.actions);
  let probabilities: Float32Array | null = null;
  let found = false;
  if (policy) {
    const publicId = publicObservation(game.board);
    const result = policy.decoder._poker_strategy(
      Number(publicId & 0xffffffffn),
      Number(publicId >> 32n),
      decision.history,
      privateObservation(game.holes[game.toAct], game.board),
      decision.actions.length,
      policy.output
    );
    if (result < 0) throw new Error("Policy decoder rejected the game state");
    found = result === 1;
    probabilities = policy.decoder.HEAPF32.subarray(
      policy.output / Float32Array.BYTES_PER_ELEMENT,
      policy.output / Float32Array.BYTES_PER_ELEMENT + decision.actions.length
    );
  }
  let roll = random();
  let chosen = decision.actions.length - 1;
  for (let index = 0; index < decision.actions.length; index += 1) {
    roll -= probabilities?.[index] ?? 1 / decision.actions.length;
    if (roll <= 0) {
      chosen = index;
      break;
    }
  }
  return { ...moveFor(decision.actions[chosen]), found };
}

export const policyHistoryNodeCount = TREE.length;
