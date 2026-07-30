import type { Action, Card, Game, Suit } from "./poker";

const MAX_LOGGED_ACTIONS = 64;
const MAX_ACTIONS = 8;
const MAX_CARDS = 7;
const INPUT_KINDS = 0;
const INPUT_TARGETS = INPUT_KINDS + MAX_LOGGED_ACTIONS;
const CARDS = INPUT_TARGETS + MAX_LOGGED_ACTIONS * Int32Array.BYTES_PER_ELEMENT;
const OUTPUT_KINDS = CARDS + MAX_CARDS;
const OUTPUT_TARGETS = (OUTPUT_KINDS + MAX_ACTIONS + 3) & ~3;
const OUTPUT_PROBABILITIES =
  OUTPUT_TARGETS + MAX_ACTIONS * Int32Array.BYTES_PER_ELEMENT;
const SCRATCH_SIZE =
  OUTPUT_PROBABILITIES + MAX_ACTIONS * Float32Array.BYTES_PER_ELEMENT;

const policyKinds = { uniform: 0, tabular: 1, neural: 2 } as const;
const actionKinds: Record<Action, number> = {
  fold: 1,
  call: 2,
  raise: 3,
  check: 4
};
const suitOffsets: Record<Suit, number> = {
  H: 0,
  D: 13,
  C: 26,
  S: 39
};

interface Decoder {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPF32: Float32Array;
  _poker_allocate(size: number): number;
  _poker_free(pointer: number): void;
  _poker_load_policy(pointer: number, size: number): number;
  _poker_load_neural_policy(pointer: number, size: number): number;
  _poker_query(
    policyKind: number,
    inputKinds: number,
    inputTargets: number,
    inputCount: number,
    cards: number,
    boardCount: number,
    outputKinds: number,
    outputTargets: number,
    outputProbabilities: number
  ): number;
  _poker_query_found(): number;
}

interface Runtime {
  decoder: Decoder;
  scratch: number;
}

export interface Policy {
  kind: "tabular" | "neural";
  runtime: Runtime;
}

export interface PolicyAction {
  action: Action;
  raiseTo?: number;
  allIn?: boolean;
}

export interface PolicyMove extends PolicyAction {
  found: boolean;
}

interface QueryResult {
  actions: PolicyAction[];
  probabilities: number[];
  found: boolean;
}

let runtimePromise: Promise<Runtime> | undefined;

function loadRuntime(): Promise<Runtime> {
  return runtimePromise ??= import("@poker/policy_decoder").then(async (module) => {
    const decoder = (await module.default()) as Decoder;
    const scratch = decoder._poker_allocate(SCRATCH_SIZE);
    if (!scratch) throw new Error("Could not allocate policy query memory");
    return { decoder, scratch };
  });
}

async function loadPolicy(
  kind: Policy["kind"],
  url: string
): Promise<Policy> {
  const [runtime, response] = await Promise.all([loadRuntime(), fetch(url)]);
  if (!response.ok) throw new Error(`Could not load policy: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const pointer = runtime.decoder._poker_allocate(bytes.length);
  if (!pointer) throw new Error("Could not allocate policy memory");
  try {
    runtime.decoder.HEAPU8.set(bytes, pointer);
    const loaded = kind === "tabular"
      ? runtime.decoder._poker_load_policy(pointer, bytes.length)
      : runtime.decoder._poker_load_neural_policy(pointer, bytes.length);
    if (loaded !== 1) throw new Error(`Invalid ${kind} policy`);
  } finally {
    runtime.decoder._poker_free(pointer);
  }
  return { kind, runtime };
}

export function loadTabularPolicy(url = "/pokerbot.policy"): Promise<Policy> {
  return loadPolicy("tabular", url);
}

export function loadNeuralPolicy(url = "/deep-cfr.pnn"): Promise<Policy> {
  return loadPolicy("neural", url);
}

const cardIndex = (card: Card): number =>
  suitOffsets[card[1] as Suit] + "23456789TJQKA".indexOf(card[0]);

function moveFor(kind: number, target: number): PolicyAction {
  if (kind === 0 || kind === 3) return { action: "raise", raiseTo: target };
  if (kind === 5) return { action: "raise", raiseTo: target, allIn: true };
  if (kind === 1) return { action: "fold" };
  if (kind === 2) return { action: "call" };
  if (kind === 4) return { action: "check" };
  throw new Error("Policy decoder returned an unknown action");
}

function query(
  runtime: Runtime,
  kind: number,
  game: Game
): QueryResult {
  const { decoder, scratch } = runtime;
  if (game.actions.length > MAX_LOGGED_ACTIONS) {
    throw new Error("Policy history is too long");
  }
  decoder.HEAPU8.set(
    game.actions.map(({ action }) => actionKinds[action]),
    scratch + INPUT_KINDS
  );
  const targetBegin = (scratch + INPUT_TARGETS) / Int32Array.BYTES_PER_ELEMENT;
  game.actions.forEach(({ raiseTo }, index) => {
    decoder.HEAP32[targetBegin + index] = raiseTo ?? 0;
  });
  decoder.HEAPU8.set(
    [...game.holes[game.toAct], ...game.board].map(cardIndex),
    scratch + CARDS
  );

  const actionCount = decoder._poker_query(
    kind,
    scratch + INPUT_KINDS,
    scratch + INPUT_TARGETS,
    game.actions.length,
    scratch + CARDS,
    game.board.length,
    scratch + OUTPUT_KINDS,
    scratch + OUTPUT_TARGETS,
    scratch + OUTPUT_PROBABILITIES
  );
  if (actionCount < 0) {
    throw new Error("Game state is outside the policy abstraction");
  }

  const targetOutput =
    (scratch + OUTPUT_TARGETS) / Int32Array.BYTES_PER_ELEMENT;
  const probabilityOutput =
    (scratch + OUTPUT_PROBABILITIES) / Float32Array.BYTES_PER_ELEMENT;
  return {
    actions: Array.from({ length: actionCount }, (_, index) =>
      moveFor(
        decoder.HEAPU8[scratch + OUTPUT_KINDS + index],
        decoder.HEAP32[targetOutput + index]
      )
    ),
    probabilities: Array.from({ length: actionCount }, (_, index) =>
      decoder.HEAPF32[probabilityOutput + index]
    ),
    found: decoder._poker_query_found() === 1
  };
}

export function policyActions(policy: Policy, game: Game): PolicyAction[] {
  return query(policy.runtime, policyKinds.uniform, game).actions;
}

export function policyMove(
  policy: Policy,
  game: Game,
  random: () => number = Math.random
): PolicyMove {
  const result = query(policy.runtime, policyKinds[policy.kind], game);
  let roll = random();
  let chosen = result.actions.length - 1;
  for (let index = 0; index < result.actions.length; index += 1) {
    roll -= result.probabilities[index];
    if (roll <= 0) {
      chosen = index;
      break;
    }
  }
  return { ...result.actions[chosen], found: result.found };
}
