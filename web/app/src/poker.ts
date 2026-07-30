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

export interface GameAction {
  action: Action;
  raiseTo?: number;
  allIn?: boolean;
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
  toAct: Player | null;
  street: Street;
  currentBet: number;
  toCall: number;
  message: string;
  winner: Player[] | null;
  showdown: boolean;
  actions: LoggedAction[];
  options: GameAction[];
}

const MAX_LOGGED_ACTIONS = 64;
const MAX_ACTIONS = 8;
const MAX_CARDS = 9;
const INPUT_KINDS = 0;
const INPUT_TARGETS = INPUT_KINDS + MAX_LOGGED_ACTIONS;
const CARDS = INPUT_TARGETS + MAX_LOGGED_ACTIONS * Int32Array.BYTES_PER_ELEMENT;
const OUTPUT_KINDS = CARDS + MAX_CARDS;
const OUTPUT_TARGETS = (OUTPUT_KINDS + MAX_ACTIONS + 3) & ~3;
const OUTPUT_PROBABILITIES =
  OUTPUT_TARGETS + MAX_ACTIONS * Int32Array.BYTES_PER_ELEMENT;
const OUTPUT_STATE =
  OUTPUT_PROBABILITIES + MAX_ACTIONS * Float32Array.BYTES_PER_ELEMENT;
const STATE_FIELDS = 12;
const SCRATCH_SIZE =
  OUTPUT_STATE + STATE_FIELDS * Int32Array.BYTES_PER_ELEMENT;

const PHASE_FOLD = 2;
const PHASE_SHOWDOWN = 3;
const STATE = {
  phase: 0,
  street: 1,
  actor: 2,
  stack0: 3,
  stack1: 4,
  bet0: 5,
  bet1: 6,
  pot: 7,
  currentWager: 8,
  callAmount: 9,
  winnerMask: 10,
  cardsNeeded: 11
} as const;
const policyKinds = { tabular: 1, neural: 2 } as const;
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
const RANKS: Rank[] = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
const SUITS: Suit[] = ["S", "H", "D", "C"];
const STREETS: Street[] = ["preflop", "flop", "turn", "river"];
const STARTING_STACK = 200;

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
    dealer: number,
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
  _poker_replay(
    dealer: number,
    inputKinds: number,
    inputTargets: number,
    inputCount: number,
    cards: number,
    boardCount: number,
    outputState: number,
    outputKinds: number,
    outputTargets: number
  ): number;
}

export interface Runtime {
  decoder: Decoder;
  scratch: number;
}

export interface Policy {
  kind: "tabular" | "neural";
  runtime: Runtime;
}

export interface PolicyMove extends GameAction {
  found: boolean;
}

let runtimePromise: Promise<Runtime> | undefined;

export function loadRuntime(): Promise<Runtime> {
  return runtimePromise ??= import("@poker/policy_decoder").then(async (module) => {
    const decoder = (await module.default()) as Decoder;
    const scratch = decoder._poker_allocate(SCRATCH_SIZE);
    if (!scratch) throw new Error("Could not allocate poker memory");
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

function moveFor(kind: number, target: number): GameAction {
  if (kind === 0 || kind === 3) return { action: "raise", raiseTo: target };
  if (kind === 5) return { action: "raise", raiseTo: target, allIn: true };
  if (kind === 1) return { action: "fold" };
  if (kind === 2) return { action: "call" };
  if (kind === 4) return { action: "check" };
  throw new Error("Poker engine returned an unknown action");
}

function writeHistory(runtime: Runtime, game: Pick<Game, "actions">): void {
  if (game.actions.length > MAX_LOGGED_ACTIONS) {
    throw new Error("Poker history is too long");
  }
  runtime.decoder.HEAPU8.set(
    game.actions.map(({ action }) => actionKinds[action]),
    runtime.scratch + INPUT_KINDS
  );
  const targetBegin =
    (runtime.scratch + INPUT_TARGETS) / Int32Array.BYTES_PER_ELEMENT;
  game.actions.forEach(({ raiseTo }, index) => {
    runtime.decoder.HEAP32[targetBegin + index] = raiseTo ?? 0;
  });
}

function readActions(runtime: Runtime, count: number): GameAction[] {
  const targetBegin =
    (runtime.scratch + OUTPUT_TARGETS) / Int32Array.BYTES_PER_ELEMENT;
  return Array.from({ length: count }, (_, index) =>
    moveFor(
      runtime.decoder.HEAPU8[runtime.scratch + OUTPUT_KINDS + index],
      runtime.decoder.HEAP32[targetBegin + index]
    )
  );
}

function deck(): Card[] {
  return RANKS.flatMap((rank) => SUITS.map((suit) => `${rank}${suit}` as Card));
}

function shuffle(cards: Card[]): Card[] {
  const result = [...cards];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(Math.random() * (index + 1));
    [result[index], result[selected]] = [result[selected], result[index]];
  }
  return result;
}

type ReplayInput = Pick<
  Game,
  "deck" | "holes" | "board" | "startingStacks" | "dealer" | "actions"
>;

function replay(runtime: Runtime, input: ReplayInput): Game {
  const { decoder, scratch } = runtime;
  const deck = [...input.deck];
  const board = [...input.board];
  writeHistory(runtime, input);

  while (true) {
    decoder.HEAPU8.set(
      [...input.holes[0], ...input.holes[1], ...board].map(cardIndex),
      scratch + CARDS
    );
    const actionCount = decoder._poker_replay(
      input.dealer,
      scratch + INPUT_KINDS,
      scratch + INPUT_TARGETS,
      input.actions.length,
      scratch + CARDS,
      board.length,
      scratch + OUTPUT_STATE,
      scratch + OUTPUT_KINDS,
      scratch + OUTPUT_TARGETS
    );
    if (actionCount < 0) throw new Error("Invalid poker state");

    const stateBegin =
      (scratch + OUTPUT_STATE) / Int32Array.BYTES_PER_ELEMENT;
    const state = decoder.HEAP32.subarray(
      stateBegin,
      stateBegin + STATE_FIELDS
    );
    if (state[STATE.cardsNeeded] > 0) {
      board.push(...deck.splice(0, state[STATE.cardsNeeded]));
      continue;
    }

    const winnerMask = state[STATE.winnerMask];
    const winner = winnerMask === 0
      ? null
      : ([0, 1] as Player[]).filter((player) => winnerMask & (1 << player));
    const phase = state[STATE.phase];
    const message = winnerMask === 3
      ? "Split pot."
      : winner
        ? `Player ${winner[0] + 1} wins ${phase === PHASE_FOLD ? "by fold" : "at showdown"}.`
        : "";
    return {
      ...input,
      deck,
      board,
      stacks: [state[STATE.stack0], state[STATE.stack1]],
      bets: [state[STATE.bet0], state[STATE.bet1]],
      pot: state[STATE.pot],
      toAct: state[STATE.actor] < 0 ? null : state[STATE.actor] as Player,
      street: STREETS[state[STATE.street]],
      currentBet: state[STATE.currentWager],
      toCall: state[STATE.callAmount],
      message,
      winner,
      showdown: phase === PHASE_SHOWDOWN,
      options: readActions(runtime, actionCount)
    };
  }
}

export function newHand(runtime: Runtime, dealer: Player = 0): Game {
  const cards = shuffle(deck());
  return replay(runtime, {
    deck: cards.slice(4),
    holes: [cards.slice(0, 2), cards.slice(2, 4)],
    board: [],
    startingStacks: [STARTING_STACK, STARTING_STACK],
    dealer,
    actions: []
  });
}

export function nextHand(runtime: Runtime, game: Game): Game {
  return newHand(runtime, game.dealer === 0 ? 1 : 0);
}

export function act(
  runtime: Runtime,
  game: Game,
  action: Action,
  raiseTo?: number
): Game {
  if (game.winner || game.toAct === null) return game;
  if (!game.options.some((option) =>
    option.action === action && option.raiseTo === raiseTo)) {
    throw new Error("Illegal poker action");
  }
  return replay(runtime, {
    ...game,
    actions: [
      ...game.actions,
      {
        action,
        player: game.toAct,
        street: game.street,
        ...(raiseTo === undefined ? {} : { raiseTo })
      }
    ]
  });
}

export function policyMove(
  policy: Policy,
  game: Game,
  random: () => number = Math.random
): PolicyMove {
  if (game.toAct === null) throw new Error("Cannot query a terminal game");
  const { decoder, scratch } = policy.runtime;
  writeHistory(policy.runtime, game);
  decoder.HEAPU8.set(
    [...game.holes[0], ...game.holes[1], ...game.board].map(cardIndex),
    scratch + CARDS
  );
  const actionCount = decoder._poker_query(
    policyKinds[policy.kind],
    game.dealer,
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

  const actions = readActions(policy.runtime, actionCount);
  const probabilityBegin =
    (scratch + OUTPUT_PROBABILITIES) / Float32Array.BYTES_PER_ELEMENT;
  let roll = random();
  let chosen = actionCount - 1;
  for (let index = 0; index < actionCount; index += 1) {
    roll -= decoder.HEAPF32[probabilityBegin + index];
    if (roll <= 0) {
      chosen = index;
      break;
    }
  }
  return {
    ...actions[chosen],
    found: decoder._poker_query_found() === 1
  };
}
