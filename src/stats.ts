import type { Game, LoggedAction, Street } from "./poker";

export const FACINGS = ["first-in", "blind", "check", "call", "bet", "raise"] as const;
export const STAT_ACTIONS = ["fold", "check", "call", "bet", "raise"] as const;
export const STREETS: Street[] = ["preflop", "flop", "turn", "river"];

export type Facing = typeof FACINGS[number];
export type StatAction = typeof STAT_ACTIONS[number];

export interface DecisionStats extends Record<StatAction, number> {
  opportunities: number;
}

export interface PokerStats {
  version: 1;
  hands: number;
  netBb: number;
  squaredBb: number;
  vpipHands: number;
  pfrHands: number;
  threeBets: number;
  threeBetOpportunities: number;
  streets: Record<Street, Record<Facing, DecisionStats>>;
}

const STORAGE_KEY = "heads-up-poker-stats-v1";
const BIG_BLIND = 2;

const emptyDecision = (): DecisionStats => ({ opportunities: 0, fold: 0, check: 0, call: 0, bet: 0, raise: 0 });
const emptyStreet = (): Record<Facing, DecisionStats> => Object.fromEntries(
  FACINGS.map((facing) => [facing, emptyDecision()])
) as Record<Facing, DecisionStats>;

export function emptyStats(): PokerStats {
  return {
    version: 1,
    hands: 0,
    netBb: 0,
    squaredBb: 0,
    vpipHands: 0,
    pfrHands: 0,
    threeBets: 0,
    threeBetOpportunities: 0,
    streets: Object.fromEntries(STREETS.map((street) => [street, emptyStreet()])) as PokerStats["streets"]
  };
}

function facing(previous: StatAction[], street: Street): Facing {
  if (previous.length === 0) return street === "preflop" ? "blind" : "first-in";
  const last = previous.at(-1);
  return last === "check" || last === "call" || last === "bet" || last === "raise" ? last : "first-in";
}

function statAction(action: LoggedAction, previous: StatAction[]): StatAction {
  if (action.action !== "raise") return action.action;
  return action.street !== "preflop" && !previous.some((value) => value === "bet" || value === "raise") ? "bet" : "raise";
}

export function recordHand(current: PokerStats, game: Game): PokerStats {
  if (!game.winner) throw new Error("Cannot record an unfinished hand");
  const next = structuredClone(current);
  const result = (game.stacks[0] - game.startingStacks[0]) / BIG_BLIND;
  next.hands += 1;
  next.netBb += result;
  next.squaredBb += result * result;

  const previous: Record<Street, StatAction[]> = { preflop: [], flop: [], turn: [], river: [] };
  let vpip = false;
  let pfr = false;
  for (const action of game.actions) {
    const prior = previous[action.street];
    const situation = facing(prior, action.street);
    const normalized = statAction(action, prior);
    if (action.player === 0) {
      const decision = next.streets[action.street][situation];
      decision.opportunities += 1;
      decision[normalized] += 1;
      if (action.street === "preflop") {
        vpip ||= normalized === "call" || normalized === "raise";
        pfr ||= normalized === "raise";
        const priorRaises = prior.filter((value) => value === "raise").length;
        if (situation === "raise" && priorRaises === 1) {
          next.threeBetOpportunities += 1;
          if (normalized === "raise") next.threeBets += 1;
        }
      }
    }
    prior.push(normalized);
  }
  if (vpip) next.vpipHands += 1;
  if (pfr) next.pfrHands += 1;
  return next;
}

export const rate = (value: number, total: number): number | null => total > 0 ? value / total : null;
export const bbPer100 = (stats: PokerStats): number | null => rate(stats.netBb * 100, stats.hands);
export const stdDevPer100 = (stats: PokerStats): number | null => stats.hands > 1
  ? Math.sqrt(100 * Math.max(0, (stats.squaredBb - stats.netBb * stats.netBb / stats.hands) / (stats.hands - 1)))
  : null;

export function loadStats(): PokerStats {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PokerStats | null;
    return value?.version === 1 && value.streets ? value : emptyStats();
  } catch {
    return emptyStats();
  }
}

export function saveStats(stats: PokerStats): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stats));
  } catch {
    // Statistics remain available for the current page session.
  }
}
