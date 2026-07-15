<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { act, botAction, legalActions, nextHand, newHand, type Action, type Card, type Street, type Suit } from "./poker";
  import { FACINGS, STAT_ACTIONS, STREETS, bbPer100, emptyStats, loadStats, rate, recordHand, saveStats, variance, type PokerStats } from "./stats";

  let game = newHand();
  let stats: PokerStats = emptyStats();
  let selectedStreet: Street = "preflop";
  let botTimer: ReturnType<typeof setTimeout>;
  $: legal = legalActions(game);
  $: raiseTo = legal.minRaiseTo;
  $: busted = game.stacks.some((stack) => stack === 0);
  $: {
    clearTimeout(botTimer);
    if (!game.winner && game.toAct === 1) {
      botTimer = setTimeout(() => {
        const action = botAction(game);
        play(action, action === "raise" ? legalActions(game).minRaiseTo ?? undefined : undefined);
      }, 500);
    }
  }

  const suits: Record<Suit, string> = { C: "clubs", D: "diamonds", H: "hearts", S: "spades" };
  const play = (action: Action, raiseTo?: number) => {
    const next = act(game, action, raiseTo);
    if (!game.winner && next.winner) {
      stats = recordHand(stats, next);
      saveStats(stats);
    }
    game = next;
  };
  const next = () => {
    const hand = busted ? newHand() : nextHand(game);
    game = hand;
    if (hand.winner) {
      stats = recordHand(stats, hand);
      saveStats(stats);
    }
  };
  const resetStats = () => {
    if (!confirm("Reset all stored poker statistics?")) return;
    stats = emptyStats();
    saveStats(stats);
  };
  const percent = (value: number, total: number) => {
    const result = rate(value, total);
    return result === null ? "—" : `${(result * 100).toFixed(1)}%`;
  };
  const estimate = (value: number | null, suffix = "") => value === null ? "—" : `${value.toFixed(2)}${suffix}`;
  const facingLabel = (facing: typeof FACINGS[number]) => ({
    "first-in": "First to act",
    blind: "Blind",
    check: "Check",
    call: "Call",
    bet: "Bet",
    raise: "Raise"
  })[facing];
  const suitName = (card: Card) => suits[card[1] as Suit];
  const cardImage = (card: Card) => `/cards/${card}.svg`;
  onMount(() => (stats = loadStats()));
  onDestroy(() => clearTimeout(botTimer));
</script>

<main>
  <section class="table">
    {#each [0, 1] as seat}
      <article class:active={!game.winner && game.toAct === seat}>
        <h2>{seat === 0 ? "You" : "Computer"}{game.dealer === seat ? " (button)" : ""}</h2>
        <p>${game.stacks[seat]} stack / ${game.bets[seat]} bet</p>
        <div class="cards">
          {#each game.holes[seat] as card}
            {#if seat === 1 && !game.showdown}
              <img class="card" src="/cards/1B.svg" alt="Hidden card" />
            {:else}
              <img class="card" src={cardImage(card)} alt={`${card[0]} of ${suitName(card)}`} />
            {/if}
          {/each}
        </div>
      </article>
    {/each}

    <section class="board" aria-label="Board">
      <h2>Pot ${game.pot}</h2>
      <div class="cards">
        {#each game.board as card}
          <img class="card" src={cardImage(card)} alt={`${card[0]} of ${suitName(card)}`} />
        {/each}
      </div>
    </section>

    <section class="controls" aria-label="Game controls">
      <p class="message">{game.message}</p>

      {#if game.winner}
        <button on:click={next}>{busted ? "Reset game" : "Next hand"}</button>
      {:else if game.toAct === 0}
        <div class="actions">
          {#if legal.canFold}<button on:click={() => play("fold")}>Fold</button>{/if}
          {#if legal.canCheck}<button on:click={() => play("check")}>Check</button>{/if}
          {#if legal.canCall}<button on:click={() => play("call")}>Call ${legal.toCall}</button>{/if}
          {#if raiseTo !== null}
            <button on:click={() => play("raise", raiseTo!)}>{game.currentBet ? `Raise to $${raiseTo}` : `Bet $${raiseTo}`}</button>
          {/if}
        </div>
      {:else}
        <p class="thinking">Computer is thinking...</p>
      {/if}
    </section>

    <details class="stats">
      <summary>Statistics <span>{stats.hands} hands</span></summary>
      <div class="stats-content">
        <div class="stats-header">
          <button class="reset-stats" on:click={resetStats}>Reset stats</button>
        </div>

        <dl class="summary-stats">
          <div><dt>Hands</dt><dd>{stats.hands}</dd></div>
          <div><dt>Net</dt><dd>{stats.netBb.toFixed(1)} BB</dd></div>
          <div><dt>Est. BB/100</dt><dd>{estimate(bbPer100(stats))}</dd></div>
          <div><dt>Est. variance</dt><dd>{estimate(variance(stats), " BB²/hand")}</dd></div>
          <div><dt>VPIP</dt><dd>{percent(stats.vpipHands, stats.hands)}</dd></div>
          <div><dt>PFR</dt><dd>{percent(stats.pfrHands, stats.hands)}</dd></div>
          <div><dt>3-bet</dt><dd>{percent(stats.threeBets, stats.threeBetOpportunities)}</dd></div>
        </dl>

        <div class="street-tabs" role="tablist" aria-label="Street statistics">
          {#each STREETS as street}
            <button class:active-tab={selectedStreet === street} role="tab" aria-selected={selectedStreet === street} on:click={() => (selectedStreet = street)}>
              {street}
            </button>
          {/each}
        </div>

        <div class="stats-table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Facing</th>
                <th scope="col">N</th>
                {#each STAT_ACTIONS as action}<th scope="col">{action}</th>{/each}
              </tr>
            </thead>
            <tbody>
              {#each FACINGS as facing}
                <tr>
                  <th scope="row">{facingLabel(facing)}</th>
                  <td>{stats.streets[selectedStreet][facing].opportunities}</td>
                  {#each STAT_ACTIONS as action}
                    <td>{percent(stats.streets[selectedStreet][facing][action], stats.streets[selectedStreet][facing].opportunities)}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  </section>

</main>

<style>
  :global(body) {
    margin: 0;
    min-width: 320px;
    color: #f7f3e8;
    background: #122018;
    font-family: system-ui, sans-serif;
  }

  main {
    width: min(1400px, calc(100vw - 16px));
    margin: 0 auto;
    padding: 8px 0;
  }

  h2,
  p {
    margin: 0;
  }

  h2 {
    font-size: 16px;
    text-transform: capitalize;
  }

  .table {
    position: relative;
    display: grid;
    grid-template-areas:
      "player-one"
      "board"
      "player-two"
      "controls"
      "stats";
    grid-template-rows: auto minmax(140px, 1fr) auto minmax(72px, auto) auto;
    gap: 12px;
    min-height: calc(100dvh - 16px);
    padding: 16px;
    overflow: hidden;
    box-sizing: border-box;
    border-radius: 8px;
    background: #08100c url("/poker-table.png") center / 100% 100% no-repeat;
  }

  article {
    padding: 12px;
    border: 1px solid rgb(255 255 255 / 0.18);
    border-radius: 8px;
    background: rgb(0 0 0 / 0.16);
  }

  article.active {
    outline: 3px solid #f4ce68;
  }

  article {
    width: min(260px, calc(100% - 24px));
    justify-self: center;
  }

  article:first-of-type {
    grid-area: player-two;
  }

  article:nth-of-type(2) {
    grid-area: player-one;
  }

  .board {
    grid-area: board;
    place-self: center;
    width: min(680px, calc(100% - 24px));
    text-align: center;
  }

  .board .cards {
    min-height: 140px;
    align-items: center;
    gap: 0;
  }

  .cards {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    min-height: 88px;
    margin-top: 10px;
  }

  .card {
    display: block;
    width: 112px;
    height: 140px;
    border-radius: 6px;
  }

  .message {
    min-height: 24px;
    margin-bottom: 10px;
    text-align: center;
  }

  .thinking {
    min-height: 44px;
    text-align: center;
  }

  .controls {
    grid-area: controls;
    align-self: end;
    justify-self: center;
    width: min(720px, calc(100% - 24px));
    min-height: 72px;
    text-align: center;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
  }

  button {
    min-width: 112px;
    min-height: 44px;
    border: 0;
    border-radius: 6px;
    color: #111;
    background: #f4ce68;
    font: inherit;
    font-weight: 800;
    cursor: pointer;
  }

  button:hover {
    background: #ffe08a;
  }

  .stats {
    grid-area: stats;
    z-index: 5;
    justify-self: center;
  }

  .stats summary {
    position: relative;
    z-index: 6;
    width: fit-content;
    padding: 8px 12px;
    cursor: pointer;
    border-radius: 6px;
    background: rgb(8 16 12 / 0.72);
    font-size: 16px;
    font-weight: 750;
  }

  .stats summary span {
    margin-left: 8px;
    color: rgb(247 243 232 / 0.6);
    font-size: 13px;
    font-weight: 500;
  }

  .stats-content {
    position: absolute;
    right: 16px;
    bottom: 64px;
    left: 16px;
    z-index: 4;
    max-height: calc(100% - 88px);
    padding: 16px;
    overflow: auto;
    border: 1px solid rgb(255 255 255 / 0.24);
    border-radius: 6px;
    background: rgb(8 16 12 / 0.96);
  }

  .stats-header {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 16px;
  }

  .reset-stats {
    min-width: 0;
    min-height: 36px;
    padding: 0 12px;
    border: 1px solid rgb(255 255 255 / 0.3);
    color: #f7f3e8;
    background: transparent;
    font-weight: 600;
  }

  .reset-stats:hover {
    background: rgb(255 255 255 / 0.1);
  }

  .summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    margin: 20px 0;
    border-block: 1px solid rgb(255 255 255 / 0.14);
  }

  .summary-stats div {
    padding: 12px 8px;
  }

  .summary-stats dt {
    color: rgb(247 243 232 / 0.65);
    font-size: 12px;
  }

  .summary-stats dd {
    margin: 4px 0 0;
    font-size: 18px;
    font-weight: 750;
  }

  .street-tabs {
    display: flex;
    width: fit-content;
    max-width: 100%;
    margin-bottom: 12px;
    overflow-x: auto;
    border: 1px solid rgb(255 255 255 / 0.24);
    border-radius: 6px;
  }

  .street-tabs button {
    min-width: 88px;
    min-height: 36px;
    padding: 0 12px;
    border-right: 1px solid rgb(255 255 255 / 0.18);
    border-radius: 0;
    color: #f7f3e8;
    background: transparent;
    font-weight: 600;
    text-transform: capitalize;
  }

  .street-tabs button:last-child {
    border-right: 0;
  }

  .street-tabs .active-tab {
    color: #111;
    background: #f4ce68;
  }

  .stats-table-wrap {
    overflow-x: auto;
  }

  table {
    width: 100%;
    min-width: 680px;
    border-collapse: collapse;
    font-size: 14px;
  }

  th,
  td {
    padding: 10px 12px;
    border-bottom: 1px solid rgb(255 255 255 / 0.12);
    text-align: right;
    white-space: nowrap;
  }

  th:first-child {
    text-align: left;
  }

  thead th {
    color: rgb(247 243 232 / 0.65);
    font-size: 12px;
    text-transform: uppercase;
  }

  @media (max-width: 620px) {
    .table {
      min-height: 780px;
      padding: 12px;
    }

    .stats-content {
      right: 12px;
      bottom: 60px;
      left: 12px;
    }
  }
</style>
