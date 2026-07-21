<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { act, legalActions, nextHand, newHand, type Action, type Card, type Street, type Suit } from "./poker";
  import { loadNeuralPolicy, loadTabularPolicy, policyActions, policyMove, type Policy } from "./policy";
  import { FACINGS, STAT_ACTIONS, STREETS, bbPer100, emptyStats, loadStats, rate, recordHand, saveStats, stdDevPer100, type PokerStats } from "./stats";

  let game = newHand();
  let stats: PokerStats = emptyStats();
  let selectedStreet: Street = "preflop";
  let botTimer: ReturnType<typeof setTimeout>;
  let selectedModel: "deep" | "tabular" = "deep";
  let deepPolicy: Policy | null = null;
  let tabularPolicy: Policy | null = null;
  let policyReady = false;
  const debug = new URLSearchParams(location.search).get("debug") === "1";
  $: policy = selectedModel === "deep" ? deepPolicy : tabularPolicy;
  $: legal = legalActions(game);
  $: raiseOptions = game.winner ? [] : policyActions(game).filter((option) => option.action === "raise");
  $: busted = game.stacks.some((stack) => stack === 0);
  $: {
    clearTimeout(botTimer);
    if (!game.winner && game.toAct === 1 && policyReady) {
      botTimer = setTimeout(() => {
        const move = policyMove(policy, game);
        play(move.action, move.raiseTo);
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
  const lastActionWasCheck = (player: number) => {
    for (let index = game.actions.length - 1; index >= 0; index -= 1) {
      const action = game.actions[index];
      if (action.player === player && action.street === game.street) return action.action === "check";
    }
    return false;
  };
  onMount(() => {
    stats = loadStats();
    void Promise.allSettled([loadTabularPolicy(), loadNeuralPolicy()])
      .then(([tabular, deep]) => {
        if (tabular.status === "fulfilled") tabularPolicy = tabular.value;
        else console.error("Could not load tabular policy.", tabular.reason);
        if (deep.status === "fulfilled") deepPolicy = deep.value;
        else console.error("Could not load Deep CFR policy.", deep.reason);
        if (!deepPolicy && tabularPolicy) selectedModel = "tabular";
        policyReady = true;
      });
  });
  onDestroy(() => clearTimeout(botTimer));
</script>

<main>
  <section class="table">
    {#each [0, 1] as seat}
      <article class:active={!game.winner && game.toAct === seat}>
        <header class="player-info">
          <h2>
            {seat === 0 ? "You" : "Computer"}
            {#if game.dealer === seat}<span class="dealer" title="Dealer button">D</span>{/if}
          </h2>
          <p><span>Stack</span><strong>${game.stacks[seat]}</strong></p>
        </header>
        {#if seat === 1}
          <div class="model-selector" role="group" aria-label="Computer model">
            <button class:active-model={selectedModel === "deep"} disabled={!deepPolicy} on:click={() => (selectedModel = "deep")}>Deep CFR</button>
            <button class:active-model={selectedModel === "tabular"} disabled={!tabularPolicy} on:click={() => (selectedModel = "tabular")}>Tabular CFR</button>
          </div>
        {/if}
        <div class="cards">
          {#each game.holes[seat] as card}
            {#if seat === 1 && !game.showdown && !debug}
              <img class="card" src="/cards/1B.svg" alt="Hidden card" />
            {:else}
              <img class="card" src={cardImage(card)} alt={`${card[0]} of ${suitName(card)}`} />
            {/if}
          {/each}
        </div>
        {#if game.bets[seat] > 0 || lastActionWasCheck(seat)}
          <p class="action-bubble" aria-label={`${seat === 0 ? "Your" : "Computer"} last action: ${game.bets[seat] > 0 ? `$${game.bets[seat]}` : "Check"}`}>
            <strong>{game.bets[seat] > 0 ? `$${game.bets[seat]}` : "Check"}</strong>
          </p>
        {/if}
      </article>
    {/each}

    <section class="board" aria-label="Board">
      <div class="table-totals">
        <p class="pot"><span>Pot</span><strong>${game.pot}</strong></p>
      </div>
      <div class="cards">
        {#each game.board as card}
          <img class="card" src={cardImage(card)} alt={`${card[0]} of ${suitName(card)}`} />
        {/each}
      </div>
    </section>

    <section class="controls" aria-label="Game controls">
      {#if game.winner}
        <p class="message">{game.message}</p>
        <button class="next-action" on:click={next}>{busted ? "Reset game" : "Next hand"}</button>
      {:else if game.toAct === 0}
        <div class="actions">
          {#if legal.canFold}<button class="fold-action" on:click={() => play("fold")}>Fold</button>{/if}
          {#if legal.canCheck}<button class="check-action" on:click={() => play("check")}>Check</button>{/if}
          {#if legal.canCall}<button class="wager-action" on:click={() => play("call")}>Call <strong>${legal.toCall}</strong></button>{/if}
          {#each raiseOptions as option}
            <button class="wager-action" on:click={() => play("raise", option.raiseTo)}>
              {#if option.allIn}
                All In
              {:else}
                {game.currentBet ? "Raise to" : "Bet"} <strong>${option.raiseTo}</strong>
              {/if}
            </button>
          {/each}
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
          <div><dt>Est. std dev</dt><dd>{estimate(stdDevPer100(stats), " BB/100")}</dd></div>
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
    background: #0b0f0d;
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
    font-size: 14px;
    letter-spacing: 0;
    text-transform: capitalize;
  }

  .table {
    position: relative;
    display: grid;
    grid-template-areas:
      "player-one"
      "board"
      "player-two"
      "controls";
    grid-template-rows: auto minmax(140px, 1fr) auto minmax(72px, auto);
    gap: 12px;
    min-height: calc(100dvh - 16px);
    padding: 16px;
    overflow: hidden;
    isolation: isolate;
    box-sizing: border-box;
    border-radius: 8px;
    background: #080c0a;
    box-shadow: inset 0 0 100px rgb(0 0 0 / 0.45);
  }

  .table::before {
    position: absolute;
    inset: 0;
    z-index: 0;
    content: "";
    pointer-events: none;
    background: url("/poker-table.png") center / 100% 100% no-repeat;
    filter: brightness(0.52) saturate(0.72);
  }

  .table > article,
  .board,
  .controls {
    z-index: 1;
  }

  article {
    position: relative;
    padding: 10px;
    border: 1px solid rgb(255 255 255 / 0.14);
    border-radius: 6px;
    background: rgb(10 14 12 / 0.88);
    box-shadow: 0 8px 24px rgb(0 0 0 / 0.28);
  }

  article.active {
    border-color: #e7c766;
    box-shadow: 0 0 0 2px #e7c766, 0 8px 28px rgb(0 0 0 / 0.42);
  }

  article {
    width: min(300px, calc(100% - 24px));
    justify-self: center;
    box-sizing: border-box;
  }

  .player-info {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 12px;
  }

  .player-info h2 {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #c9d0cd;
    white-space: nowrap;
  }

  .player-info p {
    display: flex;
    align-items: baseline;
    gap: 5px;
    white-space: nowrap;
  }

  .player-info p > span,
  .table-totals span {
    color: #9ca8a3;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 0;
    line-height: 1;
    text-transform: uppercase;
  }

  .player-info strong,
  .table-totals strong,
  .actions strong {
    font-variant-numeric: tabular-nums;
  }

  .player-info strong {
    color: #f6f8f7;
    font-size: 18px;
    line-height: 1;
  }

  .dealer {
    display: inline-grid;
    width: 22px;
    height: 22px;
    place-items: center;
    border-radius: 50%;
    color: #17201d;
    background: #e8ecea;
    font-size: 11px;
    font-weight: 850;
    line-height: 1;
  }

  .model-selector {
    display: flex;
    width: fit-content;
    margin: 8px auto 0;
    overflow: hidden;
    border: 1px solid rgb(255 255 255 / 0.2);
    border-radius: 6px;
  }

  .model-selector button {
    min-width: 84px;
    min-height: 30px;
    padding: 0 9px;
    border: 0;
    border-right: 1px solid rgb(255 255 255 / 0.16);
    border-radius: 0;
    color: #c9d0cd;
    background: transparent;
    font-size: 12px;
    font-weight: 650;
  }

  .model-selector button:last-child {
    border-right: 0;
  }

  .model-selector .active-model {
    color: #171b19;
    background: #e7c766;
  }

  article:first-of-type {
    grid-area: player-two;
  }

  article:nth-of-type(2) {
    grid-area: player-one;
  }

  .action-bubble {
    position: absolute;
    left: 50%;
    display: flex;
    min-width: 60px;
    padding: 8px 14px;
    align-items: center;
    justify-content: center;
    gap: 8px;
    border: 1px solid #fff;
    border-radius: 6px;
    box-sizing: border-box;
    color: #17201d;
    background: #f7f3e8;
    font-size: 16px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    text-align: center;
    transform: translateX(-50%);
    box-shadow: 0 3px 8px rgb(0 0 0 / 0.35);
  }

  article:first-of-type .action-bubble {
    top: -43px;
  }

  article:nth-of-type(2) .action-bubble {
    bottom: -43px;
  }

  .board {
    grid-area: board;
    place-self: center;
    width: min(680px, calc(100% - 24px));
    text-align: center;
  }

  .table-totals {
    display: flex;
    justify-content: center;
    gap: 8px;
  }

  .pot {
    display: inline-flex;
    align-items: baseline;
    gap: 10px;
    padding: 7px 18px;
    border: 1px solid rgb(255 255 255 / 0.14);
    border-radius: 6px;
    background: rgb(8 14 11 / 0.82);
    box-shadow: 0 6px 20px rgb(0 0 0 / 0.22);
  }

  .table-totals strong {
    color: #f7f8f7;
    font-size: 25px;
    line-height: 1;
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
    filter: drop-shadow(0 5px 5px rgb(0 0 0 / 0.34));
  }

  .message {
    width: fit-content;
    max-width: 100%;
    min-height: 20px;
    margin: 0 auto 10px;
    padding: 7px 12px;
    box-sizing: border-box;
    border: 1px solid rgb(255 255 255 / 0.12);
    border-radius: 6px;
    color: #dfe5e2;
    background: rgb(8 13 11 / 0.82);
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
    border: 1px solid #66726d;
    border-radius: 6px;
    color: #f2f5f3;
    background: #151b18;
    font: inherit;
    font-weight: 800;
    cursor: pointer;
  }

  button:hover {
    background: #232c28;
  }

  .actions button {
    min-width: 124px;
    min-height: 54px;
    padding: 7px 14px;
    line-height: 1.1;
  }

  .actions strong {
    display: block;
    margin-top: 3px;
    font-size: 17px;
  }

  .fold-action {
    border-color: #a84d47;
    color: #ff9088;
  }

  .check-action {
    border-color: #4d9b70;
    color: #8cddb0;
  }

  .wager-action {
    border-color: #b7974e;
    color: #f1cf73;
  }

  .next-action {
    border-color: #e7c766;
    color: #171b19;
    background: #e7c766;
  }

  .next-action:hover {
    background: #f4d878;
  }

  .stats {
    grid-area: player-one;
    z-index: 5;
    align-self: start;
    justify-self: start;
  }

  .stats summary {
    position: relative;
    z-index: 6;
    width: fit-content;
    padding: 8px 12px;
    cursor: pointer;
    border-radius: 6px;
    border: 1px solid rgb(255 255 255 / 0.12);
    background: rgb(8 13 11 / 0.84);
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
    top: 64px;
    right: 16px;
    bottom: 16px;
    left: 16px;
    z-index: 4;
    max-height: calc(100% - 88px);
    padding: 16px;
    overflow: auto;
    border: 1px solid rgb(255 255 255 / 0.24);
    border-radius: 6px;
    background: rgb(8 13 11 / 0.98);
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
    border: 0;
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
      grid-template-areas:
        "stats"
        "player-one"
        "board"
        "player-two"
        "controls";
      grid-template-rows: auto auto minmax(140px, 1fr) auto minmax(72px, auto);
      min-height: 780px;
      padding: 12px;
    }

    .stats {
      grid-area: stats;
    }

    .stats-content {
      top: 60px;
      right: 12px;
      bottom: 12px;
      left: 12px;
    }
  }

</style>
