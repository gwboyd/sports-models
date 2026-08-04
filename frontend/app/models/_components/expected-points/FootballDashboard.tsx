"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/app/components/Input";
import { TeamIdentity } from "@/app/components/TeamIdentity";
import { displayProbability, formatKickoff, formatUpdatedAt } from "@/app/lib/formatting";
import { CFB_TEAM_MANIFEST, getSlateTeams, getTeamIdentity, NFL_TEAM_MANIFEST, normalizedSearch, teamSearchText } from "@/app/lib/team-data";
import type { ExpectedPointsPick, FootballLeague } from "@/app/types/types";
import { OverlayPanel } from "./OverlayPanel";
import {
  filterGamesByConference,
  gameDomId,
  getConferences,
  getLocks,
  groupGamesByDate,
  predictedScoreLabel,
  searchGames,
  spreadModelLabel,
  spreadPickLabel,
  totalPickLabel,
  type LockPick,
} from "./view-model";
import { useFavoriteTeams } from "./use-favorites";

function SectionHeading({ title, description, action }: { title: string; description?: string; action?: React.ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-[var(--ink)] sm:text-2xl">{title}</h2>
        {description ? <p className="mt-1 text-sm text-[var(--muted)]">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

function Matchup({ game, league, compact = false }: { game: ExpectedPointsPick; league: FootballLeague; compact?: boolean }) {
  const away = getTeamIdentity(game.away_team, league);
  const home = getTeamIdentity(game.home_team, league);
  return (
    <div className="flex min-w-0 items-center gap-2">
      <TeamIdentity team={away} compact={compact} showName={false} />
      <span className="truncate font-semibold text-[var(--ink)]">{away.abbreviation}</span>
      <span className="text-xs text-slate-400">at</span>
      <TeamIdentity team={home} compact={compact} showName={false} />
      <span className="truncate font-semibold text-[var(--ink)]">{home.abbreviation}</span>
    </div>
  );
}

function MarketDetails({ game, market, showStatus = true }: { game: ExpectedPointsPick; market: "spread" | "total"; showStatus?: boolean }) {
  const probability = market === "spread" ? game.spread_win_prob : game.total_win_prob;
  const locked = market === "spread" ? Boolean(game.spread_lock) : Boolean(game.total_lock);
  return (
    <div className="space-y-2 text-sm">
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Pick</span><strong>{market === "spread" ? spreadPickLabel(game) : totalPickLabel(game)}</strong></div>
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Model</span><strong>{market === "spread" ? spreadModelLabel(game) : game.total_pred.toFixed(1)}</strong></div>
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Pick win probability</span><strong className="numbers-tabular">{displayProbability(probability)}</strong></div>
      {showStatus ? <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Status</span><strong className={locked ? "text-[var(--warning)]" : "text-slate-600"}>{locked ? "Lock" : "Standard pick"}</strong></div> : null}
    </div>
  );
}

function LockBadge({ market }: { market: "spread" | "total" }) {
  return (
    <span className="rounded-md bg-[var(--lock-soft)] px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] text-[var(--lock)]">
      {market === "spread" ? "Spread lock" : "Total lock"}
    </span>
  );
}

function FavoriteGameCard({ game, league }: { game: ExpectedPointsPick; league: FootballLeague }) {
  const spreadLocked = Boolean(game.spread_lock);
  const totalLocked = Boolean(game.total_lock);
  const locked = spreadLocked || totalLocked;

  return (
    <article className={`rounded-lg border bg-white p-3.5 ${locked ? "border-[var(--lock-border)]" : "border-[var(--border)]"}`}>
      <div className="flex items-start justify-between gap-3">
        <Matchup game={game} league={league} />
        <span className="shrink-0 text-xs font-medium text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
      </div>
      {locked ? (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {spreadLocked ? <LockBadge market="spread" /> : null}
          {totalLocked ? <LockBadge market="total" /> : null}
        </div>
      ) : null}
      <p className="mt-2.5 border-y border-slate-100 py-2 text-sm text-[var(--muted)]">Model score · <span className="font-medium text-[var(--ink)]">{predictedScoreLabel(game)}</span></p>
      <div className="mt-2.5 grid gap-3 sm:grid-cols-2">
        <MarketDetails game={game} market="spread" showStatus={false} />
        <MarketDetails game={game} market="total" showStatus={false} />
      </div>
    </article>
  );
}

function LockCard({ lock, league }: { lock: LockPick; league: FootballLeague }) {
  const isSpread = lock.market === "spread";
  return (
    <article className="w-[76vw] max-w-[300px] shrink-0 snap-start rounded-lg border border-[var(--lock-border)] bg-white p-3 md:w-auto md:max-w-none">
      <div className="flex items-center justify-between gap-3">
        <LockBadge market={lock.market} />
        <span className="numbers-tabular text-sm font-semibold text-[var(--lock)]">{displayProbability(lock.probability)}</span>
      </div>
      <div className="mt-3"><Matchup game={lock.game} league={league} compact /></div>
      <p className="mt-3 text-xl font-bold tracking-tight text-[var(--ink)]">{isSpread ? spreadPickLabel(lock.game) : totalPickLabel(lock.game)}</p>
      <div className="mt-2 space-y-0.5 text-xs leading-5 text-[var(--muted)]">
        <p>Model · <span className="font-medium text-[var(--ink)]">{isSpread ? spreadModelLabel(lock.game) : lock.game.total_pred.toFixed(1)}</span></p>
        <p>{formatKickoff(lock.game.date_time)}</p>
      </div>
    </article>
  );
}

function GameMobileCard({ game, league, highlighted }: { game: ExpectedPointsPick; league: FootballLeague; highlighted: boolean }) {
  return (
    <article data-game-id={gameDomId(game.game_id)} className={`scroll-mt-36 rounded-lg border bg-white transition-all ${highlighted ? "border-[var(--accent)] ring-2 ring-[var(--accent-soft)]" : "border-[var(--border)]"}`}>
      <details className="group">
        <summary className="cursor-pointer list-none p-3 [&::-webkit-details-marker]:hidden">
          <div className="flex items-start justify-between gap-3">
            <Matchup game={game} league={league} compact />
            <span className="shrink-0 text-xs font-medium text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-2"><span className="block text-[10px] font-bold uppercase tracking-wider text-[var(--muted)]">Spread pick</span><strong className="mt-0.5 block text-sm">{spreadPickLabel(game)}</strong></div>
            <div className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-2"><span className="block text-[10px] font-bold uppercase tracking-wider text-[var(--muted)]">Total pick</span><strong className="mt-0.5 block text-sm">{totalPickLabel(game)}</strong></div>
          </div>
          <span className="mt-3 block text-center text-xs font-semibold text-[var(--accent)] group-open:hidden">View model details</span>
          <span className="mt-3 hidden text-center text-xs font-semibold text-[var(--accent)] group-open:block">Hide model details</span>
        </summary>
        <div className="border-t border-[var(--border)] px-3 pb-3 pt-2.5">
          <p className="mb-4 text-sm text-[var(--muted)]">Predicted score · <span className="font-medium text-[var(--ink)]">{predictedScoreLabel(game)}</span></p>
          <div className="grid gap-4 sm:grid-cols-2"><MarketDetails game={game} market="spread" /><MarketDetails game={game} market="total" /></div>
        </div>
      </details>
    </article>
  );
}

function DesktopMarketCell({ game, market }: { game: ExpectedPointsPick; market: "spread" | "total" }) {
  const label = market === "spread" ? spreadPickLabel(game) : totalPickLabel(game);
  const locked = market === "spread" ? Boolean(game.spread_lock) : Boolean(game.total_lock);
  return (
    <div tabIndex={0} className="group relative inline-flex min-h-11 items-center gap-2 rounded-md px-2 focus:bg-[var(--accent-soft)]">
      <span className="font-semibold text-[var(--ink)]">{label}</span>
      {locked ? <span className="rounded-md bg-[var(--lock-soft)] px-1.5 py-0.5 text-[10px] font-bold uppercase text-[var(--lock)]">Lock</span> : null}
      <div className="pointer-events-none absolute left-1/2 top-[calc(100%+8px)] z-30 hidden w-64 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3.5 text-left shadow-lg group-hover:block group-focus:block">
        <MarketDetails game={game} market={market} />
      </div>
    </div>
  );
}

function GameDesktopTable({ games, league, highlightedId }: { games: ExpectedPointsPick[]; league: FootballLeague; highlightedId: string | null }) {
  return (
    <div className="hidden overflow-visible rounded-lg border border-[var(--border)] bg-white md:block">
      <table className="w-full table-fixed text-left">
        <thead><tr className="border-b border-[var(--border)] bg-slate-50 text-xs uppercase tracking-wider text-[var(--muted)]"><th className="w-[38%] px-4 py-3">Matchup</th><th className="w-[20%] px-4 py-3">Kickoff</th><th className="w-[21%] px-4 py-3">Spread pick</th><th className="w-[21%] px-4 py-3">Total pick</th></tr></thead>
        <tbody>
          {games.map((game) => (
            <tr data-game-id={gameDomId(game.game_id)} key={game.game_id} className={`scroll-mt-36 border-b border-slate-100 last:border-0 ${highlightedId === game.game_id ? "bg-[var(--accent-soft)]" : "hover:bg-slate-50/70"}`}>
              <td className="px-4 py-3"><Matchup game={game} league={league} compact /></td>
              <td className="px-4 py-3 text-sm text-[var(--muted)]">{formatKickoff(game.date_time)}</td>
              <td className="px-2 py-2"><DesktopMarketCell game={game} market="spread" /></td>
              <td className="px-2 py-2"><DesktopMarketCell game={game} market="total" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function FootballDashboard({ league, games }: { league: FootballLeague; games: ExpectedPointsPick[] }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [favoritesOpen, setFavoritesOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [teamQuery, setTeamQuery] = useState("");
  const [conference, setConference] = useState("");
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const teamResultsRef = useRef<HTMLDivElement>(null);
  const { favoriteIds, ready, update } = useFavoriteTeams(league);

  const slateTeams = useMemo(() => getSlateTeams(games, league), [games, league]);
  const managerTeams = useMemo(() => {
    const manifest = league === "nfl" ? NFL_TEAM_MANIFEST : CFB_TEAM_MANIFEST;
    const teams = new Map(manifest.map((team) => [team.id, team]));
    for (const team of slateTeams) teams.set(team.id, team);
    return [...teams.values()].sort((a, b) => a.displayName.localeCompare(b.displayName));
  }, [league, slateTeams]);
  const favoriteGames = useMemo(() => games.filter((game) => {
    const homeId = getTeamIdentity(game.home_team, league).id;
    const awayId = getTeamIdentity(game.away_team, league).id;
    return favoriteIds.includes(homeId) || favoriteIds.includes(awayId);
  }), [favoriteIds, games, league]);
  const locks = useMemo(() => getLocks(games), [games]);
  const conferences = useMemo(() => getConferences(games), [games]);
  const visibleGames = useMemo(() => filterGamesByConference(games, conference), [conference, games]);
  const dateGroups = useMemo(() => groupGamesByDate(visibleGames), [visibleGames]);
  const searchResults = useMemo(() => searchGames(games, league, searchQuery), [games, league, searchQuery]);
  const visibleManagerTeams = useMemo(() => {
    const needle = normalizedSearch(teamQuery);
    return needle ? managerTeams.filter((team) => teamSearchText(team).includes(needle)) : managerTeams;
  }, [managerTeams, teamQuery]);
  const latestWrite = games.map((game) => game.write_time).sort().at(-1) ?? "";
  const leagueName = league === "nfl" ? "NFL" : "College Football";

  useEffect(() => {
    if (teamResultsRef.current) teamResultsRef.current.scrollTop = 0;
  }, [teamQuery]);

  function selectSearchResult(game: ExpectedPointsPick) {
    setSearchOpen(false);
    setSearchQuery("");
    setConference("");
    setHighlightedId(game.game_id);
    window.setTimeout(() => {
      const candidates = [...document.querySelectorAll<HTMLElement>(`[data-game-id="${gameDomId(game.game_id)}"]`)];
      candidates.find((element) => element.offsetParent !== null)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
    window.setTimeout(() => setHighlightedId(null), 2400);
  }

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-5 pb-16 sm:px-6 sm:py-7 lg:px-8">
      <header className="flex flex-col gap-4 border-b border-[var(--border)] pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-[var(--accent)]">{games[0].season} · Week {games[0].week}</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">{leagueName} predictions</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">Every game includes the model&apos;s preferred spread and total. Locks are the highest-conviction opportunities that pass additional model checks.</p>
          {latestWrite ? <p className="mt-2 text-xs font-medium text-slate-500">Model updated {formatUpdatedAt(latestWrite)}</p> : null}
        </div>
        <button type="button" onClick={() => setSearchOpen(true)} className="flex min-h-11 w-full items-center gap-2.5 rounded-lg border border-[var(--border)] bg-white px-3.5 text-left text-sm text-[var(--muted)] hover:border-[var(--lock-border)] hover:bg-slate-50 sm:w-80">
          <svg aria-hidden="true" viewBox="0 0 20 20" className="h-4 w-4 shrink-0 fill-none stroke-current" strokeWidth="1.8"><circle cx="8.5" cy="8.5" r="5.5" /><path d="m12.5 12.5 4 4" /></svg>
          <span>Search games</span>
        </button>
      </header>

      <div className="mt-7 space-y-8">
        <section className="space-y-3" aria-labelledby="favorites-title">
          <SectionHeading title="Favorites" description="Current-week matchups for your selected teams." action={<button type="button" onClick={() => setFavoritesOpen(true)} className="min-h-11 whitespace-nowrap text-sm font-semibold text-[var(--accent)]">Edit teams</button>} />
          {!ready ? <div className="h-20 animate-pulse rounded-lg bg-slate-200" /> : favoriteIds.length === 0 ? (
            <button type="button" onClick={() => setFavoritesOpen(true)} className="flex min-h-20 w-full items-center justify-between gap-3 rounded-lg border border-dashed border-[var(--lock-border)] bg-white p-3.5 text-left hover:bg-[var(--accent-soft)]">
              <span><strong className="block text-[var(--ink)]">No favorite teams selected</strong><span className="mt-1 block text-sm text-[var(--muted)]">Select teams to keep their weekly matchups here.</span></span><span className="shrink-0 text-sm font-semibold text-[var(--accent)]">Select teams</span>
            </button>
          ) : favoriteGames.length === 0 ? <p className="rounded-lg border border-[var(--border)] bg-white p-3.5 text-sm text-[var(--muted)]">None of your favorite teams has a game in the current slate.</p> : (
            <div className="grid gap-3 lg:grid-cols-2">{favoriteGames.map((game) => <FavoriteGameCard key={game.game_id} game={game} league={league} />)}</div>
          )}
        </section>

        <section className="space-y-3" aria-labelledby="locks-title">
          <SectionHeading title="Locks" description="Model-qualified spread and total picks, ordered by win probability." />
          {locks.length === 0 ? <p className="rounded-lg border border-[var(--border)] bg-white p-3.5 text-sm text-[var(--muted)]">No locks are available for this slate.</p> : (
            <div className="hide-scrollbar -mr-4 flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2 pr-4 md:mr-0 md:grid md:grid-cols-2 md:overflow-visible md:pr-0 xl:grid-cols-4">{locks.map((lock) => <LockCard key={lock.id} lock={lock} league={league} />)}</div>
          )}
        </section>

        <section className="space-y-4" aria-labelledby="games-title">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <SectionHeading title="All games" description="Tap a game on mobile or focus a pick on desktop to see the model forecast." />
            {league === "cfb" ? (
              <label className="text-sm font-medium text-[var(--muted)]">Conference
                <select value={conference} onChange={(event) => setConference(event.target.value)} className="ml-2 min-h-11 rounded-lg border border-[var(--border)] bg-white px-3 text-[var(--ink)]">
                  <option value="">All conferences</option>
                  {conferences.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
            ) : null}
          </div>
          {dateGroups.length === 0 ? <p className="rounded-lg border border-[var(--border)] bg-white p-3.5 text-sm text-[var(--muted)]">No games match this conference.</p> : dateGroups.map((group) => (
            <div key={group.date} className="space-y-3">
              <h3 className="text-sm font-semibold text-[var(--muted)]">{group.date}</h3>
              <div className="space-y-3 md:hidden">{group.games.map((game) => <GameMobileCard key={game.game_id} game={game} league={league} highlighted={highlightedId === game.game_id} />)}</div>
              <GameDesktopTable games={group.games} league={league} highlightedId={highlightedId} />
            </div>
          ))}
        </section>
      </div>

      <aside className="mt-10 border-l-2 border-slate-300 py-1 pl-3 text-xs leading-5 text-[var(--muted)]">
        Win probability refers to the selected spread or total pick hitting, not the team winning outright. Model outputs are informational and are not guarantees.
      </aside>

      {searchOpen ? (
        <OverlayPanel title={`Search ${leagueName} games`} description="Search by team, abbreviation, alias, or conference." onClose={() => setSearchOpen(false)}>
          <div className="flex h-full min-h-0 flex-col">
            <Input autoFocus type="search" placeholder="Try 49ers, SF, SEC…" value={searchQuery} onChange={(value) => setSearchQuery(String(value))} />
            <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain pr-1">
              {searchResults.length === 0 ? <p className="py-8 text-center text-sm text-[var(--muted)]">No games match that search.</p> : searchResults.map((game) => (
                <button key={game.game_id} type="button" onClick={() => selectSearchResult(game)} className="flex min-h-14 w-full items-center justify-between gap-3 rounded-lg border border-[var(--border)] px-3 py-2 text-left hover:border-[var(--lock-border)] hover:bg-[var(--accent-soft)]">
                  <Matchup game={game} league={league} compact /><span className="shrink-0 text-xs text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
                </button>
              ))}
            </div>
          </div>
        </OverlayPanel>
      ) : null}

      {favoritesOpen ? (
        <OverlayPanel title="Manage favorite teams" description="Favorites are stored only on this device." onClose={() => { setFavoritesOpen(false); setTeamQuery(""); }}>
          <div className="flex h-full min-h-0 flex-col">
            <Input autoFocus type="search" placeholder="Search teams" value={teamQuery} onChange={(value) => setTeamQuery(String(value))} />
            <div ref={teamResultsRef} className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain pr-1">
              {visibleManagerTeams.length === 0 ? <p className="py-6 text-center text-sm text-[var(--muted)]">No teams match that search.</p> : visibleManagerTeams.map((team) => {
                const checked = favoriteIds.includes(team.id);
                return (
                  <label key={team.id} className="flex min-h-14 cursor-pointer items-center justify-between gap-3 rounded-lg border border-[var(--border)] px-3 py-2 hover:bg-slate-50">
                    <TeamIdentity team={team} compact />
                    <input aria-label={`Favorite ${team.displayName}`} type="checkbox" checked={checked} onChange={(event) => update(team.id, event.target.checked)} className="h-5 w-5 rounded border-slate-300 text-[var(--accent)]" />
                  </label>
                );
              })}
            </div>
          </div>
        </OverlayPanel>
      ) : null}
    </main>
  );
}
