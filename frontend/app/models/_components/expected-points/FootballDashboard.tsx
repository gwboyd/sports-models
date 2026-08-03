"use client";

import { useMemo, useState } from "react";
import { Input } from "@/app/components/Input";
import { TeamIdentity } from "@/app/components/TeamIdentity";
import { displayProbability, formatKickoff, formatUpdatedAt } from "@/app/lib/formatting";
import { getSlateTeams, getTeamIdentity, NFL_TEAM_MANIFEST, normalizedSearch, teamSearchText } from "@/app/lib/team-data";
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

function MarketDetails({ game, market }: { game: ExpectedPointsPick; market: "spread" | "total" }) {
  const probability = market === "spread" ? game.spread_win_prob : game.total_win_prob;
  const locked = market === "spread" ? Boolean(game.spread_lock) : Boolean(game.total_lock);
  return (
    <div className="space-y-2 text-sm">
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Pick</span><strong>{market === "spread" ? spreadPickLabel(game) : totalPickLabel(game)}</strong></div>
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Model</span><strong>{market === "spread" ? spreadModelLabel(game) : game.total_pred.toFixed(1)}</strong></div>
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Pick win probability</span><strong className="numbers-tabular">{displayProbability(probability)}</strong></div>
      <div className="flex justify-between gap-3"><span className="text-[var(--muted)]">Status</span><strong className={locked ? "text-[var(--warning)]" : "text-slate-600"}>{locked ? "Lock" : "Standard pick"}</strong></div>
    </div>
  );
}

function FavoriteGameCard({ game, league }: { game: ExpectedPointsPick; league: FootballLeague }) {
  return (
    <article className="rounded-2xl border border-blue-100 bg-white p-4 shadow-[0_4px_16px_rgba(49,94,251,0.06)]">
      <div className="flex items-start justify-between gap-3">
        <Matchup game={game} league={league} />
        <span className="shrink-0 text-xs font-medium text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
      </div>
      <p className="mt-3 border-y border-slate-100 py-2 text-sm text-[var(--muted)]">Model score · <span className="font-medium text-[var(--ink)]">{predictedScoreLabel(game)}</span></p>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <MarketDetails game={game} market="spread" />
        <MarketDetails game={game} market="total" />
      </div>
    </article>
  );
}

function LockCard({ lock, league }: { lock: LockPick; league: FootballLeague }) {
  const isSpread = lock.market === "spread";
  return (
    <article className="w-[82vw] max-w-[340px] shrink-0 snap-start rounded-2xl border border-amber-200 bg-white p-4 shadow-[0_4px_16px_rgba(154,103,0,0.08)] md:w-auto md:max-w-none">
      <div className="flex items-center justify-between gap-3">
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-[var(--warning)]">{isSpread ? "Spread lock" : "Total lock"}</span>
        <span className="numbers-tabular text-sm font-semibold text-[var(--warning)]">{displayProbability(lock.probability)}</span>
      </div>
      <div className="mt-4"><Matchup game={lock.game} league={league} compact /></div>
      <p className="mt-4 text-2xl font-bold tracking-tight text-[var(--ink)]">{isSpread ? spreadPickLabel(lock.game) : totalPickLabel(lock.game)}</p>
      <div className="mt-3 space-y-1 text-sm text-[var(--muted)]">
        <p>Model · <span className="font-medium text-[var(--ink)]">{isSpread ? spreadModelLabel(lock.game) : lock.game.total_pred.toFixed(1)}</span></p>
        <p>{formatKickoff(lock.game.date_time)}</p>
      </div>
    </article>
  );
}

function GameMobileCard({ game, league, highlighted }: { game: ExpectedPointsPick; league: FootballLeague; highlighted: boolean }) {
  return (
    <article data-game-id={gameDomId(game.game_id)} className={`scroll-mt-36 rounded-2xl border bg-white transition-all ${highlighted ? "border-blue-400 ring-4 ring-blue-100" : "border-[var(--border)]"}`}>
      <details className="group">
        <summary className="cursor-pointer list-none p-4 [&::-webkit-details-marker]:hidden">
          <div className="flex items-start justify-between gap-3">
            <Matchup game={game} league={league} compact />
            <span className="shrink-0 text-xs font-medium text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="rounded-xl bg-slate-50 px-3 py-2.5"><span className="block text-[10px] font-bold uppercase tracking-wider text-[var(--muted)]">Spread pick</span><strong className="mt-1 block text-sm">{spreadPickLabel(game)}</strong></div>
            <div className="rounded-xl bg-slate-50 px-3 py-2.5"><span className="block text-[10px] font-bold uppercase tracking-wider text-[var(--muted)]">Total pick</span><strong className="mt-1 block text-sm">{totalPickLabel(game)}</strong></div>
          </div>
          <span className="mt-3 block text-center text-xs font-semibold text-[var(--accent)] group-open:hidden">View model details</span>
          <span className="mt-3 hidden text-center text-xs font-semibold text-[var(--accent)] group-open:block">Hide model details</span>
        </summary>
        <div className="border-t border-[var(--border)] px-4 pb-4 pt-3">
          <p className="mb-4 text-sm text-[var(--muted)]">Predicted score · <span className="font-medium text-[var(--ink)]">{predictedScoreLabel(game)}</span></p>
          <div className="grid gap-5 sm:grid-cols-2"><MarketDetails game={game} market="spread" /><MarketDetails game={game} market="total" /></div>
        </div>
      </details>
    </article>
  );
}

function DesktopMarketCell({ game, market }: { game: ExpectedPointsPick; market: "spread" | "total" }) {
  const label = market === "spread" ? spreadPickLabel(game) : totalPickLabel(game);
  const locked = market === "spread" ? Boolean(game.spread_lock) : Boolean(game.total_lock);
  return (
    <div tabIndex={0} className="group relative inline-flex min-h-11 items-center gap-2 rounded-lg px-2 focus:bg-blue-50">
      <span className="font-semibold text-[var(--ink)]">{label}</span>
      {locked ? <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-bold uppercase text-[var(--warning)]">Lock</span> : null}
      <div className="pointer-events-none absolute left-1/2 top-[calc(100%+8px)] z-30 hidden w-64 -translate-x-1/2 rounded-xl border border-slate-200 bg-white p-4 text-left shadow-xl group-hover:block group-focus:block">
        <MarketDetails game={game} market={market} />
      </div>
    </div>
  );
}

function GameDesktopTable({ games, league, highlightedId }: { games: ExpectedPointsPick[]; league: FootballLeague; highlightedId: string | null }) {
  return (
    <div className="hidden overflow-visible rounded-2xl border border-[var(--border)] bg-white md:block">
      <table className="w-full table-fixed text-left">
        <thead><tr className="border-b border-[var(--border)] bg-slate-50 text-xs uppercase tracking-wider text-[var(--muted)]"><th className="w-[38%] px-4 py-3">Matchup</th><th className="w-[20%] px-4 py-3">Kickoff</th><th className="w-[21%] px-4 py-3">Spread pick</th><th className="w-[21%] px-4 py-3">Total pick</th></tr></thead>
        <tbody>
          {games.map((game) => (
            <tr data-game-id={gameDomId(game.game_id)} key={game.game_id} className={`scroll-mt-36 border-b border-slate-100 last:border-0 ${highlightedId === game.game_id ? "bg-blue-50" : "hover:bg-slate-50/70"}`}>
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
  const { favoriteIds, ready, update } = useFavoriteTeams(league);

  const slateTeams = useMemo(() => getSlateTeams(games, league), [games, league]);
  const managerTeams = useMemo(() => {
    const teams = new Map((league === "nfl" ? NFL_TEAM_MANIFEST : []).map((team) => [team.id, team]));
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
    <main className="mx-auto w-full max-w-7xl px-4 py-6 pb-20 sm:px-6 sm:py-8 lg:px-8">
      <header className="flex flex-col gap-5 border-b border-[var(--border)] pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-[var(--accent)]">{games[0].season} · Week {games[0].week}</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">{leagueName} predictions</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">Every game includes the model&apos;s preferred spread and total. Locks are the highest-conviction opportunities that pass additional model checks.</p>
          {latestWrite ? <p className="mt-2 text-xs font-medium text-slate-500">Model updated {formatUpdatedAt(latestWrite)}</p> : null}
        </div>
        <div className="grid grid-cols-2 gap-2 sm:flex">
          <button type="button" onClick={() => setSearchOpen(true)} className="min-h-11 rounded-xl border border-[var(--border)] bg-white px-4 text-sm font-semibold text-[var(--ink)] shadow-sm hover:bg-slate-50">Search games</button>
          <button type="button" onClick={() => setFavoritesOpen(true)} className="min-h-11 rounded-xl bg-[var(--accent)] px-4 text-sm font-semibold text-white shadow-sm hover:bg-blue-700">Manage teams</button>
        </div>
      </header>

      <div className="mt-8 space-y-10">
        <section className="space-y-4" aria-labelledby="favorites-title">
          <SectionHeading title="Favorites" description="Your teams, saved on this device." action={<button type="button" onClick={() => setFavoritesOpen(true)} className="min-h-11 text-sm font-semibold text-[var(--accent)]">Edit teams</button>} />
          {!ready ? <div className="h-28 animate-pulse rounded-2xl bg-slate-200" /> : favoriteIds.length === 0 ? (
            <button type="button" onClick={() => setFavoritesOpen(true)} className="flex min-h-24 w-full items-center justify-between gap-4 rounded-2xl border border-dashed border-blue-300 bg-blue-50/50 p-4 text-left hover:bg-blue-50">
              <span><strong className="block text-[var(--ink)]">Follow your teams</strong><span className="mt-1 block text-sm text-[var(--muted)]">Choose favorites for an instant full-game view each week.</span></span><span className="shrink-0 text-sm font-semibold text-[var(--accent)]">Choose teams</span>
            </button>
          ) : favoriteGames.length === 0 ? <p className="rounded-2xl border border-[var(--border)] bg-white p-5 text-sm text-[var(--muted)]">None of your favorite teams has a game in the current slate.</p> : (
            <div className="grid gap-3 lg:grid-cols-2">{favoriteGames.map((game) => <FavoriteGameCard key={game.game_id} game={game} league={league} />)}</div>
          )}
        </section>

        <section className="space-y-4" aria-labelledby="locks-title">
          <SectionHeading title="Locks" description="Model-qualified spread and total picks, ordered by win probability." />
          {locks.length === 0 ? <p className="rounded-2xl border border-[var(--border)] bg-white p-5 text-sm text-[var(--muted)]">No locks are available for this slate.</p> : (
            <div className="hide-scrollbar -mx-4 flex snap-x snap-mandatory gap-3 overflow-x-auto px-4 pb-2 md:mx-0 md:grid md:grid-cols-2 md:overflow-visible md:px-0 lg:grid-cols-3">{locks.map((lock) => <LockCard key={lock.id} lock={lock} league={league} />)}</div>
          )}
        </section>

        <section className="space-y-5" aria-labelledby="games-title">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <SectionHeading title="All games" description="Tap a game on mobile or focus a pick on desktop to see the model forecast." />
            {league === "cfb" ? (
              <label className="text-sm font-medium text-[var(--muted)]">Conference
                <select value={conference} onChange={(event) => setConference(event.target.value)} className="ml-2 min-h-11 rounded-xl border border-[var(--border)] bg-white px-3 text-[var(--ink)]">
                  <option value="">All conferences</option>
                  {conferences.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
            ) : null}
          </div>
          {dateGroups.length === 0 ? <p className="rounded-2xl border border-[var(--border)] bg-white p-5 text-sm text-[var(--muted)]">No games match this conference.</p> : dateGroups.map((group) => (
            <div key={group.date} className="space-y-3">
              <h3 className="text-sm font-semibold text-[var(--muted)]">{group.date}</h3>
              <div className="space-y-3 md:hidden">{group.games.map((game) => <GameMobileCard key={game.game_id} game={game} league={league} highlighted={highlightedId === game.game_id} />)}</div>
              <GameDesktopTable games={group.games} league={league} highlightedId={highlightedId} />
            </div>
          ))}
        </section>
      </div>

      <aside className="mt-12 rounded-2xl bg-slate-100 p-4 text-xs leading-5 text-[var(--muted)]">
        Win probability refers to the selected spread or total pick hitting, not the team winning outright. Model outputs are informational and are not guarantees.
      </aside>

      {searchOpen ? (
        <OverlayPanel title={`Search ${leagueName} games`} description="Search by team, abbreviation, alias, or conference." onClose={() => setSearchOpen(false)}>
          <Input autoFocus type="search" placeholder="Try 49ers, SF, SEC…" value={searchQuery} onChange={(value) => setSearchQuery(String(value))} />
          <div className="mt-4 space-y-2">
            {searchResults.length === 0 ? <p className="py-8 text-center text-sm text-[var(--muted)]">No games match that search.</p> : searchResults.map((game) => (
              <button key={game.game_id} type="button" onClick={() => selectSearchResult(game)} className="flex min-h-16 w-full items-center justify-between gap-4 rounded-xl border border-[var(--border)] px-3 py-2 text-left hover:border-blue-200 hover:bg-blue-50">
                <Matchup game={game} league={league} compact /><span className="shrink-0 text-xs text-[var(--muted)]">{formatKickoff(game.date_time)}</span>
              </button>
            ))}
          </div>
        </OverlayPanel>
      ) : null}

      {favoritesOpen ? (
        <OverlayPanel title="Manage favorite teams" description="Favorites are stored only on this device." onClose={() => setFavoritesOpen(false)}>
          <Input autoFocus type="search" placeholder="Search teams" value={teamQuery} onChange={(value) => setTeamQuery(String(value))} />
          <div className="mt-4 space-y-2">
            {visibleManagerTeams.map((team) => {
              const checked = favoriteIds.includes(team.id);
              return (
                <label key={team.id} className="flex min-h-14 cursor-pointer items-center justify-between gap-4 rounded-xl border border-[var(--border)] px-3 py-2 hover:bg-slate-50">
                  <TeamIdentity team={team} compact />
                  <input aria-label={`Favorite ${team.displayName}`} type="checkbox" checked={checked} onChange={(event) => update(team.id, event.target.checked)} className="h-5 w-5 rounded border-slate-300 text-[var(--accent)]" />
                </label>
              );
            })}
          </div>
        </OverlayPanel>
      ) : null}
    </main>
  );
}
