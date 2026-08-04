"use client";

import { useMemo, useState } from "react";
import { availableResultSeasons, filterResultsBySeason, summarizeResults, weeklyResults, type RecordSummary } from "@/app/lib/results-data";
import type { FootballLeague, GameResult } from "@/app/types/types";

function recordLabel(record: RecordSummary): string {
  return `${record.wins}–${record.losses}–${record.pushes}`;
}

function StatCard({ label, record, hint }: { label: string; record?: RecordSummary; hint?: string }) {
  return (
    <article className="rounded-lg border border-[var(--border)] bg-white p-3.5">
      <p className="text-xs font-bold uppercase tracking-wider text-[var(--muted)]">{label}</p>
      <p className="numbers-tabular mt-2 text-xl font-bold tracking-tight text-[var(--ink)]">{record ? recordLabel(record) : hint}</p>
      {record ? <p className="numbers-tabular mt-1 text-sm font-semibold text-[var(--success)]">{record.winPct.toFixed(1)}% win rate</p> : null}
      {record ? <p className="mt-1 text-xs text-[var(--muted)]">{record.predictions} graded picks</p> : null}
    </article>
  );
}

function WeekRecord({ title, record, locks }: { title: string; record: RecordSummary; locks: RecordSummary }) {
  return (
    <div>
      <p className="text-[11px] font-bold uppercase tracking-wider text-[var(--muted)]">{title}</p>
      <p className="numbers-tabular mt-1 font-semibold text-[var(--ink)]">{recordLabel(record)} <span className="font-normal text-[var(--muted)]">· {record.winPct.toFixed(1)}%</span></p>
      <p className="numbers-tabular mt-1 text-xs text-[var(--muted)]">Locks {recordLabel(locks)}</p>
    </div>
  );
}

export function ResultsDashboard({ league, games, initialSeason }: { league: FootballLeague; games: GameResult[]; initialSeason?: string }) {
  const seasons = useMemo(() => availableResultSeasons(games), [games]);
  const defaultSeason = initialSeason === "all" || seasons.includes(Number(initialSeason)) ? initialSeason! : String(seasons[0] ?? "all");
  const [season, setSeason] = useState(defaultSeason);
  const filtered = useMemo(() => filterResultsBySeason(games, season), [games, season]);
  const summary = useMemo(() => summarizeResults(filtered), [filtered]);
  const weeks = useMemo(() => weeklyResults(filtered), [filtered]);
  const leagueName = league === "nfl" ? "NFL" : "College Football";

  function changeSeason(value: string) {
    setSeason(value);
    const url = new URL(window.location.href);
    url.searchParams.set("season", value);
    window.history.replaceState(null, "", url);
  }

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-5 pb-16 sm:px-6 sm:py-7 lg:px-8">
      <header className="flex flex-col gap-4 border-b border-[var(--border)] pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div><p className="text-sm font-semibold text-[var(--accent)]">Graded performance</p><h1 className="mt-1 text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">{leagueName} results</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">Transparent records for every model pick and the smaller set that qualified as locks.</p></div>
        <label className="text-sm font-medium text-[var(--muted)]">Season
          <select aria-label="Season" value={season} onChange={(event) => changeSeason(event.target.value)} className="ml-2 min-h-11 rounded-lg border border-[var(--border)] bg-white px-3 font-semibold text-[var(--ink)]">
            {seasons.map((item) => <option key={item} value={item}>{item}</option>)}
            <option value="all">All time</option>
          </select>
        </label>
      </header>

      <section className="mt-7" aria-labelledby="season-summary"><h2 id="season-summary" className="text-xl font-semibold tracking-tight text-[var(--ink)]">{season === "all" ? "All-time summary" : `${season} summary`}</h2>
        <div className="mt-3 grid grid-cols-2 gap-2.5 lg:grid-cols-5">
          <StatCard label="Predicted games" hint={String(summary.predictedGames)} />
          <StatCard label="All spreads" record={summary.spread} />
          <StatCard label="All totals" record={summary.total} />
          <StatCard label="Spread locks" record={summary.spreadLocks} />
          <StatCard label="Total locks" record={summary.totalLocks} />
        </div>
      </section>

      <section className="mt-8" aria-labelledby="weekly-breakdown"><div><h2 id="weekly-breakdown" className="text-xl font-semibold tracking-tight text-[var(--ink)]">Weekly breakdown</h2><p className="mt-1 text-sm text-[var(--muted)]">Records are wins–losses–pushes. Win rate excludes pushes.</p></div>
        <div className="mt-3 space-y-2.5 md:hidden">
          {weeks.map((item) => <article key={`${item.season}-${item.week}`} className="rounded-lg border border-[var(--border)] bg-white p-3.5"><div className="mb-3 flex items-center justify-between"><strong>Week {item.week}</strong>{season === "all" ? <span className="text-xs font-semibold text-[var(--muted)]">{item.season}</span> : null}</div><div className="grid grid-cols-2 gap-3"><WeekRecord title="Spread" record={item.summary.spread} locks={item.summary.spreadLocks} /><WeekRecord title="Total" record={item.summary.total} locks={item.summary.totalLocks} /></div></article>)}
        </div>
        <div className="mt-3 hidden overflow-hidden rounded-lg border border-[var(--border)] bg-white md:block">
          <table className="w-full text-left"><thead><tr className="border-b border-[var(--border)] bg-slate-50 text-xs uppercase tracking-wider text-[var(--muted)]"><th className="px-4 py-3">Season</th><th className="px-4 py-3">Week</th><th className="px-4 py-3">Spread</th><th className="px-4 py-3">Spread locks</th><th className="px-4 py-3">Total</th><th className="px-4 py-3">Total locks</th></tr></thead><tbody>{weeks.map((item) => <tr key={`${item.season}-${item.week}`} className="border-b border-slate-100 last:border-0"><td className="px-4 py-3 text-sm text-[var(--muted)]">{item.season}</td><td className="px-4 py-3 font-semibold">{item.week}</td><td className="numbers-tabular px-4 py-3">{recordLabel(item.summary.spread)} · {item.summary.spread.winPct.toFixed(1)}%</td><td className="numbers-tabular px-4 py-3">{recordLabel(item.summary.spreadLocks)}</td><td className="numbers-tabular px-4 py-3">{recordLabel(item.summary.total)} · {item.summary.total.winPct.toFixed(1)}%</td><td className="numbers-tabular px-4 py-3">{recordLabel(item.summary.totalLocks)}</td></tr>)}</tbody></table>
        </div>
      </section>
    </main>
  );
}

export function ResultsEmptyState({ league }: { league: FootballLeague }) {
  return <main className="mx-auto max-w-3xl px-4 py-14 text-center sm:px-6"><div className="rounded-lg border border-[var(--border)] bg-white p-6"><p className="text-sm font-semibold text-[var(--accent)]">Graded performance</p><h1 className="mt-2 text-3xl font-bold tracking-tight">{league === "nfl" ? "NFL" : "College Football"} results</h1><p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-[var(--muted)]">Results will appear here after games have been completed and graded by the model workflow.</p></div></main>;
}
