/* eslint-disable @next/next/no-img-element */
import type { Metadata } from "next";

export const metadata: Metadata = { title: "NFL Model Insights" };

const insights = [
  { title: "Power rankings", description: "A neutral-field view of how the current model rates every NFL team using its underlying efficiency profile.", src: "https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/power_rankings.png", wide: true },
  { title: "Offensive EPA", description: "The model's current view of how efficiently each offense creates expected points.", src: "https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/offensive_epa.png" },
  { title: "Defensive EPA", description: "A live comparison of how effectively each defense limits expected points.", src: "https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/defensive_epa.png" },
  { title: "Feature importance", description: "Which inputs currently contribute most to the expected-score model.", src: "https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/feature_importance.png" },
  { title: "Dynamic moving averages", description: "An example of how recent performance is smoothed to reduce single-game noise without losing current form.", src: "https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png", wide: true },
] as const;

export default function InsightsPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 pb-20 sm:px-6 lg:px-8">
      <header className="max-w-3xl"><p className="text-sm font-semibold text-[var(--accent)]">Behind the forecast</p><h1 className="mt-1 text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">Model insights</h1><p className="mt-3 text-sm leading-6 text-[var(--muted)] sm:text-base">A live look at the team strength, efficiency, and model inputs shaping this week&apos;s predictions.</p></header>
      <div className="mt-8 grid gap-5 lg:grid-cols-2">
        {insights.map((insight) => <figure key={insight.src} className={`rounded-2xl border border-[var(--border)] bg-white p-4 shadow-[0_1px_2px_rgba(16,24,40,0.04)] sm:p-5 ${"wide" in insight && insight.wide ? "lg:col-span-2" : ""}`}><figcaption><h2 className="text-lg font-semibold text-[var(--ink)]">{insight.title}</h2><p className="mt-1 text-sm leading-6 text-[var(--muted)]">{insight.description}</p></figcaption><a href={insight.src} target="_blank" rel="noreferrer" className="mt-4 block overflow-hidden rounded-xl border border-slate-100 bg-white focus:ring-4 focus:ring-blue-100"><img src={insight.src} alt={`${insight.title} chart`} className="h-auto w-full object-contain" /></a><p className="mt-2 text-right text-xs text-[var(--muted)]">Open chart full size</p></figure>)}
      </div>
    </main>
  );
}
