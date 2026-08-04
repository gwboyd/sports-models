import type { Metadata } from "next";
import { connection } from "next/server";
import { fetchOptionalApi, NFL_PICK_RESULTS_KEY } from "@/app/lib/api";
import type { NFLResultsResponse } from "@/app/types/types";
import { ResultsDashboard, ResultsEmptyState } from "../../_components/expected-points/ResultsDashboard";

export const metadata: Metadata = { title: "NFL Results" };

export default async function NflResultsPage({ searchParams }: { searchParams: Promise<{ season?: string }> }) {
  await connection();
  const [response, query] = await Promise.all([fetchOptionalApi<NFLResultsResponse>(NFL_PICK_RESULTS_KEY), searchParams]);
  if (!response?.games.length) return <ResultsEmptyState league="nfl" />;
  return <ResultsDashboard league="nfl" games={response.games} initialSeason={query.season} />;
}
