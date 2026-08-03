import type { Metadata } from "next";
import { connection } from "next/server";
import { CFB_PICK_RESULTS_KEY, fetchOptionalApi } from "@/app/lib/api";
import type { CFBResultsResponse } from "@/app/types/types";
import { ResultsDashboard, ResultsEmptyState } from "../../_components/expected-points/ResultsDashboard";

export const metadata: Metadata = { title: "College Football Results" };

export default async function CfbResultsPage({ searchParams }: { searchParams: Promise<{ season?: string }> }) {
  await connection();
  const [response, query] = await Promise.all([fetchOptionalApi<CFBResultsResponse>(CFB_PICK_RESULTS_KEY), searchParams]);
  if (!response?.games.length) return <ResultsEmptyState league="cfb" />;
  return <ResultsDashboard league="cfb" games={response.games} initialSeason={query.season} />;
}
