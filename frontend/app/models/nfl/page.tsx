import type { Metadata } from "next";
import { connection } from "next/server";
import { fetchApi, NFL_PICKS_KEY } from "@/app/lib/api";
import { prepareNflData } from "@/app/lib/model-data";
import type { NFLPick } from "@/app/types/types";
import { FootballDashboard } from "../_components/expected-points/FootballDashboard";

export const metadata: Metadata = { title: "NFL Games" };

export default async function NflModelPage() {
  await connection();
  const { data } = prepareNflData(await fetchApi<NFLPick[]>(NFL_PICKS_KEY));
  if (data.length === 0) return <main className="mx-auto max-w-7xl p-6 text-[var(--muted)]">No NFL games are available.</main>;
  return <FootballDashboard league="nfl" games={data} />;
}
