import type { Metadata } from "next";
import { connection } from "next/server";
import { CFB_PICKS_KEY, fetchApi } from "@/app/lib/api";
import { prepareCfbData } from "@/app/lib/model-data";
import type { CFBPick } from "@/app/types/types";
import { FootballDashboard } from "../_components/expected-points/FootballDashboard";

export const metadata: Metadata = { title: "College Football Games" };

export default async function CfbModelPage() {
  await connection();
  const { data } = prepareCfbData(await fetchApi<CFBPick[]>(CFB_PICKS_KEY));
  if (data.length === 0) return <main className="mx-auto max-w-7xl p-6 text-[var(--muted)]">No college football games are available.</main>;
  return <FootballDashboard league="cfb" games={data} />;
}
