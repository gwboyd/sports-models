import { Suspense } from "react";
import { connection } from "next/server";
import { fetchApi, NBA_FIRST_BASKET_PICKS_KEY } from "@/app/lib/api";
import { prepareNbaData } from "@/app/lib/model-data";
import type { NBAFirstBasketPick } from "@/app/types/types";
import { NbaModelClient } from "./_components/NbaModelClient";

export default async function NbaModelPage() {
  await connection();
  const picks = prepareNbaData(await fetchApi<NBAFirstBasketPick[]>(NBA_FIRST_BASKET_PICKS_KEY));

  return (
    <Suspense fallback={<div className="mx-auto w-full max-w-7xl px-4 py-8 text-[var(--muted)] sm:px-6 lg:px-8">Loading model data…</div>}>
      <NbaModelClient picks={picks} />
    </Suspense>
  );
}
