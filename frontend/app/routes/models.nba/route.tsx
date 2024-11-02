import { json, type LoaderFunctionArgs } from "@remix-run/node";
import type { ShouldRevalidateFunctionArgs } from "@remix-run/react";
import { useLoaderData, useSearchParams } from "@remix-run/react";
import { NBA_FIRST_BASKET_PICKS_KEY, fetchWithCache } from "~/api/data-utils";
import type { NBAFirstBasketPick } from "~/types/types";
import { FirstBasketTable } from "./FirstBasketTable";
import { SectionTitle } from "~/components/Typography";
import { Input } from "~/components/Input";
import { Card } from "~/components/Card";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const picks = await fetchWithCache<NBAFirstBasketPick[]>(
    NBA_FIRST_BASKET_PICKS_KEY
  );

  return json({
    picks: picks.sort((a, b) => a.sportsbook.localeCompare(b.sportsbook)),
  });
};

export function shouldRevalidate({
  currentUrl,
  nextUrl,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  return currentUrl.pathname === nextUrl.pathname
    ? false
    : defaultShouldRevalidate;
}

export default function NBAModel() {
  const { picks } = useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();
  const bankroll = Number(searchParams.get("bankroll")) || 0;
  if (!Array.isArray(picks) || picks.length === 0) {
    return <div>No data available</div>;
  }

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-28 lg:pb-24 lg:px-12">
      <Card className="flex flex-col gap-4" title="Bankroll">
        <Input
          type="text"
          pattern="[0-9]*"
          value={bankroll}
          onChange={(value) => {
            if (!isNaN(Number(value)))
              setSearchParams({ bankroll: `${value}` }, { replace: true });
          }}
        />
      </Card>
      <SectionTitle>NBA First Basket Picks</SectionTitle>
      <FirstBasketTable data={picks} bankroll={bankroll} />
    </div>
  );
}
