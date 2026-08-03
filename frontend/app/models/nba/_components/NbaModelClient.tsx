"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { Card } from "@/app/components/Card";
import { Input } from "@/app/components/Input";
import { SectionTitle } from "@/app/components/Typography";
import type { NBAFirstBasketPick } from "@/app/types/types";
import { FirstBasketTable } from "./FirstBasketTable";

function parseBankroll(value: string | null): number {
  return Number(value) || 0;
}

export function NbaModelClient({ picks }: { picks: NBAFirstBasketPick[] }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const urlBankroll = parseBankroll(searchParams.get("bankroll"));
  if (picks.length === 0) return <div>No data available</div>;

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-28 lg:pb-24 lg:px-12">
      <Card className="flex flex-col gap-4" title="Bankroll">
        <Input
          type="text"
          pattern="[0-9]*"
          value={urlBankroll}
          onChange={(value) => {
            const rawValue = String(value);
            if (Number.isNaN(Number(rawValue))) return;
            const query = new URLSearchParams({ bankroll: rawValue });
            window.history.replaceState(null, "", `${pathname}?${query.toString()}`);
          }}
        />
      </Card>
      <SectionTitle>NBA First Basket Picks</SectionTitle>
      <FirstBasketTable data={picks} bankroll={urlBankroll} />
    </div>
  );
}
