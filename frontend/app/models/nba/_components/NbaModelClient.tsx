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
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-8 pb-20 sm:px-6 lg:px-8">
      <header>
        <p className="text-sm font-semibold text-[var(--accent)]">NBA model</p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">First basket picks</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">Size suggested bets using your current bankroll.</p>
      </header>
      <Card className="max-w-md" title="Bankroll">
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
      <SectionTitle>Current picks</SectionTitle>
      <FirstBasketTable data={picks} bankroll={urlBankroll} />
    </main>
  );
}
