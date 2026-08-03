"use client";

import { createColumnHelper } from "@tanstack/react-table";
import dayjs from "dayjs";
import type { NBAFirstBasketPick } from "@/app/types/types";

const columnHelper = createColumnHelper<NBAFirstBasketPick>();

export const Date = columnHelper.accessor("date", { header: "Date", cell: (info) => dayjs(info.getValue()).format("MMM D, YYYY") });
export const PlayerName = columnHelper.accessor("player_name", { header: "Player", cell: (info) => info.getValue() });
export const Team = columnHelper.accessor("team", { header: "Team", cell: (info) => info.getValue() });
export const Sportsbook = columnHelper.accessor("sportsbook", { header: "Sportsbook", cell: (info) => info.getValue() });
export const Odds = columnHelper.accessor("odds", { header: "Line", cell: (info) => `+${info.getValue()}` });
export const BetAmount = columnHelper.accessor("units", {
  header: "Amt",
  cell: ({ getValue, table }) => `$${Math.round((table.options.meta?.bankroll ?? 0) * (5 / 800) * getValue())}`,
});

export const columns = [Date, PlayerName, Team, Sportsbook, BetAmount, Odds];
