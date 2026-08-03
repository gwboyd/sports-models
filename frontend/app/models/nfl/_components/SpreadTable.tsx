"use client";

import { Table } from "@/app/components/Table";
import type { NFLPick } from "@/app/types/types";
import { AwayTeam, HomeTeam, SpreadLine, SpreadPlay, SpreadPred, SpreadWinProb } from "./MatchupCols";

const columns = [HomeTeam, AwayTeam, SpreadLine, SpreadPred, SpreadPlay, SpreadWinProb];

export function SpreadTable({ data }: { data: NFLPick[] }) {
  return <Table stickyHeader columns={columns} data={data} />;
}
