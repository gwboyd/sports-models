"use client";

import { Table } from "@/app/components/Table";
import type { NFLPick } from "@/app/types/types";
import { AwayTeam, HomeTeam, TotalLine, TotalPlay, TotalPred, TotalWinProb } from "./MatchupCols";

const columns = [HomeTeam, AwayTeam, TotalLine, TotalPred, TotalPlay, TotalWinProb];

export function TotalTable({ data }: { data: NFLPick[] }) {
  return <Table stickyHeader columns={columns} data={data} />;
}
