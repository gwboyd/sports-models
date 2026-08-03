"use client";

import { Table } from "@/app/components/Table";
import type { ExpectedPointsPick } from "@/app/types/types";
import { AwayTeam, HomeTeam, TotalLine, TotalPlay, TotalPred, TotalWinProb } from "./MatchupCols";

const columns = [HomeTeam, AwayTeam, TotalLine, TotalPred, TotalPlay, TotalWinProb];

export function TotalTable({
  data,
  compact = false,
}: {
  data: ExpectedPointsPick[];
  compact?: boolean;
}) {
  return <Table stickyHeader compact={compact} columns={columns} data={data} />;
}
