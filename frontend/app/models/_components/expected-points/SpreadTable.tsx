"use client";

import { Table } from "@/app/components/Table";
import type { ExpectedPointsPick } from "@/app/types/types";
import { AwayTeam, HomeTeam, SpreadLine, SpreadPlay, SpreadPred, SpreadWinProb } from "./MatchupCols";

const columns = [HomeTeam, AwayTeam, SpreadLine, SpreadPred, SpreadPlay, SpreadWinProb];

export function SpreadTable({
  data,
  compact = false,
}: {
  data: ExpectedPointsPick[];
  compact?: boolean;
}) {
  return <Table stickyHeader compact={compact} columns={columns} data={data} />;
}
