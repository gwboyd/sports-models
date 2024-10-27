import type { NFLPick } from "~/types/types";
import { Table } from "../Table";
import {
  AwayTeam,
  HomeTeam,
  TotalLine,
  TotalPlay,
  TotalPred,
  TotalWinProb,
} from "./MatchupCols";

export function TotalTable({ data }: { data: NFLPick[] }) {
  return <Table stickyHeader columns={columns} data={data} />;
}

const columns = [
  HomeTeam,
  AwayTeam,
  TotalLine,
  TotalPred,
  TotalPlay,
  TotalWinProb,
];
