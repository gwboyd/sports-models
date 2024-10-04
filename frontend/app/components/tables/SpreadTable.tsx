import type { Matchup } from "~/types/apiData";
import { Table } from "../Table";
import {
  AwayTeam,
  HomeTeam,
  SpreadLine,
  SpreadPlay,
  SpreadPred,
  SpreadWinProb,
} from "./MatchupCols";

export function SpreadTable({ data }: { data: Matchup[] }) {
  return <Table stickyHeader columns={columns} data={data} />;
}

const columns = [
  HomeTeam,
  AwayTeam,
  SpreadLine,
  SpreadPred,
  SpreadPlay,
  SpreadWinProb,
];
