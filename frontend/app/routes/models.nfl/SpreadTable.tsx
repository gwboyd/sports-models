import type { NFLPick } from "~/types/types";
import { Table } from "../../components/Table";
import {
  AwayTeam,
  HomeTeam,
  SpreadLine,
  SpreadPlay,
  SpreadPred,
  SpreadWinProb,
} from "./MatchupCols";

export function SpreadTable({ data }: { data: NFLPick[] }) {
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
