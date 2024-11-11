import type { NBAFirstBasketPick } from "~/types/types";
import { Table } from "../../components/Table";
import { BetAmount, Date, Odds, PlayerName, Sportsbook, Team } from "./Cols";

export function FirstBasketTable({
  data,
  bankroll,
}: {
  data: NBAFirstBasketPick[];
  bankroll: number;
}) {
  return (
    <Table stickyHeader columns={columns} data={data} meta={{ bankroll }} />
  );
}

const columns = [Date, PlayerName, Team, Sportsbook, BetAmount, Odds];
