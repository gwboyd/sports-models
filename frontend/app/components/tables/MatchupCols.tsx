import { createColumnHelper } from "@tanstack/react-table";
import type { Matchup } from "~/types/apiData";

export const columnHelper = createColumnHelper<Matchup>();

export const HomeTeam = columnHelper.accessor("home_team", {
  header: "Home Team",
  cell: (info) => info.getValue(),
});

export const AwayTeam = columnHelper.accessor("away_team", {
  header: "Away Team",
  cell: (info) => info.getValue(),
});

export const HomeScorePred = columnHelper.accessor("home_score_pred", {
  header: "Home Score Pred",
  cell: (info) => info.getValue().toFixed(2),
});

export const AwayScorePred = columnHelper.accessor("away_score_pred", {
  header: "Away Score Pred",
  cell: (info) => info.getValue().toFixed(2),
});

export const SpreadPred = columnHelper.accessor("spread_pred", {
  header: "Spread Pred",
  cell: (info) =>
    `${info.row.original.home_team} (${info.getValue().toFixed(2)})`,
});

export const SpreadLine = columnHelper.accessor("spread_line", {
  header: "Spread Line",
  cell: (info) =>
    `${info.row.original.home_team} (${info.getValue().toFixed(1)})`,
});

export const SpreadPlay = columnHelper.accessor("spread_play", {
  header: "Spread Play",
  cell: (info) => info.getValue(),
});

export const SpreadWinProb = columnHelper.accessor("spread_win_prob", {
  header: "Spread Win Prob",
  cell: (info) => info.getValue().toFixed(2),
});

export const TotalPred = columnHelper.accessor("total_pred", {
  header: "Total Pred",
  cell: (info) => info.getValue().toFixed(2),
});

export const TotalLine = columnHelper.accessor("total_line", {
  header: "Total Line",
  cell: (info) => info.getValue().toFixed(2),
});

export const TotalPlay = columnHelper.accessor("total_play", {
  header: "Total Play",
  cell: (info) => info.getValue(),
});

export const TotalWinProb = columnHelper.accessor("total_win_prob", {
  header: "Total Win Prob",
  cell: (info) => info.getValue().toFixed(2),
});
