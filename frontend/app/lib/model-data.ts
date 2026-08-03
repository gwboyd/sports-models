import type { CFBPick, NBAFirstBasketPick, NFLPick, NFLResultsResponse } from "@/app/types/types";
import { convertDateTime } from "@/app/lib/formatting";

export function prepareNflData(picksData: NFLPick[], resultsData?: NFLResultsResponse) {
  const data = [...picksData].sort(
    (a, b) => convertDateTime(a.date_time).getTime() - convertDateTime(b.date_time).getTime(),
  );

  return {
    data,
    spreadLocks: data
      .filter((game) => game.spread_lock)
      .sort((a, b) => b.spread_win_prob - a.spread_win_prob),
    totalLocks: data
      .filter((game) => game.total_lock)
      .sort((a, b) => b.total_win_prob - a.total_win_prob),
    overallResults: resultsData?.data,
  };
}

export function prepareCfbData(picksData: CFBPick[]) {
  const data = [...picksData].sort(
    (a, b) => convertDateTime(a.date_time).getTime() - convertDateTime(b.date_time).getTime(),
  );

  return {
    data,
    spreadLocks: data
      .filter((game) => game.spread_lock)
      .sort((a, b) => b.spread_win_prob - a.spread_win_prob),
    totalLocks: data
      .filter((game) => game.total_lock)
      .sort((a, b) => b.total_win_prob - a.total_win_prob),
  };
}

export function prepareNbaData(picks: NBAFirstBasketPick[]) {
  return [...picks].sort((a, b) => a.sportsbook.localeCompare(b.sportsbook));
}
