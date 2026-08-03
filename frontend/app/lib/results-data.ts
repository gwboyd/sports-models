import type { GameResult } from "@/app/types/types";

export type RecordSummary = {
  predictions: number;
  wins: number;
  losses: number;
  pushes: number;
  winPct: number;
};

export type ResultsSummary = {
  predictedGames: number;
  spread: RecordSummary;
  total: RecordSummary;
  spreadLocks: RecordSummary;
  totalLocks: RecordSummary;
};

function record(values: Array<number | null | undefined>): RecordSummary {
  const wins = values.filter((value) => value === 1).length;
  const losses = values.filter((value) => value === 0).length;
  const pushes = values.filter((value) => value == null).length;
  const decided = wins + losses;
  return { predictions: values.length, wins, losses, pushes, winPct: decided ? (100 * wins) / decided : 0 };
}

export function summarizeResults(games: GameResult[]): ResultsSummary {
  const spreadLocks = games.filter((game) => Boolean(game.spread_lock));
  const totalLocks = games.filter((game) => Boolean(game.total_lock));
  return {
    predictedGames: games.length,
    spread: record(games.map((game) => game.spread_win)),
    total: record(games.map((game) => game.total_win)),
    spreadLocks: record(spreadLocks.map((game) => game.spread_win)),
    totalLocks: record(totalLocks.map((game) => game.total_win)),
  };
}

export function availableResultSeasons(games: GameResult[]): number[] {
  return [...new Set(games.map((game) => game.season))].sort((a, b) => b - a);
}

export function filterResultsBySeason(games: GameResult[], season: string): GameResult[] {
  if (season === "all") return games;
  return games.filter((game) => game.season === Number(season));
}

export function weeklyResults(games: GameResult[]) {
  const groups = new Map<string, GameResult[]>();
  for (const game of games) {
    const key = `${game.season}-${game.week}`;
    groups.set(key, [...(groups.get(key) ?? []), game]);
  }
  return [...groups.values()]
    .map((weekGames) => ({ season: weekGames[0].season, week: weekGames[0].week, summary: summarizeResults(weekGames) }))
    .sort((a, b) => b.season - a.season || Number(b.week) - Number(a.week));
}
