import { describe, expect, it } from "vitest";
import { availableResultSeasons, filterResultsBySeason, summarizeResults, weeklyResults } from "@/app/lib/results-data";
import type { GameResult } from "@/app/types/types";

function game(overrides: Partial<GameResult>): GameResult {
  return {
    season: 2025, week: "1", home_team: "SF", away_team: "SEA", home_score: 24, away_score: 20,
    home_score_pred: 27, away_score_pred: 21, spread_pred: -6, spread_line: -3.5, true_spread: -4,
    spread_play: "SF", spread_win_prob: 60, spread_lock: 1, correct_spread_play: "SF", spread_win: 1,
    total_pred: 48, total_line: 45.5, true_total: 44, total_play: "over", total_win_prob: 55,
    total_lock: 0, correct_total_play: "under", total_win: 0, year_week: "2025-1", game_id: "1",
    date_time: "2025-09-07-13:00", ...overrides,
  };
}

describe("results aggregation", () => {
  const games = [
    game({ game_id: "1", spread_win: 1, total_win: 0 }),
    game({ game_id: "2", spread_win: 0, total_win: 1, spread_lock: 0, total_lock: 1 }),
    game({ game_id: "3", spread_win: undefined, total_win: undefined, week: "2" }),
    game({ game_id: "4", season: 2024, year_week: "2024-1", spread_win: 1, total_win: 1 }),
  ];

  it("excludes pushes from win percentage while retaining them in the record", () => {
    const summary = summarizeResults(games.slice(0, 3));
    expect(summary.spread).toMatchObject({ wins: 1, losses: 1, pushes: 1, winPct: 50 });
    expect(summary.total).toMatchObject({ wins: 1, losses: 1, pushes: 1, winPct: 50 });
  });

  it("filters seasons and builds newest-first weekly summaries", () => {
    expect(availableResultSeasons(games)).toEqual([2025, 2024]);
    expect(filterResultsBySeason(games, "2024")).toHaveLength(1);
    expect(weeklyResults(games).map((item) => `${item.season}-${item.week}`)).toEqual(["2025-2", "2025-1", "2024-1"]);
  });
});
