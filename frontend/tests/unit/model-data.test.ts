import { describe, expect, it } from "vitest";
import { prepareCfbData, prepareNbaData, prepareNflData } from "@/app/lib/model-data";
import { displaySpread } from "@/app/lib/formatting";
import type { CFBPick, NBAFirstBasketPick, NFLPick, NFLResultsResponse } from "@/app/types/types";

const basePick: NFLPick = {
  season: 2024,
  week: "1",
  home_team: "Home",
  away_team: "Away",
  home_score_pred: 24,
  away_score_pred: 21,
  spread_pred: 3,
  spread_line: 2.5,
  spread_play: "Home",
  spread_win_prob: 55,
  spread_lock: 0,
  total_pred: 45,
  total_line: 44.5,
  total_play: "Over",
  total_win_prob: 55,
  total_lock: 0,
  game_id: "game-1",
  year_week: "2024-1",
  date_time: "2024-09-08-13:00",
  write_time: "2024-09-01T00:00:00Z",
};

const results: NFLResultsResponse = {
  data: {
    predicted_games: 1,
    spread_wins: 1,
    spread_losses: 0,
    spread_pushes: 0,
    spread_win_pct: 100,
    spread_lock_predictions: 1,
    spread_lock_wins: 1,
    spread_lock_losses: 0,
    spread_lock_pushes: 0,
    spread_lock_win_pct: 100,
    total_wins: 1,
    total_losses: 0,
    total_pushes: 0,
    total_win_pct: 100,
    total_lock_predictions: 1,
    total_lock_wins: 1,
    total_lock_losses: 0,
    total_lock_pushes: 0,
    total_lock_win_pct: 100,
  },
  games: [],
};

const baseCfbPick: CFBPick = {
  ...basePick,
  season: 2026,
  year_week: "2026_1",
  home_conference: "SEC",
  away_conference: "SEC",
};

describe("model data preparation", () => {
  it("sorts NFL games chronologically and locks by descending probability", () => {
    const later: NFLPick = { ...basePick, game_id: "later", date_time: "2024-09-08-16:00", spread_lock: 1, spread_win_prob: 60 };
    const earlier: NFLPick = { ...basePick, game_id: "earlier", date_time: "2024-09-08-10:00", spread_lock: 1, spread_win_prob: 70, total_lock: 1, total_win_prob: 58 };
    const prepared = prepareNflData([later, earlier], results);

    expect(prepared.data.map((pick) => pick.game_id)).toEqual(["earlier", "later"]);
    expect(prepared.spreadLocks.map((pick) => pick.game_id)).toEqual(["earlier", "later"]);
    expect(prepared.totalLocks.map((pick) => pick.game_id)).toEqual(["earlier"]);
  });

  it("sorts NBA picks by sportsbook without mutating the input", () => {
    const picks: NBAFirstBasketPick[] = [
      { date: "2024-01-01", player_name: "B", team: "B", fb_model_prob: 0.2, fb_model_odds: 100, odds: 100, sportsbook: "Zulu", units: 1 },
      { date: "2024-01-01", player_name: "A", team: "A", fb_model_prob: 0.2, fb_model_odds: 100, odds: 100, sportsbook: "Alpha", units: 1 },
    ];

    expect(prepareNbaData(picks).map((pick) => pick.sportsbook)).toEqual(["Alpha", "Zulu"]);
    expect(picks.map((pick) => pick.sportsbook)).toEqual(["Zulu", "Alpha"]);
  });

  it("groups CFB games in display order and duplicates cross-conference games", () => {
    const sameConference: CFBPick = {
      ...baseCfbPick,
      game_id: "sec-game",
      date_time: "2026-09-05-10:00",
    };
    const crossConference: CFBPick = {
      ...baseCfbPick,
      game_id: "cross-game",
      date_time: "2026-09-05-11:00",
      away_conference: "Big Ten",
      spread_lock: 1,
      spread_win_prob: 60,
    };
    const otherConference: CFBPick = {
      ...baseCfbPick,
      game_id: "other-game",
      date_time: "2026-09-05-12:00",
      home_conference: "Big Sky",
      away_conference: "Ivy",
    };

    const prepared = prepareCfbData([otherConference, crossConference, sameConference]);
    const groups = Object.fromEntries(
      prepared.conferenceGroups.map((group) => [
        group.conference,
        group.games.map((game) => game.game_id),
      ]),
    );

    expect(prepared.data.map((game) => game.game_id)).toEqual([
      "sec-game",
      "cross-game",
      "other-game",
    ]);
    expect(prepared.conferenceGroups.map((group) => group.conference)).toEqual([
      "SEC",
      "BIG 10",
      "Others",
    ]);
    expect(groups.SEC).toEqual(["sec-game", "cross-game"]);
    expect(groups["BIG 10"]).toEqual(["cross-game"]);
    expect(groups.Others).toEqual(["other-game", "other-game"]);
    expect(prepared.spreadLocks.map((game) => game.game_id)).toEqual(["cross-game"]);
  });

  it("retains spread formatting", () => {
    expect(displaySpread(2.5)).toBe("+2.5");
    expect(displaySpread(-2.5, 2)).toBe("-2.50");
  });
});
