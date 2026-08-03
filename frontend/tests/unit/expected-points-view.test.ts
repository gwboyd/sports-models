import { describe, expect, it } from "vitest";
import { filterGamesByConference, getLocks, searchGames, spreadModelLabel, spreadPickLabel, totalPickLabel } from "@/app/models/_components/expected-points/view-model";
import type { CFBPick, NFLPick } from "@/app/types/types";

const pick: NFLPick = {
  season: 2025, week: "1", home_team: "SF", away_team: "SEA", home_score_pred: 27.31,
  away_score_pred: 22.69, spread_pred: -4.62, spread_line: 1.5, spread_play: "SF",
  spread_win_prob: 63.25, spread_lock: 1, total_pred: 46, total_line: 44.5,
  total_play: "over", total_win_prob: 58.5, total_lock: 1, game_id: "1",
  year_week: "2025-1", date_time: "2025-09-07-13:00", write_time: "2025-09-01T00:00:00Z",
};

describe("expected points view models", () => {
  it("shows the actionable spread instead of separate line and prediction columns", () => {
    expect(spreadPickLabel(pick)).toBe("SF +1.5");
    expect(spreadModelLabel(pick)).toBe("SF -4.6");
    expect(totalPickLabel(pick)).toBe("Over 44.5");
  });

  it("creates a separate card for each locked market", () => {
    expect(getLocks([pick]).map((lock) => lock.market)).toEqual(["spread", "total"]);
  });

  it("searches NFL aliases and CFB conferences", () => {
    expect(searchGames([pick], "nfl", "Niners")).toEqual([pick]);
    const cfbPick: CFBPick = { ...pick, home_team: "LSU", away_team: "Clemson", home_conference: "SEC", away_conference: "ACC" };
    expect(searchGames([cfbPick], "cfb", "SEC")).toEqual([cfbPick]);
    expect(filterGamesByConference([cfbPick], "ACC")).toEqual([cfbPick]);
    expect(filterGamesByConference([cfbPick], "Big Ten")).toEqual([]);
  });
});
