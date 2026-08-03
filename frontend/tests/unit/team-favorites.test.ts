import { describe, expect, it } from "vitest";
import { getTeamIdentity } from "@/app/lib/team-data";
import { parseFavoriteStore } from "@/app/models/_components/expected-points/use-favorites";

describe("team identity and favorites", () => {
  it("normalizes NFL aliases and provides a stable CFB fallback", () => {
    expect(getTeamIdentity("SF", "nfl")).toMatchObject({ id: "sf", displayName: "San Francisco 49ers" });
    expect(getTeamIdentity("Niners", "nfl")).toMatchObject({ id: "sf" });
    expect(getTeamIdentity("Ohio State", "cfb")).toMatchObject({ id: "ohio-state", abbreviation: "OS" });
  });

  it("recovers from invalid storage and deduplicates league favorites", () => {
    expect(parseFavoriteStore("not json")).toEqual({ nfl: [], cfb: [] });
    expect(parseFavoriteStore(JSON.stringify({ nfl: ["sf", "sf"], cfb: ["lsu"] }))).toEqual({ nfl: ["sf"], cfb: ["lsu"] });
  });
});
