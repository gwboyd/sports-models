import { describe, expect, it } from "vitest";
import { getTeamIdentity } from "@/app/lib/team-data";
import { parseFavoriteStore } from "@/app/models/_components/expected-points/use-favorites";

describe("team identity and favorites", () => {
  it("normalizes NFL aliases and resolves generated CFB metadata", () => {
    expect(getTeamIdentity("SF", "nfl")).toMatchObject({ id: "sf", displayName: "San Francisco 49ers" });
    expect(getTeamIdentity("Niners", "nfl")).toMatchObject({ id: "sf" });
    expect(getTeamIdentity("Ohio State", "cfb")).toMatchObject({
      id: "cfb-194",
      abbreviation: "OSU",
      logoPath: "/teams/cfb/194.png",
    });
    expect(getTeamIdentity("Air Force Falcons", "cfb")).toMatchObject({ id: "cfb-2005", conference: "Mountain West" });
    const unknownTeam = getTeamIdentity("Unmapped University", "cfb");
    expect(unknownTeam).toMatchObject({ id: "unmapped-university" });
    expect(unknownTeam.logoPath).toBeUndefined();
  });

  it("recovers from invalid storage and deduplicates league favorites", () => {
    expect(parseFavoriteStore("not json")).toEqual({ nfl: [], cfb: [] });
    expect(parseFavoriteStore(JSON.stringify({ nfl: ["sf", "sf"], cfb: ["lsu"] }))).toEqual({ nfl: ["sf"], cfb: ["lsu"] });
  });
});
