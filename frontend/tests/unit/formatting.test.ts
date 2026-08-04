import { describe, expect, it } from "vitest";
import { convertDateTime, formatGameDate, formatKickoff } from "@/app/lib/formatting";

describe("football kickoff timezone formatting", () => {
  it("interprets NFL wall-clock values in New York with daylight saving time", () => {
    expect(convertDateTime("2025-09-07-13:00", "nfl").toISOString()).toBe("2025-09-07T17:00:00.000Z");
    expect(convertDateTime("2026-01-04-13:00", "nfl").toISOString()).toBe("2026-01-04T18:00:00.000Z");
  });

  it("interprets CFB values as UTC and displays both leagues in the requested device timezone", () => {
    expect(convertDateTime("2026-09-05-17:00", "cfb").toISOString()).toBe("2026-09-05T17:00:00.000Z");
    expect(formatKickoff("2025-09-07-13:00", "nfl", "America/Los_Angeles")).toBe("Sun, Sep 7, 10:00 AM");
    expect(formatKickoff("2026-09-05-17:00", "cfb", "America/Los_Angeles")).toBe("Sat, Sep 5, 10:00 AM");
  });

  it("uses the local calendar date when a UTC kickoff crosses midnight", () => {
    expect(formatGameDate("2026-09-06-01:00", "cfb", "America/Los_Angeles")).toBe("Saturday, September 5");
  });
});
