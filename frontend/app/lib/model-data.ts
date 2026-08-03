import type { CFBPick, NBAFirstBasketPick, NFLPick, NFLResultsResponse } from "@/app/types/types";
import { convertDateTime } from "@/app/lib/formatting";

export const CFB_CONFERENCE_ORDER = [
  "SEC",
  "BIG 10",
  "ACC",
  "Big 12",
  "American",
  "Pac 12",
  "Mountain West",
  "Sun Belt",
  "Conference USA",
  "MAC",
  "Others",
] as const;

export type CfbConferenceName = (typeof CFB_CONFERENCE_ORDER)[number];

const CONFERENCE_NAMES: Record<string, CfbConferenceName> = {
  sec: "SEC",
  "big ten": "BIG 10",
  acc: "ACC",
  "big 12": "Big 12",
  "american athletic": "American",
  "pac-12": "Pac 12",
  "pac 12": "Pac 12",
  "mountain west": "Mountain West",
  "sun belt": "Sun Belt",
  "conference usa": "Conference USA",
  "mid-american": "MAC",
  mac: "MAC",
};

function conferenceIdentity(conference: string | null | undefined): string {
  return conference?.trim().toLocaleLowerCase() || "__unknown__";
}

function conferenceGroup(conference: string | null | undefined): CfbConferenceName {
  return CONFERENCE_NAMES[conferenceIdentity(conference)] ?? "Others";
}

export function prepareNflData(picksData: NFLPick[], resultsData: NFLResultsResponse) {
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
    overallResults: resultsData.data,
  };
}

export function prepareCfbData(picksData: CFBPick[]) {
  const data = [...picksData].sort(
    (a, b) => convertDateTime(a.date_time).getTime() - convertDateTime(b.date_time).getTime(),
  );
  const grouped = new Map<CfbConferenceName, CFBPick[]>(
    CFB_CONFERENCE_ORDER.map((conference) => [conference, []]),
  );

  for (const game of data) {
    const homeIdentity = conferenceIdentity(game.home_conference);
    const awayIdentity = conferenceIdentity(game.away_conference);
    const placements =
      homeIdentity === awayIdentity
        ? [game.home_conference]
        : [game.home_conference, game.away_conference];

    for (const conference of placements) {
      grouped.get(conferenceGroup(conference))?.push(game);
    }
  }

  return {
    data,
    spreadLocks: data
      .filter((game) => game.spread_lock)
      .sort((a, b) => b.spread_win_prob - a.spread_win_prob),
    totalLocks: data
      .filter((game) => game.total_lock)
      .sort((a, b) => b.total_win_prob - a.total_win_prob),
    conferenceGroups: CFB_CONFERENCE_ORDER.map((conference) => ({
      conference,
      games: grouped.get(conference) ?? [],
    })).filter((group) => group.games.length > 0),
  };
}

export function prepareNbaData(picks: NBAFirstBasketPick[]) {
  return [...picks].sort((a, b) => a.sportsbook.localeCompare(b.sportsbook));
}
