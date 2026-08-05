import { displaySpread, formatGameDate } from "@/app/lib/formatting";
import { getTeamIdentity, normalizedSearch, teamSearchText } from "@/app/lib/team-data";
import type { CFBPick, ExpectedPointsPick, FootballLeague } from "@/app/types/types";

export type LockPick = {
  id: string;
  market: "spread" | "total";
  game: ExpectedPointsPick;
  probability: number;
};

export function spreadPickLabel(game: ExpectedPointsPick): string {
  const multiplier = game.spread_play === game.away_team ? -1 : 1;
  return `${game.spread_play} ${displaySpread(multiplier * game.spread_line)}`;
}

export function spreadModelLabel(game: ExpectedPointsPick): string {
  const multiplier = game.spread_play === game.away_team ? -1 : 1;
  return `${game.spread_play} ${displaySpread(multiplier * game.spread_pred)}`;
}

export function totalPickLabel(game: ExpectedPointsPick): string {
  const play = game.total_play ? `${game.total_play[0].toUpperCase()}${game.total_play.slice(1).toLowerCase()}` : "Total";
  return `${play} ${game.total_line.toFixed(1)}`;
}

export function predictedScoreLabel(game: ExpectedPointsPick): string {
  return `${game.away_team} ${game.away_score_pred.toFixed(1)} · ${game.home_team} ${game.home_score_pred.toFixed(1)}`;
}

export function getLocks(games: ExpectedPointsPick[]): LockPick[] {
  return games
    .flatMap((game): LockPick[] => [
      ...(game.spread_lock ? [{ id: `${game.game_id}-spread`, market: "spread" as const, game, probability: game.spread_win_prob }] : []),
      ...(game.total_lock ? [{ id: `${game.game_id}-total`, market: "total" as const, game, probability: game.total_win_prob }] : []),
    ])
    .sort((a, b) => b.probability - a.probability);
}

export function groupGamesByDate(games: ExpectedPointsPick[], timeZone?: string): Array<{ date: string; games: ExpectedPointsPick[] }> {
  const grouped = new Map<string, ExpectedPointsPick[]>();
  for (const game of games) {
    const date = formatGameDate(game.date_time, timeZone);
    grouped.set(date, [...(grouped.get(date) ?? []), game]);
  }
  return [...grouped].map(([date, dateGames]) => ({ date, games: dateGames }));
}

function cfbConferences(game: ExpectedPointsPick): string[] {
  if (!("home_conference" in game)) return [];
  const pick = game as CFBPick;
  return [pick.home_conference, pick.away_conference].filter((value): value is string => Boolean(value));
}

export function getConferences(games: ExpectedPointsPick[]): string[] {
  return [...new Set(games.flatMap(cfbConferences))].sort((a, b) => a.localeCompare(b));
}

export function filterGamesByConference(games: ExpectedPointsPick[], conference: string): ExpectedPointsPick[] {
  if (!conference) return games;
  return games.filter((game) => cfbConferences(game).includes(conference));
}

export function searchGames(games: ExpectedPointsPick[], league: FootballLeague, query: string): ExpectedPointsPick[] {
  const needle = normalizedSearch(query);
  if (!needle) return games;
  return games.filter((game) => {
    const home = getTeamIdentity(game.home_team, league);
    const away = getTeamIdentity(game.away_team, league);
    const haystack = normalizedSearch(`${teamSearchText(home)} ${teamSearchText(away)} ${cfbConferences(game).join(" ")}`);
    return haystack.includes(needle);
  });
}

export function gameDomId(gameId: string): string {
  return `game-${gameId.replace(/[^A-Za-z0-9_-]/g, "-")}`;
}
