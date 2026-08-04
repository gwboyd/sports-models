import type { ExpectedPointsPick, FootballLeague } from "@/app/types/types";
import cfbTeamsData from "@/app/generated/cfb-teams.json";
import nflTeamsData from "@/app/generated/nfl-teams.json";

export type TeamIdentity = {
  id: string;
  league: FootballLeague;
  displayName: string;
  abbreviation: string;
  aliases: string[];
  logoPath?: string;
  externalId?: number;
  mascot?: string;
  conference?: string;
  color?: string;
  alternateColor?: string;
};

type GeneratedTeam = Omit<TeamIdentity, "league">;

type TeamSeed = [id: string, displayName: string, abbreviation: string, aliases: string[]];

const nflTeamSeeds: TeamSeed[] = [
  ["ari", "Arizona Cardinals", "ARI", ["Arizona", "Cardinals"]],
  ["atl", "Atlanta Falcons", "ATL", ["Atlanta", "Falcons"]],
  ["bal", "Baltimore Ravens", "BAL", ["Baltimore", "Ravens"]],
  ["buf", "Buffalo Bills", "BUF", ["Buffalo", "Bills"]],
  ["car", "Carolina Panthers", "CAR", ["Carolina", "Panthers"]],
  ["chi", "Chicago Bears", "CHI", ["Chicago", "Bears"]],
  ["cin", "Cincinnati Bengals", "CIN", ["Cincinnati", "Bengals"]],
  ["cle", "Cleveland Browns", "CLE", ["Cleveland", "Browns"]],
  ["dal", "Dallas Cowboys", "DAL", ["Dallas", "Cowboys"]],
  ["den", "Denver Broncos", "DEN", ["Denver", "Broncos"]],
  ["det", "Detroit Lions", "DET", ["Detroit", "Lions"]],
  ["gb", "Green Bay Packers", "GB", ["Green Bay", "Packers"]],
  ["hou", "Houston Texans", "HOU", ["Houston", "Texans"]],
  ["ind", "Indianapolis Colts", "IND", ["Indianapolis", "Colts"]],
  ["jax", "Jacksonville Jaguars", "JAX", ["Jacksonville", "Jaguars"]],
  ["kc", "Kansas City Chiefs", "KC", ["Kansas City", "Chiefs"]],
  ["lv", "Las Vegas Raiders", "LV", ["Las Vegas", "Raiders", "Oakland"]],
  ["lac", "Los Angeles Chargers", "LAC", ["Chargers", "LA Chargers"]],
  ["lar", "Los Angeles Rams", "LA", ["LAR", "Rams", "LA Rams"]],
  ["mia", "Miami Dolphins", "MIA", ["Miami", "Dolphins"]],
  ["min", "Minnesota Vikings", "MIN", ["Minnesota", "Vikings"]],
  ["ne", "New England Patriots", "NE", ["New England", "Patriots"]],
  ["no", "New Orleans Saints", "NO", ["New Orleans", "Saints"]],
  ["nyg", "New York Giants", "NYG", ["Giants", "NY Giants"]],
  ["nyj", "New York Jets", "NYJ", ["Jets", "NY Jets"]],
  ["phi", "Philadelphia Eagles", "PHI", ["Philadelphia", "Eagles"]],
  ["pit", "Pittsburgh Steelers", "PIT", ["Pittsburgh", "Steelers"]],
  ["sea", "Seattle Seahawks", "SEA", ["Seattle", "Seahawks"]],
  ["sf", "San Francisco 49ers", "SF", ["San Francisco", "49ers", "Niners"]],
  ["tb", "Tampa Bay Buccaneers", "TB", ["Tampa Bay", "Buccaneers", "Bucs"]],
  ["ten", "Tennessee Titans", "TEN", ["Tennessee", "Titans"]],
  ["was", "Washington Commanders", "WAS", ["Washington", "Commanders"]],
];

function normalize(value: string): string {
  return value.toLocaleLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function slugify(value: string): string {
  return normalize(value).replaceAll(" ", "-") || "team";
}

function fallbackAbbreviation(value: string): string {
  const words = value.replace(/[^A-Za-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);
  if (words.length > 1) return words.map((word) => word[0]).join("").slice(0, 4).toUpperCase();
  return (words[0] ?? "TEAM").slice(0, 4).toUpperCase();
}

const generatedNflTeams = (nflTeamsData.teams as GeneratedTeam[]).map((team) => ({ ...team, league: "nfl" as const }));

export const NFL_TEAM_MANIFEST: TeamIdentity[] = nflTeamSeeds.map(([id, displayName, abbreviation, aliases]) => {
  const generated = generatedNflTeams.find((team) => normalize(team.abbreviation) === normalize(abbreviation));
  return {
    id,
    league: "nfl",
    displayName,
    abbreviation,
    aliases: [...new Set([...aliases, ...(generated?.aliases ?? [])])],
    logoPath: generated?.logoPath,
    conference: generated?.conference,
    color: generated?.color,
    alternateColor: generated?.alternateColor,
  };
});

export const CFB_TEAM_MANIFEST: TeamIdentity[] = (cfbTeamsData.teams as GeneratedTeam[]).map((team) => ({
  ...team,
  league: "cfb",
  abbreviation: team.abbreviation || fallbackAbbreviation(team.displayName),
}));

const teamManifests: Record<FootballLeague, TeamIdentity[]> = {
  nfl: NFL_TEAM_MANIFEST,
  cfb: CFB_TEAM_MANIFEST,
};

export function getTeamIdentity(name: string, league: FootballLeague): TeamIdentity {
  const needle = normalize(name);
  const match = teamManifests[league].find((team) =>
    [team.id, team.displayName, team.abbreviation, team.mascot ?? "", ...team.aliases]
      .some((value) => normalize(value) === needle),
  );
  if (match) return match;

  const id = slugify(name);
  return {
    id,
    league,
    displayName: name,
    abbreviation: fallbackAbbreviation(name),
    aliases: [name],
  };
}

export function getSlateTeams(games: ExpectedPointsPick[], league: FootballLeague): TeamIdentity[] {
  const teams = new Map<string, TeamIdentity>();
  for (const game of games) {
    for (const name of [game.away_team, game.home_team]) {
      const team = getTeamIdentity(name, league);
      teams.set(team.id, team);
    }
  }
  return [...teams.values()].sort((a, b) => a.displayName.localeCompare(b.displayName));
}

export function teamSearchText(team: TeamIdentity): string {
  return normalize([
    team.displayName,
    team.abbreviation,
    team.mascot,
    team.conference,
    ...team.aliases,
  ].filter(Boolean).join(" "));
}

export function normalizedSearch(value: string): string {
  return normalize(value);
}
