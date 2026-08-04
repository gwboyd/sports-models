import type { components } from "./generated/openapi";

export type ExpectedPointsPick = components["schemas"]["PickResponse"];
export type NFLPick = ExpectedPointsPick;
export type NFLResultsResponse = components["schemas"]["PickResultsResponse"];
export type OverallNFLResults = NFLResultsResponse["data"];
export type CFBPick = components["schemas"]["CFBPickResponse"];
export type CFBResultsResponse = components["schemas"]["PickResultsResponse"];
export type GameResult = components["schemas"]["GameResult"];
export type FootballLeague = "nfl" | "cfb";

export type NBAFirstBasketPick = components["schemas"]["NBAFirstBasketPick"];
