import type { components } from "./generated/openapi";

export type NFLPick = components["schemas"]["PickResponse"];
export type NFLResultsResponse = components["schemas"]["PickResultsResponse"];
export type OverallNFLResults = NFLResultsResponse["data"];
