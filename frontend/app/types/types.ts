import type { components } from "./generated/openapi";

export type NFLPick = components["schemas"]["PickResponse"];
export type OverallNFLResults =
  components["schemas"]["PickResultsResponse"]["data"];
