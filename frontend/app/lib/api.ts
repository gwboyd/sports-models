import "server-only";

export const NFL_PICKS_KEY = "nfl-picks";
export const NFL_PICK_RESULTS_KEY = "nfl-pick-results";
export const NBA_FIRST_BASKET_PICKS_KEY = "nba-first-basket-picks";

export class UpstreamApiError extends Error {
  constructor(
    public readonly path: string,
    public readonly status: number,
    public readonly statusText: string,
  ) {
    super(`Failed to fetch ${path}: ${status}`);
    this.name = "UpstreamApiError";
  }
}

function getEndpoint(): string {
  const endpoint = process.env.ENDPOINT?.replace(/\/+$/, "");
  if (!endpoint) {
    throw new Error("ENDPOINT must be configured for server-side model data requests.");
  }
  return endpoint;
}

export async function fetchApi<T>(path: string, revalidateSeconds = 300): Promise<T> {
  const response = await fetch(`${getEndpoint()}/${path}`, {
    headers: { Authorization: process.env.AUTHORIZATION_TOKEN ?? "" },
    next: { revalidate: revalidateSeconds },
  });

  if (!response.ok) {
    console.error("Model API request failed", { path, status: response.status });
    throw new UpstreamApiError(path, response.status, response.statusText);
  }

  return (await response.json()) as T;
}
