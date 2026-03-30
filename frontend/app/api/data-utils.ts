export async function fetchWithCache<T>(key: string): Promise<T> {
  const response = await fetch(`${process.env.ENDPOINT}/${key}`, {
    headers: {
      Authorization: process.env.AUTHORIZATION_TOKEN ?? "",
    },
  });

  if (!response.ok) {
    throw new Response("Failed to fetch data", {
      status: response.status,
    });
  }

  const data = (await response.json()) as Awaited<T>;
  return data;
}

export const NFL_PICKS_KEY = "nfl-picks";
export const NFL_PICK_RESULTS_KEY = "nfl-pick-results";
export const NBA_FIRST_BASKET_PICKS_KEY = "nba-first-basket-picks";
