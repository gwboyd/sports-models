import { Redis } from "@upstash/redis";

// cache responses for 5 minutes
const CACHE_TTL = 5 * 60;

const redis = Redis.fromEnv();

export async function fetchWithCache<T>(key: string): Promise<T> {
  let data = await redis.get<T>(key);
  if (data) return data;
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

  data = (await response.json()) as Awaited<T>;

  await redis.setex(key, CACHE_TTL, data);
  return data;
}

export const NFL_PICKS_KEY = "nfl-picks";
export const NFL_PICK_RESULTS_KEY = "nfl-pick-results";
export const NBA_FIRST_BASKET_PICKS_KEY = "nba-first-basket-picks";
