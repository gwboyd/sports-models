import { Redis } from "@upstash/redis";

// cache responses for 5 minutes
const CACHE_TTL = 5 * 60;

const hasRedisEnv =
  Boolean(process.env.UPSTASH_REDIS_REST_URL) &&
  Boolean(process.env.UPSTASH_REDIS_REST_TOKEN);

const redis = hasRedisEnv ? Redis.fromEnv() : null;

export async function fetchWithCache<T>(key: string): Promise<T> {
  if (redis) {
    const cached = await redis.get<T>(key);
    if (cached) return cached;
  }

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

  if (redis) {
    await redis.setex(key, CACHE_TTL, data);
  }

  return data;
}

export const NFL_PICKS_KEY = "nfl-picks";
export const NFL_PICK_RESULTS_KEY = "nfl-pick-results";
export const NBA_FIRST_BASKET_PICKS_KEY = "nba-first-basket-picks";
