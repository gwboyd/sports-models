// Cache responses for 5 minutes
const CACHE_TTL = 5 * 60 * 1000;

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

export function createCache<T>() {
  let cache: CacheEntry<T> | null = null;

  return {
    get: () => cache?.data,
    set: (data: T) => {
      cache = { data, timestamp: Date.now() };
    },
    isValid: () => {
      return cache && Date.now() - cache.timestamp < CACHE_TTL;
    },
  };
}

export async function fetchWithCache<T>(
  endpoint: string,
  cache: ReturnType<typeof createCache<T>>
): Promise<T> {
  if (cache.isValid()) {
    const data = cache.get();
    if (data) return data;
  }

  const response = await fetch(`${process.env.ENDPOINT}${endpoint}`, {
    headers: {
      Authorization: process.env.AUTHORIZATION_TOKEN ?? "",
    },
  });

  if (!response.ok) {
    throw new Response("Failed to fetch data", {
      status: response.status,
    });
  }

  const data = await response.json();
  cache.set(data);
  return data;
}
