"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { FootballLeague } from "@/app/types/types";

export const FAVORITES_STORAGE_KEY = "sports-models:favorites:v1";
const FAVORITES_CHANGE_EVENT = "sports-models:favorites-change";
const EMPTY_STORE = JSON.stringify({ nfl: [], cfb: [] });

export type FavoriteStore = Record<FootballLeague, string[]>;

export function parseFavoriteStore(value: string | null): FavoriteStore {
  if (!value) return { nfl: [], cfb: [] };
  try {
    const parsed = JSON.parse(value) as Partial<FavoriteStore>;
    return {
      nfl: Array.isArray(parsed.nfl) ? [...new Set(parsed.nfl.filter((item): item is string => typeof item === "string"))] : [],
      cfb: Array.isArray(parsed.cfb) ? [...new Set(parsed.cfb.filter((item): item is string => typeof item === "string"))] : [],
    };
  } catch {
    return { nfl: [], cfb: [] };
  }
}

export function useFavoriteTeams(league: FootballLeague) {
  const serialized = useSyncExternalStore(
    (onStoreChange) => {
      window.addEventListener("storage", onStoreChange);
      window.addEventListener(FAVORITES_CHANGE_EVENT, onStoreChange);
      return () => {
        window.removeEventListener("storage", onStoreChange);
        window.removeEventListener(FAVORITES_CHANGE_EVENT, onStoreChange);
      };
    },
    () => window.localStorage.getItem(FAVORITES_STORAGE_KEY) ?? EMPTY_STORE,
    () => EMPTY_STORE,
  );
  const store = parseFavoriteStore(serialized);

  const update = useCallback((teamId: string, selected: boolean) => {
    const current = parseFavoriteStore(window.localStorage.getItem(FAVORITES_STORAGE_KEY));
    const nextLeague = selected
      ? [...new Set([...current[league], teamId])]
      : current[league].filter((id) => id !== teamId);
    window.localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify({ ...current, [league]: nextLeague }));
    window.dispatchEvent(new Event(FAVORITES_CHANGE_EVENT));
  }, [league]);

  return { favoriteIds: store[league], ready: true, update };
}
