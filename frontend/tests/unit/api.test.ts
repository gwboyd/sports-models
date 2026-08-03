import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchApi, UpstreamApiError } from "@/app/lib/api";

describe("fetchApi", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("uses the server endpoint, authorization header, and revalidation value", async () => {
    vi.stubEnv("ENDPOINT", "https://api.example.test/");
    vi.stubEnv("AUTHORIZATION_TOKEN", "test-token");
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchApi<{ ok: boolean }>("nfl-picks", 300)).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledWith("https://api.example.test/nfl-picks", {
      headers: { Authorization: "test-token" },
      next: { revalidate: 300 },
    });
  });

  it("throws a typed error for unsuccessful upstream responses", async () => {
    vi.stubEnv("ENDPOINT", "https://api.example.test");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503, statusText: "Unavailable" })));

    await expect(fetchApi("nfl-picks")).rejects.toBeInstanceOf(UpstreamApiError);
  });
});
