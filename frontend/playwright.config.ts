import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  use: { baseURL: "http://127.0.0.1:5173" },
  webServer: [
    {
      command: "node tests/fixtures/mock-api.mjs",
      url: "http://127.0.0.1:4010/health",
      reuseExistingServer: false,
    },
    {
      command: "ENDPOINT=http://127.0.0.1:4010 AUTHORIZATION_TOKEN=test-token npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
    },
  ],
});
