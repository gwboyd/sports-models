import { expect, test } from "@playwright/test";

test("root redirects to the mobile-first NFL games page", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/models\/nfl$/);
  await expect(page.getByRole("heading", { name: "NFL predictions" })).toBeVisible();
  await expect(page.getByRole("link", { name: "NFL" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByText("Spread lock", { exact: true })).toBeVisible();
  await expect(page.getByText("Total lock", { exact: true })).toBeVisible();
});

test("CFB renders each game once and offers a conference filter", async ({ page }) => {
  await page.goto("/models/cfb");
  await expect(page.getByRole("heading", { name: "College Football predictions" })).toBeVisible();
  await expect(page.getByLabel("Conference")).toBeVisible();
  await expect(page.locator("[data-game-id='game-cfb-test-game']:visible")).toHaveCount(1);
});

test("game search finds the current slate", async ({ page }) => {
  await page.goto("/models/nfl");
  await page.getByRole("button", { name: "Search games" }).click();
  await page.getByPlaceholder("Try 49ers, SF, SEC…").fill("Home");
  await expect(page.getByRole("dialog")).toContainText("HOME");
});

test("favorites are saved across reloads", async ({ page }) => {
  await page.goto("/models/nfl");
  await page.getByRole("button", { name: "Edit teams" }).click();
  await page.getByRole("checkbox", { name: "Favorite Home" }).check();
  await page.getByRole("button", { name: "Close" }).click();
  await page.reload();
  const favorites = page.locator("[aria-labelledby='favorites-title']");
  const favoriteCard = favorites.locator("article");
  await expect(page.getByRole("heading", { name: "Favorites" })).toBeVisible();
  await expect(favoriteCard).toContainText("Model score");
  await expect(favoriteCard).toContainText("Spread lock");
  await expect(favoriteCard).toContainText("Total lock");
  await expect(favoriteCard).toHaveClass(/border-\[var\(--lock-border\)\]/);
});

test("results support shareable season selection and CFB empty state", async ({ page }) => {
  await page.goto("/models/nfl/results?season=2024");
  await expect(page.getByRole("heading", { name: "NFL results" })).toBeVisible();
  await expect(page.getByLabel("Season")).toHaveValue("2024");
  await expect(page.getByText("2024 summary")).toBeVisible();

  await page.goto("/models/cfb/results");
  await expect(page.getByText("Results will appear here after games have been completed and graded by the model workflow.")).toBeVisible();
});

test("legacy model information redirects into the NFL section", async ({ page }) => {
  await page.goto("/models/info");
  await expect(page).toHaveURL(/\/models\/nfl\/how-it-works$/);
  await expect(page.getByRole("heading", { name: "NFL Expected Points Model" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feature importance" })).toBeVisible();
  await expect(page.getByAltText("Model feature importances")).toBeVisible();
});

test("NBA route preserves the bankroll workflow", async ({ page }) => {
  await page.goto("/models/nba?bankroll=500");
  await expect(page.locator("input")).toHaveValue("500");
  await page.locator("input").fill("800");
  await expect(page).toHaveURL(/\/models\/nba\?bankroll=800$/);
});

test("primary pages do not create horizontal document overflow on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ["/models/nfl", "/models/cfb", "/models/nfl/results?season=2024", "/models/nfl/how-it-works"]) {
    await page.goto(route);
    const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
    expect(widths.scroll).toBeLessThanOrEqual(widths.client);
  }
});
