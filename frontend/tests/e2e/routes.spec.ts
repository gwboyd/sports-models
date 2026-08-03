import { expect, test } from "@playwright/test";

test("root redirects to the NFL model page", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/models\/nfl$/);
  await expect(page.getByRole("heading", { name: "2024, Week 1" })).toBeVisible();
  await expect(page.getByRole("link", { name: "NFL" })).toHaveAttribute("aria-current", "page");
});

test("models index redirects to the NFL model page", async ({ page }) => {
  await page.goto("/models");
  await expect(page).toHaveURL(/\/models\/nfl$/);
});

test("CFB groups a cross-conference game into both conference sections", async ({ page }) => {
  await page.goto("/models/cfb");

  await expect(page.getByRole("link", { name: "CFB" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { name: "2026, Week 1" })).toBeVisible();
  await expect(page.locator("section h4")).toHaveText(["SEC", "BIG 10"]);
  await expect(page.getByRole("cell", { name: "Home", exact: true })).toHaveCount(4);
});

test("NBA route preserves the bankroll query string", async ({ page }) => {
  await page.goto("/models/nba?bankroll=500");
  await expect(page.locator("input")).toHaveValue("500");

  await page.locator("input").fill("800");
  await expect(page).toHaveURL(/\/models\/nba\?bankroll=800$/);
  await expect(page.locator("input")).toHaveValue("800");
});

test("model information renders from the canonical README", async ({ page }) => {
  const pageErrors: Error[] = [];
  page.on("pageerror", (error) => pageErrors.push(error));

  await page.goto("/models/info");
  await expect(page.getByRole("heading", { name: "NFL Expected Points Model" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Info" })).toHaveAttribute("aria-current", "page");
  expect(pageErrors).toEqual([]);
});
