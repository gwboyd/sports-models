import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";

const README_PATH = path.resolve(
  process.cwd(),
  "../src/sports/football/nfl/expected_points/README.md",
);

export function getModelInfoMarkdown() {
  return readFile(README_PATH, "utf8");
}
