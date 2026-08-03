import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";

const README_PATH = path.resolve(
  process.cwd(),
  "content/nfl-how-it-works.md",
);

export function getModelInfoMarkdown() {
  return readFile(README_PATH, "utf8");
}
