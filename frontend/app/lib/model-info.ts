import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";

export function getModelInfoMarkdown(league: "nfl" | "cfb") {
  return readFile(path.resolve(process.cwd(), `content/${league}-how-it-works.md`), "utf8");
}
