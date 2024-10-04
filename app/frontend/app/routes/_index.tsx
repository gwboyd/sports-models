import type { MetaFunction } from "@vercel/remix";

import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";

export const meta: MetaFunction = () => {
  return [
    { title: "Sport Models" },
    { name: "description", content: "Webpage for Sport Models" },
  ];
};

export const loader = async () => {
  const response = await fetch(
    "https://j1neh7th0m.execute-api.us-east-1.amazonaws.com/nfl-picks",
    {
      headers: {
        Authorization: process.env.AUTHORIZATION_TOKEN ?? "",
      },
    }
  );

  if (!response.ok) {
    throw new Response("Failed to fetch data", { status: response.status });
  }

  const data = await response.json();
  return json(data);
};

export default function Index() {
  const data = useLoaderData();

  if (!Array.isArray(data) || data.length === 0) {
    return <div>No data available</div>;
  }
  const spreadLocks = data.filter((game) => game["spread_lock"]);
  const totalLocks = data.filter((game) => game["total_lock"]);

  return (
    <div className="bg-amber-50 p-4">
      <table className="table-auto border-collapse border border-red-800 w-full text-center">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                className="border border-red-800 p-2 bg-green-800 text-white"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex} className="bg-amber-100">
              {columns.map((column) => (
                <td key={column} className="border border-red-800 p-2">
                  {column.includes("spread_pred") ||
                  column.includes("spread_line")
                    ? displaySpread(row[column])
                    : column.includes("pred") || column.includes("prob")
                    ? row[column].toFixed(2)
                    : row[column]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 bg-red-100 p-3 rounded-md">
        <strong className="text-red-900 block mb-2">Spread Plays</strong>
        {spreadLocks.map((game) => (
          <p key={game["spread_win_prob"]} className="text-green-800 mb-1">
            {`${game["home_team"]}/${game["away_team"]}: ${displaySpread(
              game["spread_line"]
            )}
            (model ${game["spread_play"]} ${displaySpread(
              game["spread_pred"]
            )}, ${game["spread_win_prob"].toFixed(2)}% win
            probability)`}
          </p>
        ))}
      </div>
      <div className="mt-4 bg-green-100 p-3 rounded-md">
        <strong className="text-red-900 block mb-2">Total Plays</strong>
        {totalLocks.map((game) => (
          <p key={game["total_win_prob"]} className="text-green-800 mb-1">
            {`${game["home_team"]}/${game["away_team"]}: ${
              game["total_play"]
            } ${game["total_line"]} (model ${game["total_pred"].toFixed(
              2
            )}, ${game["total_win_prob"].toFixed(2)}% win probability)`}
          </p>
        ))}
      </div>
    </div>
  );
}

const displaySpread = (spread: number) =>
  spread > 0 ? `+${spread.toFixed(2)}` : spread.toFixed(2);

const columns = [
  "season",
  "week",
  "home_team",
  "away_team",
  "home_score_pred",
  "away_score_pred",
  "spread_pred",
  "spread_line",
  "spread_play",
  "spread_win_prob",
  "spread_lock",
  "total_pred",
  "total_line",
  "total_play",
  "total_win_prob",
  "total_lock",
];
