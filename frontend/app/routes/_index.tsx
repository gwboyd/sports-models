import type { MetaFunction } from "@vercel/remix";
import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import type { Matchup } from "~/types/apiData";

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

  const data: Matchup[] = await response.json();
  const spreadLocks = data
    .filter((game: Matchup) => game["spread_lock"])
    .sort((a, b) => b["spread_win_prob"] - a["spread_win_prob"]);
  const totalLocks = data
    .filter((game) => game["total_lock"])
    .sort((a, b) => b["total_win_prob"] - a["total_win_prob"]);

  return json({ data, spreadLocks, totalLocks });
};

export default function Index() {
  const { data, spreadLocks, totalLocks } = useLoaderData<typeof loader>();

  if (!Array.isArray(data) || data.length === 0) {
    return <div>No data available</div>;
  }

  return (
    <div className="bg-gray-900 p-4 overflow-auto h-screen">
      <h2 className="text-gray-300 text-2xl font-bold mb-4">
        {data[0].season}, Week {data[0].week}
      </h2>
      <div className="overflow-x-auto">
        <table className="table-auto border-collapse border border-gray-700 w-full text-center">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="border border-gray-700 p-2 bg-gray-800 text-white"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data
              .sort((a, b) => a["order"] - b["order"])
              .map((row, rowIndex) => (
                <tr key={rowIndex} className="bg-gray-800">
                  {columns.map((column: keyof Matchup) => (
                    <td
                      key={column}
                      className="border border-gray-700 p-2 text-gray-300"
                    >
                      {column.includes("spread_pred") ||
                      column.includes("spread_line")
                        ? displaySpread(row[column] as number)
                        : column.includes("pred")
                        ? (row[column] as number).toFixed(2)
                        : column.includes("prob")
                        ? `${(row[column] as number).toFixed(2)}%`
                        : row[column]}
                    </td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 bg-gray-800 p-3 rounded-md">
        <strong className="text-gray-300 block mb-2">Spread Plays</strong>
        {spreadLocks.map((game) => (
          <p key={game["spread_win_prob"]} className="text-gray-400 mb-1">
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
      <div className="mt-4 bg-gray-800 p-3 rounded-md">
        <strong className="text-gray-300 block mb-2">Total Plays</strong>
        {totalLocks.map((game) => (
          <p key={game["total_win_prob"]} className="text-gray-400 mb-1">
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
  spread > 0 ? `+${spread.toFixed(1)}` : spread.toFixed(1);

const columns: (keyof Matchup)[] = [
  "home_team",
  "away_team",
  "home_score_pred",
  "away_score_pred",
  "spread_pred",
  "spread_line",
  "spread_play",
  "spread_win_prob",
  "total_pred",
  "total_line",
  "total_play",
  "total_win_prob",
];
