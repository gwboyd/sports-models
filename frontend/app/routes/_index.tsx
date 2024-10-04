import type { MetaFunction } from "@vercel/remix";
import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import type { Matchup } from "~/types/apiData";
import { SpreadTable } from "~/components/tables/SpreadTable";
import { TotalTable } from "~/components/tables/TotalTable";
import { displaySpread } from "~/utils";

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
    <div className="overflow-y-auto flex flex-col gap-4 p-12">
      <h2 className="text-gray-300 text-2xl font-bold mb-4">
        {data[0].season}, Week {data[0].week}
      </h2>
      <SpreadTable data={data} />
      <TotalTable data={data} />
      <div className="border border-gray-700 bg-gray-800 rounded p-3 flex flex-col gap-2">
        <strong className="text-gray-300">Spread Plays</strong>
        <div>
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
      </div>
      <div className="border border-gray-700 bg-gray-800 rounded p-3 flex flex-col gap-2">
        <strong className="text-gray-300">Total Plays</strong>
        <div>
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
    </div>
  );
}
