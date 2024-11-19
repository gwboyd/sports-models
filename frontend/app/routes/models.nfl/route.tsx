import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import type { MetaFunction } from "@vercel/remix";

import {
  NFL_PICKS_KEY,
  NFL_PICK_RESULTS_KEY,
  fetchWithCache,
} from "~/api/data-utils";
import { Card } from "~/components/Card";
import { BodyText, SectionTitle } from "~/components/Typography";
import { SpreadTable } from "~/routes/models.nfl/SpreadTable";
import { TotalTable } from "~/routes/models.nfl/TotalTable";
import type { NFLPick, NFLResultsResponse } from "~/types/types";
import { convertDateTime, displaySpread } from "~/utils";

export const meta: MetaFunction = () => {
  return [
    { title: "Sport Models" },
    { name: "description", content: "Webpage for Sport Models" },
  ];
};

export const loader = async () => {
  const [picksData, resultsData] = await Promise.all([
    fetchWithCache<NFLPick[]>(NFL_PICKS_KEY),
    fetchWithCache<NFLResultsResponse>(NFL_PICK_RESULTS_KEY),
  ]);

  const picks = picksData.sort(
    (a, b) =>
      convertDateTime(a.date_time).getTime() -
      convertDateTime(b.date_time).getTime()
  );

  const overallResults = resultsData.data;

  const spreadLocks = picks
    .filter((game) => game.spread_lock)
    .sort((a, b) => b.spread_win_prob - a.spread_win_prob);

  const totalLocks = picks
    .filter((game) => game.total_lock)
    .sort((a, b) => b.total_win_prob - a.total_win_prob);

  return json({ data: picks, spreadLocks, totalLocks, overallResults });
};

export default function NFLModel() {
  const { data, spreadLocks, totalLocks, overallResults } =
    useLoaderData<typeof loader>();

  if (!Array.isArray(data) || data.length === 0) {
    return <div>No data available</div>;
  }

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-36 lg:pb-24 lg:px-12">
      <h2 className="text-gray-300 text-2xl font-bold mb-4">
        {data[0].season}, Week {data[0].week}
      </h2>
      <Card title="Spread Plays">
        {spreadLocks.map((game: NFLPick) => (
          <BodyText key={game.spread_win_prob}>
            {`${game.home_team}/${game.away_team}: ${
              game.spread_play
            } ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_line
            )}
            (model ${game.spread_play} ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_pred
            )}, ${game.spread_win_prob.toFixed(2)}%)`}
          </BodyText>
        ))}
      </Card>
      <Card title="Total Plays">
        {totalLocks.map((game: NFLPick) => (
          <BodyText key={game.total_win_prob}>
            {`${game.home_team}/${game.away_team}: ${game.total_play} ${
              game.total_line
            } (model ${game.total_pred.toFixed(
              2
            )}, ${game.total_win_prob.toFixed(2)}%)`}
          </BodyText>
        ))}
      </Card>
      <SectionTitle>Spreads</SectionTitle>
      <SpreadTable data={data} />
      <SectionTitle>Totals</SectionTitle>
      <TotalTable data={data} />
      <Card title="Overall Results 2024">
        <BodyText>{`Total predicted games: ${overallResults.predicted_games}`}</BodyText>
        <BodyText>{`Spread play record: ${overallResults.spread_lock_wins} - ${
          overallResults.spread_lock_losses
        } - ${
          overallResults.spread_lock_pushes
        } (${overallResults.spread_lock_win_pct.toFixed(2)}%)`}</BodyText>
        <BodyText>{`Total play record: ${overallResults.total_lock_wins} - ${
          overallResults.total_lock_losses
        } - ${
          overallResults.total_lock_pushes
        } (${overallResults.total_lock_win_pct.toFixed(2)}%)`}</BodyText>
        <BodyText>{`Overall spread record: ${overallResults.spread_wins} - ${
          overallResults.spread_losses
        } - ${
          overallResults.spread_pushes
        } (${overallResults.spread_win_pct.toFixed(2)}%)`}</BodyText>
        <BodyText>
          {`Overall total record: ${overallResults.total_wins} - ${
            overallResults.total_losses
          } - ${
            overallResults.total_pushes
          } (${overallResults.total_win_pct.toFixed(2)}%)`}
        </BodyText>
      </Card>
      <SectionTitle>Assorted Charts</SectionTitle>
      <div className="grid max-w-[768px] gap-4 grid-cols-1 md:grid-cols-2">
        <img
          className="col-span-1 w-full md:col-span-2"
          src="https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/power_rankings.png"
          alt="NFL Power Rankings"
        />
        <img
          className="w-full"
          src="https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/offensive_epa.png"
          alt="NFL Offensive EPA"
        />
        <img
          className="w-full"
          src="https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/defensive_epa.png"
          alt="NFL Defensive EPA"
        />
      </div>
    </div>
  );
}
