import type { MetaFunction } from "@vercel/remix";
import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";

import { SpreadTable } from "~/components/tables/SpreadTable";
import { TotalTable } from "~/components/tables/TotalTable";
import { convertDateTime, displaySpread } from "~/utils";
import type { NFLPick, OverallNFLResults } from "~/types/types";
import { createCache, fetchWithCache } from "~/api/cache";

export const meta: MetaFunction = () => {
  return [
    { title: "Sport Models" },
    { name: "description", content: "Webpage for Sport Models" },
  ];
};

const picksCache = createCache<NFLPick[]>();
const resultsCache = createCache<{ data: OverallNFLResults }>();

export const loader = async () => {
  const [picksData, resultsData] = await Promise.all([
    fetchWithCache("/nfl-picks", picksCache),
    fetchWithCache("/nfl-pick-results", resultsCache),
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
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-24 lg:p-12">
      <h2 className="text-gray-300 text-2xl font-bold mb-4">
        {data[0].season}, Week {data[0].week}
      </h2>
      <Card title="Spread Plays">
        {spreadLocks.map((game: NFLPick) => (
          <p key={game.spread_win_prob} className="text-gray-400 mb-1">
            {`${game.home_team}/${game.away_team}: ${
              game.spread_play
            } ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_line
            )}
            (model ${game.spread_play} ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_pred
            )}, ${game.spread_win_prob.toFixed(2)}%)`}
          </p>
        ))}
      </Card>
      <Card title="Total Plays">
        {totalLocks.map((game: NFLPick) => (
          <p key={game.total_win_prob} className="text-gray-400 mb-1">
            {`${game.home_team}/${game.away_team}: ${game.total_play} ${
              game.total_line
            } (model ${game.total_pred.toFixed(
              2
            )}, ${game.total_win_prob.toFixed(2)}%)`}
          </p>
        ))}
      </Card>
      <h4 className="text-gray-300 text-xl font-bold">Spreads</h4>
      <SpreadTable data={data} />
      <h4 className="text-gray-300 text-xl font-bold">Totals</h4>
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
    </div>
  );
}

const Card = ({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) => (
  <div className="border border-gray-700 bg-gray-800 rounded p-3 flex flex-col gap-2">
    {title && <strong className="text-gray-300">{title}</strong>}
    <div>{children}</div>
  </div>
);

const BodyText = ({ children }: { children: React.ReactNode }) => (
  <p className="text-gray-400 mb-1">{children}</p>
);
