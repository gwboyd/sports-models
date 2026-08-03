import { connection } from "next/server";
import { BodyText, SectionTitle } from "@/app/components/Typography";
import { Card } from "@/app/components/Card";
import { CFB_PICKS_KEY, fetchApi } from "@/app/lib/api";
import { displaySpread } from "@/app/lib/formatting";
import { prepareCfbData } from "@/app/lib/model-data";
import type { CFBPick } from "@/app/types/types";
import { SpreadTable } from "../_components/expected-points/SpreadTable";
import { TotalTable } from "../_components/expected-points/TotalTable";

export default async function CfbModelPage() {
  await connection();
  const { data, spreadLocks, totalLocks, conferenceGroups } = prepareCfbData(
    await fetchApi<CFBPick[]>(CFB_PICKS_KEY),
  );

  if (data.length === 0) return <div>No data available</div>;

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-36 lg:pb-24 lg:px-12">
      <h2 className="text-gray-300 text-2xl font-bold mb-4">
        {data[0].season}, Week {data[0].week}
      </h2>
      <Card title="Spread Plays">
        {spreadLocks.length === 0 ? (
          <BodyText>No spread plays this week.</BodyText>
        ) : (
          spreadLocks.map((game) => (
            <BodyText key={game.game_id}>{`${game.home_team}/${game.away_team}: ${game.spread_play} ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_line,
            )} (model ${game.spread_play} ${displaySpread(
              (game.spread_play === game.away_team ? -1 : 1) * game.spread_pred,
            )}, ${game.spread_win_prob.toFixed(2)}%)`}</BodyText>
          ))
        )}
      </Card>
      <Card title="Total Plays">
        {totalLocks.length === 0 ? (
          <BodyText>No total plays this week.</BodyText>
        ) : (
          totalLocks.map((game) => (
            <BodyText key={game.game_id}>{`${game.home_team}/${game.away_team}: ${game.total_play} ${game.total_line} (model ${game.total_pred.toFixed(2)}, ${game.total_win_prob.toFixed(2)}%)`}</BodyText>
          ))
        )}
      </Card>

      {conferenceGroups.map(({ conference, games }) => (
        <section
          key={conference}
          className="flex flex-col gap-3"
          aria-labelledby={`conference-${conference.replaceAll(" ", "-")}`}
        >
          <SectionTitle id={`conference-${conference.replaceAll(" ", "-")}`}>
            {conference}
          </SectionTitle>
          <h5 className="text-gray-300 text-base font-semibold">Spreads</h5>
          <SpreadTable compact data={games} />
          <h5 className="text-gray-300 text-base font-semibold mt-2">Totals</h5>
          <TotalTable compact data={games} />
        </section>
      ))}
    </div>
  );
}
