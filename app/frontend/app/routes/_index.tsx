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

  return (
    <table>
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column}>{column}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {columns.map((column) => (
              <td key={column}>{row[column]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

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
