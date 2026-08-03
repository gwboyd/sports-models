import { createServer } from "node:http";

const pick = {
  season: 2024,
  week: "1",
  home_team: "Home",
  away_team: "Away",
  home_score_pred: 24,
  away_score_pred: 21,
  spread_pred: 3,
  spread_line: 2.5,
  spread_play: "Home",
  spread_win_prob: 61,
  spread_lock: 1,
  total_pred: 45,
  total_line: 44.5,
  total_play: "Over",
  total_win_prob: 58,
  total_lock: 1,
  game_id: "test-game",
  year_week: "2024-1",
  date_time: "2024-09-08-13:00",
  write_time: "2024-09-01T00:00:00Z",
};

const results = {
  data: {
    predicted_games: 1, spread_wins: 1, spread_losses: 0, spread_pushes: 0, spread_win_pct: 100,
    spread_lock_predictions: 1, spread_lock_wins: 1, spread_lock_losses: 0, spread_lock_pushes: 0, spread_lock_win_pct: 100,
    total_wins: 1, total_losses: 0, total_pushes: 0, total_win_pct: 100,
    total_lock_predictions: 1, total_lock_wins: 1, total_lock_losses: 0, total_lock_pushes: 0, total_lock_win_pct: 100,
  },
  games: [{
    season: 2024, week: "1", home_team: "Home", away_team: "Away", home_score: 24, away_score: 20,
    home_score_pred: 24, away_score_pred: 21, spread_pred: 3, spread_line: 2.5, true_spread: -4,
    spread_play: "Home", spread_win_prob: 61, spread_lock: 1, correct_spread_play: "Home", spread_win: 1,
    total_pred: 45, total_line: 44.5, true_total: 44, total_play: "Over", total_win_prob: 58,
    total_lock: 1, correct_total_play: "Under", total_win: 0, year_week: "2024-1", game_id: "test-game",
    date_time: "2024-09-08-13:00",
  }],
};

const cfbPick = {
  ...pick,
  season: 2026,
  year_week: "2026_1",
  game_id: "cfb-test-game",
  home_conference: "SEC",
  away_conference: "Big Ten",
};

const nbaPicks = [{
  date: "2024-01-01", player_name: "Test Player", team: "TEST", fb_model_prob: 0.2,
  fb_model_odds: 100, odds: 150, sportsbook: "Test Sportsbook", units: 1,
}];

const payloads = {
  "/nfl-picks": [pick],
  "/nfl-pick-results": results,
  "/cfb-picks": [cfbPick],
  "/nba-first-basket-picks": nbaPicks,
};

createServer((request, response) => {
  if (request.url === "/health") {
    response.writeHead(200).end("ok");
    return;
  }
  if (request.headers.authorization !== "test-token") {
    response.writeHead(401).end();
    return;
  }
  const payload = payloads[request.url];
  if (!payload) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(payload));
}).listen(4010, "127.0.0.1");
