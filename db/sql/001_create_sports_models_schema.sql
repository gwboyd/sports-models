-- Supabase setup for the sports_models schema.
-- Run this once against a fresh or compatible database state.

create schema if not exists sports_models;

create table if not exists sports_models.model_releases (
    model_key text not null,
    version text not null,
    major_version integer not null,
    minor_version integer not null,
    title text not null,
    public_summary text not null,
    changes_md text not null,
    evaluation_md text,
    internal_notes_md text,
    source_git_sha text,
    deployed_at timestamptz not null,
    first_pick_at timestamptz,
    primary key (model_key, version),
    unique (model_key, major_version, minor_version),
    check (model_key in ('nfl_expected_points', 'cfb_expected_points')),
    check (major_version >= 1),
    check (minor_version >= 0),
    check (version = major_version::text || '.' || minor_version::text)
);

create table if not exists sports_models.nfl_expected_points_picks (
    season integer not null,
    week text not null,
    year_week text not null,
    game_id text not null,
    home_team text not null,
    away_team text not null,
    home_score_pred double precision not null,
    away_score_pred double precision not null,
    spread_pred double precision not null,
    spread_line double precision not null,
    spread_play text not null,
    spread_win_prob double precision not null,
    spread_lock integer not null,
    total_pred double precision not null,
    total_line double precision not null,
    total_play text not null,
    total_win_prob double precision not null,
    total_lock integer not null,
    date_time text not null,
    model_version text,
    write_time timestamptz not null,
    primary key (year_week, game_id)
);

create index if not exists nfl_expected_points_picks_game_id_idx
    on sports_models.nfl_expected_points_picks (game_id);

create index if not exists nfl_expected_points_picks_season_week_idx
    on sports_models.nfl_expected_points_picks (season, week);

create table if not exists sports_models.nfl_expected_points_results (
    season integer not null,
    week text not null,
    year_week text not null,
    game_id text not null,
    home_team text not null,
    away_team text not null,
    home_score integer not null,
    away_score integer not null,
    home_score_pred double precision not null,
    away_score_pred double precision not null,
    spread_pred double precision not null,
    spread_line double precision not null,
    true_spread double precision not null,
    spread_play text not null,
    spread_win_prob double precision not null,
    spread_lock integer not null,
    correct_spread_play text,
    spread_win integer,
    total_pred double precision not null,
    total_line double precision not null,
    true_total double precision not null,
    total_play text not null,
    total_win_prob double precision not null,
    total_lock integer not null,
    correct_total_play text,
    total_win integer,
    date_time text not null,
    model_version text,
    primary key (year_week, game_id)
);

create index if not exists nfl_expected_points_results_game_id_idx
    on sports_models.nfl_expected_points_results (game_id);

create index if not exists nfl_expected_points_results_season_week_idx
    on sports_models.nfl_expected_points_results (season, week);

create table if not exists sports_models.nfl_expected_points_pick_updates (
    id bigserial primary key,
    year_week text not null,
    write_time timestamptz not null,
    week text not null,
    season integer not null,
    environment text not null,
    client_name text not null,
    runtime double precision not null,
    model_version text,
    source_git_sha text,
    evaluation_metrics jsonb not null default '{}'::jsonb,
    pick_changes integer not null,
    pick_changes_games jsonb not null default '[]'::jsonb,
    play_changes integer not null,
    play_changes_games jsonb not null default '[]'::jsonb,
    updates_skipped integer not null,
    picks_num integer not null,
    difference_df jsonb not null default '[]'::jsonb,
    picks_df jsonb not null default '[]'::jsonb
);

create unique index if not exists nfl_expected_points_pick_updates_unique_idx
    on sports_models.nfl_expected_points_pick_updates (year_week, write_time, client_name);

create index if not exists nfl_expected_points_pick_updates_year_week_idx
    on sports_models.nfl_expected_points_pick_updates (year_week, write_time desc);

-- Add release-tracking columns before defining views so this setup remains
-- compatible with older NFL tables that predate model versioning.
alter table sports_models.nfl_expected_points_picks
    add column if not exists model_version text;
alter table sports_models.nfl_expected_points_results
    add column if not exists model_version text;
alter table sports_models.nfl_expected_points_pick_updates
    add column if not exists model_version text;
alter table sports_models.nfl_expected_points_pick_updates
    add column if not exists source_git_sha text;
alter table sports_models.nfl_expected_points_pick_updates
    add column if not exists evaluation_metrics jsonb not null default '{}'::jsonb;

create table if not exists sports_models.nba_first_basket_picks (
    pick_date date not null,
    player_name text not null,
    team text not null,
    fb_model_prob double precision not null,
    fb_model_odds double precision not null,
    odds double precision not null,
    sportsbook text not null,
    units double precision not null,
    write_time timestamptz not null default now(),
    primary key (pick_date, player_name)
);

create index if not exists nba_first_basket_picks_team_idx
    on sports_models.nba_first_basket_picks (team);

create or replace view sports_models.nfl_expected_points_latest_picks as
select
    season,
    week,
    year_week,
    game_id,
    home_team,
    away_team,
    home_score_pred,
    away_score_pred,
    spread_pred,
    spread_line,
    spread_play,
    spread_win_prob,
    spread_lock,
    total_pred,
    total_line,
    total_play,
    total_win_prob,
    total_lock,
    date_time,
    write_time,
    model_version
from sports_models.nfl_expected_points_picks
where year_week = (
    select p.year_week
    from sports_models.nfl_expected_points_picks p
    order by p.season desc, cast(p.week as integer) desc
    limit 1
);

create or replace view sports_models.nfl_expected_points_latest_updates as
select distinct on (year_week)
    year_week,
    write_time,
    week,
    season,
    environment,
    client_name,
    runtime,
    pick_changes,
    pick_changes_games,
    play_changes,
    play_changes_games,
    updates_skipped,
    picks_num,
    model_version,
    source_git_sha,
    evaluation_metrics
from sports_models.nfl_expected_points_pick_updates
order by year_week, write_time desc;

create table if not exists sports_models.cfb_expected_points_picks (
    season integer not null,
    week text not null,
    year_week text not null,
    game_id text not null,
    home_team text not null,
    away_team text not null,
    home_conference text,
    away_conference text,
    home_score_pred double precision not null,
    away_score_pred double precision not null,
    spread_pred double precision not null,
    spread_line double precision not null,
    spread_play text not null,
    spread_win_prob double precision not null,
    spread_lock integer not null,
    total_pred double precision not null,
    total_line double precision not null,
    total_play text not null,
    total_win_prob double precision not null,
    total_lock integer not null,
    date_time text not null,
    model_version text,
    write_time timestamptz not null,
    primary key (year_week, game_id)
);

-- Keep this setup script compatible with CFB tables created before conference
-- metadata was added. Review and apply these scoped statements in deployed environments.
alter table sports_models.cfb_expected_points_picks
    add column if not exists home_conference text;

alter table sports_models.cfb_expected_points_picks
    add column if not exists away_conference text;

create index if not exists cfb_expected_points_picks_game_id_idx
    on sports_models.cfb_expected_points_picks (game_id);

create index if not exists cfb_expected_points_picks_season_week_idx
    on sports_models.cfb_expected_points_picks (season, week);

create table if not exists sports_models.cfb_expected_points_results (
    season integer not null,
    week text not null,
    year_week text not null,
    game_id text not null,
    home_team text not null,
    away_team text not null,
    home_conference text,
    away_conference text,
    home_score integer not null,
    away_score integer not null,
    home_score_pred double precision not null,
    away_score_pred double precision not null,
    spread_pred double precision not null,
    spread_line double precision not null,
    true_spread double precision not null,
    spread_play text not null,
    spread_win_prob double precision not null,
    spread_lock integer not null,
    correct_spread_play text,
    spread_win integer,
    total_pred double precision not null,
    total_line double precision not null,
    true_total double precision not null,
    total_play text not null,
    total_win_prob double precision not null,
    total_lock integer not null,
    correct_total_play text,
    total_win integer,
    date_time text not null,
    model_version text,
    primary key (year_week, game_id)
);

alter table sports_models.cfb_expected_points_results
    add column if not exists home_conference text;

alter table sports_models.cfb_expected_points_results
    add column if not exists away_conference text;

create index if not exists cfb_expected_points_results_game_id_idx
    on sports_models.cfb_expected_points_results (game_id);

create index if not exists cfb_expected_points_results_season_week_idx
    on sports_models.cfb_expected_points_results (season, week);

create table if not exists sports_models.cfb_expected_points_pick_updates (
    id bigserial primary key,
    year_week text not null,
    write_time timestamptz not null,
    week text not null,
    season integer not null,
    environment text not null,
    client_name text not null,
    runtime double precision not null,
    model_version text,
    source_git_sha text,
    evaluation_metrics jsonb not null default '{}'::jsonb,
    pick_changes integer not null,
    pick_changes_games jsonb not null default '[]'::jsonb,
    play_changes integer not null,
    play_changes_games jsonb not null default '[]'::jsonb,
    updates_skipped integer not null,
    picks_num integer not null,
    difference_df jsonb not null default '[]'::jsonb,
    picks_df jsonb not null default '[]'::jsonb
);

create unique index if not exists cfb_expected_points_pick_updates_unique_idx
    on sports_models.cfb_expected_points_pick_updates (year_week, write_time, client_name);

create index if not exists cfb_expected_points_pick_updates_year_week_idx
    on sports_models.cfb_expected_points_pick_updates (year_week, write_time desc);

-- Versioning bootstrap and compatibility for databases created before release tracking.
alter table sports_models.cfb_expected_points_picks
    add column if not exists model_version text;
alter table sports_models.cfb_expected_points_results
    add column if not exists model_version text;
alter table sports_models.cfb_expected_points_pick_updates
    add column if not exists model_version text;
alter table sports_models.cfb_expected_points_pick_updates
    add column if not exists source_git_sha text;
alter table sports_models.cfb_expected_points_pick_updates
    add column if not exists evaluation_metrics jsonb not null default '{}'::jsonb;

insert into sports_models.model_releases (
    model_key, version, major_version, minor_version, title, public_summary,
    changes_md, evaluation_md, internal_notes_md, source_git_sha, deployed_at, first_pick_at
) values
    (
        'nfl_expected_points', '1.0', 1, 0, 'Versioning baseline',
        'Historical NFL expected-points records are attributed to the initial versioning baseline.',
        'Existing records are retained under version 1.0 because their original recipe was not recorded.',
        null, 'Backfilled during model-version tracking bootstrap.', null, now(), null
    ),
    (
        'cfb_expected_points', '1.0', 1, 0, 'Versioning baseline',
        'Historical CFB expected-points records are attributed to the initial versioning baseline.',
        'Existing records are retained under version 1.0 because their original recipe was not recorded.',
        null, 'Backfilled during model-version tracking bootstrap.', null, now(), null
    )
on conflict (model_key, version) do nothing;

update sports_models.nfl_expected_points_picks
set model_version = '1.0'
where model_version is null;
update sports_models.nfl_expected_points_results
set model_version = '1.0'
where model_version is null;
update sports_models.nfl_expected_points_pick_updates
set model_version = '1.0'
where model_version is null;
update sports_models.cfb_expected_points_picks
set model_version = '1.0'
where model_version is null;
update sports_models.cfb_expected_points_results
set model_version = '1.0'
where model_version is null;
update sports_models.cfb_expected_points_pick_updates
set model_version = '1.0'
where model_version is null;

update sports_models.model_releases
set first_pick_at = coalesce(
    (select min(write_time) from sports_models.nfl_expected_points_pick_updates),
    first_pick_at
)
where model_key = 'nfl_expected_points' and version = '1.0';
update sports_models.model_releases
set first_pick_at = coalesce(
    (select min(write_time) from sports_models.cfb_expected_points_pick_updates),
    first_pick_at
)
where model_key = 'cfb_expected_points' and version = '1.0';

alter table sports_models.nfl_expected_points_picks
    alter column model_version set not null;
alter table sports_models.nfl_expected_points_results
    alter column model_version set not null;
alter table sports_models.nfl_expected_points_pick_updates
    alter column model_version set not null;
alter table sports_models.cfb_expected_points_picks
    alter column model_version set not null;
alter table sports_models.cfb_expected_points_results
    alter column model_version set not null;
alter table sports_models.cfb_expected_points_pick_updates
    alter column model_version set not null;

create index if not exists nfl_expected_points_picks_model_version_idx
    on sports_models.nfl_expected_points_picks (model_version);
create index if not exists nfl_expected_points_results_model_version_idx
    on sports_models.nfl_expected_points_results (model_version);
create index if not exists cfb_expected_points_picks_model_version_idx
    on sports_models.cfb_expected_points_picks (model_version);
create index if not exists cfb_expected_points_results_model_version_idx
    on sports_models.cfb_expected_points_results (model_version);

create or replace view sports_models.cfb_expected_points_latest_picks as
select
    season,
    week,
    year_week,
    game_id,
    home_team,
    away_team,
    home_score_pred,
    away_score_pred,
    spread_pred,
    spread_line,
    spread_play,
    spread_win_prob,
    spread_lock,
    total_pred,
    total_line,
    total_play,
    total_win_prob,
    total_lock,
    date_time,
    write_time,
    home_conference,
    away_conference,
    model_version
from sports_models.cfb_expected_points_picks
where year_week = (
    select p.year_week
    from sports_models.cfb_expected_points_picks p
    order by p.season desc, cast(p.week as integer) desc
    limit 1
);

create or replace view sports_models.cfb_expected_points_latest_updates as
select distinct on (year_week)
    year_week,
    write_time,
    week,
    season,
    environment,
    client_name,
    runtime,
    pick_changes,
    pick_changes_games,
    play_changes,
    play_changes_games,
    updates_skipped,
    picks_num,
    model_version,
    source_git_sha,
    evaluation_metrics
from sports_models.cfb_expected_points_pick_updates
order by year_week, write_time desc;
