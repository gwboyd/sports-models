import pandas as pd

def calculate_wins(df):

    df = df.copy()

    df['true_spread'] = df['away_score'] - df['home_score']

    df['correct_spread_play'] = df.apply(
        lambda row: row['home_team'] if row['true_spread'] < row['spread_line'] else (
            row['away_team'] if row['true_spread'] > row['spread_line'] else None
        ), axis=1
    )

    df['spread_win'] = df.apply(
        lambda row: None if pd.isnull(row['correct_spread_play']) else (
            1 if row['spread_play'] == row['correct_spread_play'] else 0
        ), axis=1

    )

    df['true_total'] = df['away_score'] + df['home_score']

    df['correct_total_play'] = df.apply(
        lambda row: 'under' if row['true_total'] < row['total_line'] else (
            'over' if row['true_total'] > row['total_line'] else None
        ), axis=1
    )

    df['total_win'] = df.apply(
        lambda row: None if pd.isnull(row['correct_total_play']) else (
            1 if row['total_play'] == row['correct_total_play'] else 0
        ), axis=1
    )




    return df


def get_result_stats(df, Verbose=False):

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # For Spread Bets
    spread_wins = df['spread_win'] == 1
    spread_losses = df['spread_win'] == 0
    spread_pushes = df['spread_win'].isnull()

    total_spread_wins = spread_wins.sum()
    total_spread_losses = spread_losses.sum()
    total_spread_pushes = spread_pushes.sum()
    total_spread_games = total_spread_wins + total_spread_losses  # Excludes pushes

    # Calculate spread win percentage
    if total_spread_games > 0:
        total_spread_ratio = 100 * total_spread_wins / total_spread_games
    else:
        total_spread_ratio = 0

    # For Total Bets
    total_wins = df['total_win'] == 1
    total_losses = df['total_win'] == 0
    total_pushes = df['total_win'].isnull()

    total_total_wins = total_wins.sum()
    total_total_losses = total_losses.sum()
    total_total_pushes = total_pushes.sum()
    total_total_games = total_total_wins + total_total_losses  # Excludes pushes

    # Calculate total win percentage
    if total_total_games > 0:
        total_total_ratio = 100 * total_total_wins / total_total_games
    else:
        total_total_ratio = 0

    # For Spread Locks
    spread_lock_df = df[df['spread_lock'] == 1]
    spread_lock_wins = spread_lock_df['spread_win'] == 1
    spread_lock_losses = spread_lock_df['spread_win'] == 0
    spread_lock_pushes = spread_lock_df['spread_win'].isnull()

    total_spread_lock_wins = spread_lock_wins.sum()
    total_spread_lock_losses = spread_lock_losses.sum()
    total_spread_lock_pushes = spread_lock_pushes.sum()
    total_spread_lock_games = total_spread_lock_wins + total_spread_lock_losses  # Excludes pushes

    # Calculate spread lock win percentage
    if total_spread_lock_games > 0:
        total_spread_lock_ratio = 100 * total_spread_lock_wins / total_spread_lock_games
    else:
        total_spread_lock_ratio = 0

    # For Total Locks
    total_lock_df = df[df['total_lock'] == 1]
    total_lock_wins = total_lock_df['total_win'] == 1
    total_lock_losses = total_lock_df['total_win'] == 0
    total_lock_pushes = total_lock_df['total_win'].isnull()

    total_total_lock_wins = total_lock_wins.sum()
    total_total_lock_losses = total_lock_losses.sum()
    total_total_lock_pushes = total_lock_pushes.sum()
    total_total_lock_games = total_total_lock_wins + total_total_lock_losses  # Excludes pushes

    # Calculate total lock win percentage
    if total_total_lock_games > 0:
        total_total_lock_ratio = 100 * total_total_lock_wins / total_total_lock_games
    else:
        total_total_lock_ratio = 0

    if Verbose:
        print(f"Spread Win Percentage: {total_spread_ratio:.2f}%")
        print(f"Total Win Percentage: {total_total_ratio:.2f}%\n")

        if total_spread_lock_games > 0:
            print(f"Spread Lock Win Percentage: {total_spread_lock_ratio:.2f}%")
        else:
            print("No Spread Locks")

        if total_total_lock_games > 0:
            print(f"Total Lock Win Percentage: {total_total_lock_ratio:.2f}%")
        else:
            print("No Total Locks")

    data = {
        "predicted_games": len(df),

        # Spread Stats
        "spread_wins": int(total_spread_wins),
        "spread_losses": int(total_spread_losses),
        "spread_pushes": int(total_spread_pushes),
        "spread_win_pct": total_spread_ratio,
        "spread_lock_predictions": int(total_spread_lock_wins + total_spread_lock_losses + total_spread_lock_pushes),
        "spread_lock_wins": int(total_spread_lock_wins),
        "spread_lock_losses": int(total_spread_lock_losses),
        "spread_lock_pushes": int(total_spread_lock_pushes),
        "spread_lock_win_pct": total_spread_lock_ratio,

        # Total Stats
        "total_wins": int(total_total_wins),
        "total_losses": int(total_total_losses),
        "total_pushes": int(total_total_pushes),
        "total_win_pct": total_total_ratio,
        "total_lock_predictions": int(total_total_lock_wins + total_total_lock_losses + total_total_lock_pushes),
        "total_lock_wins": int(total_total_lock_wins),
        "total_lock_losses": int(total_total_lock_losses),
        "total_lock_pushes": int(total_total_lock_pushes),
        "total_lock_win_pct": total_total_lock_ratio,
    }

    return data
