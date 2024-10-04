import pandas as pd


def get_result_stats(df, Verbose = False):
    
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    total_spread_wins = df['spread_win'].sum()
    total_total_wins = df['total_win'].sum()
    total_games = len(df)

    total_spread_ratio = 100 * total_spread_wins / total_games
    total_total_ratio = 100 * total_total_wins / total_games

    total_spread_lock_wins = df[df['spread_lock']==1]['spread_win'].sum()
    total_total_lock_wins = df[df['total_lock']==1]['total_win'].sum()
    total_spread_lock_games = len(df[df['spread_lock']==1])
    total_total_lock_games = len(df[df['total_lock']==1])


    total_spread_lock_ratio = 100* total_spread_lock_wins / total_spread_lock_games 
    total_total_lock_ratio = 100 * total_total_lock_wins / total_total_lock_games

    if Verbose:

        print(f"Spread Win percentage: {total_spread_ratio:.2f}%")
        print(f"Total win percentage: {total_total_ratio:.2f}%\n")

        if total_spread_lock_games != 0:
            total_spread_lock_ratio = 100 * total_spread_lock_wins / total_spread_lock_games
            print(f"Spread lock win percentage: {total_spread_lock_ratio:.2f}%")
        else:
            print("No spread locks")

        if total_total_lock_games != 0:
            total_total_lock_ratio = 100 * total_total_lock_wins / total_total_lock_games
            print(f"Total lock win percentage: {total_total_lock_ratio:.2f}%")
        else:
            print("No total locks")

    data = {
        "pedicted_games": total_games,

        "spread_wins": total_spread_wins,
        "spread_win_pct": total_spread_ratio,
        "spread_lock_predictions": total_spread_lock_games,
        "spread_lock_wins": total_spread_lock_wins,
        "spread_lock_win_pct": total_spread_lock_ratio,

        "total_wins": total_total_wins,
        "total_win_pct": total_total_ratio,
        "total_lock_predictions": total_spread_lock_games,
        "total_lock_wins": total_spread_lock_wins,
        "total_lock_win_pct": total_spread_lock_ratio,
    }

    return data
