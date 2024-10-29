# %%
import time
start_time = time.time()

# %%
#parameters
current_year = None
current_week = None
client_name = None

# %%
import time
start_time = time.time()

# %%
if not (current_year or current_week or client_name):
    current_year = 2024
    current_week = 8
    client_name = "notebook"

# %%
start_year = 2010

current_year_week = f"{current_year}_{current_week}"

# %%
current_week

# %% [markdown]
# # Imports

# %%
import os
if os.environ.get("LOCALHOST") == "True":
    os.chdir('/Users/wiboyd/Desktop/Misc/sports-models')

# %%
#turns off any warnings
import warnings
warnings.filterwarnings('ignore')

#various modules
import nfl_data_py as nfl
#from sportsreference.nfl.boxscore import Boxscores, Boxscore

import os
import sys


import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pickle
import json
import seaborn as sns
import numpy as np
import xgboost as xgb
import math
import datetime
import pytz

from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.nfl.utils.expected_points_functions import get_result_stats, calculate_wins
from src.utils.dynamo_functions import dynamodb, delete_dynamo_enteries, dynamo_add_column, query_results, convert_to_decimal, convert_floats_to_decimal

from src.utils.data_models.picks_response import PickResponse

import boto3
from decimal import Decimal
from boto3.dynamodb.conditions import Key


#from googleapiclient.discovery import build
#from google.oauth2.service_account import Credentials

# %%
import psutil
total_memory = psutil.virtual_memory().total / (1024 ** 3)
# Get used memory in GB
used_memory = psutil.virtual_memory().used / (1024 ** 3)
# Get available memory in GB
available_memory = psutil.virtual_memory().available / (1024 ** 3)

print(f"Total Memory: {total_memory:.2f} GB")
print(f"Used Memory: {used_memory:.2f} GB")
print(f"Available Memory: {available_memory:.2f} GB")

# %% [markdown]
# # Bring in data

# %%
years = range(start_year - 1, current_year + 1)
true_years = range(start_year, current_year + 1)

pbp = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)
weekly = nfl.import_weekly_data(years, downcast=True)
season = nfl.import_seasonal_data(years, s_type='REG')
#rosetrs = nfl.import_weekly_rosters(years)
injury = nfl.import_injuries(years)
#pass_ratings = nfl.import_weekly_pfr('pass',years)
full_schedule = nfl.import_schedules(true_years)
teams = nfl.import_team_desc()


# %% [markdown]
# # Data transformation and feature engineering

# %%
pbp['home_team'] = pbp['home_team'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
pbp['away_team'] = pbp['away_team'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
pbp['posteam'] = pbp['posteam'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
pbp['defteam'] = pbp['defteam'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})

full_schedule['home_team'] = full_schedule['home_team'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
full_schedule['away_team'] = full_schedule['away_team'].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})

# %%
pbp['rz'] = np.where(((pbp['defteam'] == pbp['side_of_field']) & ((pbp['yardline_100'] <= 20) | (pbp['yardline_100'] >= 80))), 1, 0)

# %%
schedule_columns = ['game_id', 'season', 'week', 'home_team','away_team','weekday', 'home_qb_id', 'away_qb_id', 'home_moneyline', 'away_moneyline','spread_line',
                 'home_spread_odds', 'away_spread_odds','total_line',  'over_odds', 'roof', 'away_rest', 'home_rest', 'stadium_id', 'div_game', 'gametime', 'gameday']
schedule = full_schedule[schedule_columns].dropna(subset = 'total_line')

# %%
def get_implied_totals(row):
    home_points = (row['total_line'] / 2) + (row['spread_line'] / 2)
    away_points = (row['total_line'] / 2) - (row['spread_line'] / 2)
    return pd.Series([home_points, away_points])


# %%
schedule[['implied_points_home', 'implied_points_away']] = schedule.apply(get_implied_totals, axis=1)

# %% [markdown]
# ## Advanced stats

# %%
melted_schedule_teams = pd.melt(schedule, id_vars=['season', 'week'], value_vars=['home_team', 'away_team'], value_name='team').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'team']]
melted_schedule_qbs = pd.melt(schedule, id_vars=['season', 'week'], value_vars=['home_qb_id', 'away_qb_id'], value_name='qb_id').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'qb_id']]
melted_schedule = pd.concat([melted_schedule_teams, melted_schedule_qbs['qb_id']], axis=1)

# %% [markdown]
# ### QBR

# %%
def nfl_passer_rating(cmpls, yds, tds, ints):
     """ Defines a function which handles passer rating calculation for the NFL."""
     def _min_max(x, xmin, xmax):

         # Check if x is less than the minimum. If so, return the minimum.
         if x < xmin:
             return xmin
         
         # Check if x is greater than the maximum. If so, return the maximum.
         elif x > xmax:
             return xmax
         
         # Otherwise, just return x. And weep for the future.
         else:
             return x
             
     # Step 0: Make sure these are floats, dammit.
     cmpls = cmpls + 0.0
     yds = yds + 0.0
     tds = tds + 0.0
     ints = ints + 0.0
     
     # Step 1: The completion percentage.         
     step_1 = cmpls - 0.3
     step_1 = step_1 * 5
     step_1 = _min_max(step_1, 0, 2.375)
     
     # Step 2: The yards per attempt.
     step_2 = yds - 3
     step_2 = step_2 * 0.25
     step_2 = _min_max(step_2, 0, 2.375)
     
     # Step 3: Touchdown percentage.
     step_3 = tds * 20
     step_3 = _min_max(step_3, 0, 2.375)
     
     # Step 4: Interception percentage.
     step_4 = ints * 25
     step_4 = 2.375 - step_4
     step_4 = _min_max(step_4, 0, 2.375)
     
     # Step 5: Compute the rating based on the sum of steps 1-4.
     rating = step_1 + step_2 + step_3 + step_4 + 0.0
     rating = rating / 6
     rating = rating * 100
     
     # Step 6: Return the rating, formatted to 1 decimal place, as a Decimal.
     return rating

def calculate_qbr(att, comp, yds, td, ints):

    #replace with college qbr?
    if att == 0:
        return 0

    a = ((comp/att) - 0.3) * 5
    b = ((yds/att) - 3) * 0.25
    c = (td/att) * 20
    d = 2.375 - ((ints/att) * 25)

    for i in [a, b, c, d]:
        if i < 0:
            i = 0
        elif i > 2.375:
            i = 2.375

    passer_rating = ((a + b + c + d) / 6) * 100

    return passer_rating

# %%
qbs_weekly = weekly[weekly['position_group']=='QB']
qbs_weekly['passer_rating'] = qbs_weekly.apply(lambda row: nfl_passer_rating(row['completions'], row['passing_yards'], row['passing_tds'], row['interceptions']), axis=1)
qbs_weekly['qbr'] = qbs_weekly.apply(lambda row: calculate_qbr(row['attempts'], row['completions'], row['passing_yards'], row['passing_tds'], row['interceptions']), axis=1)

# %%
starting_qbs =  pd.merge(melted_schedule.rename(columns={"qb_id": "player_id"}), qbs_weekly, on=['season', 'week', 'player_id'], how='left').sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)
first_game = starting_qbs.groupby('player_id').first()[['qbr', 'passer_rating']].mean()

starting_qbs['passer_rating_shifted'] = starting_qbs.groupby('player_id')['passer_rating'].shift()
starting_qbs['qbr_shifted'] = starting_qbs.groupby('player_id')['qbr'].shift()
starting_qbs['player_name'] = starting_qbs.groupby('player_id')['player_name'].shift()

#starting_qbs['passer_rating_shifted'] = starting_qbs['passer_rating_shifted'].fillna(value = first_game['passer_rating'])
starting_qbs['qbr_shifted'] = starting_qbs['qbr_shifted'].fillna(value = first_game['qbr'])

starting_qbs['ewma_passer_rating'] = starting_qbs.groupby('player_id')['passer_rating_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
starting_qbs['ewma_qbr'] = starting_qbs.groupby('player_id')['qbr_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())

starting_qbs[(starting_qbs['season'] == 2023) & (starting_qbs['week'] == current_week)][['team', 'player_name', 'passer_rating_shifted', 'qbr_shifted', 'ewma_passer_rating', 'ewma_qbr']].head()

# %% [markdown]
# ### EPA per Play

# %%
def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted EPA with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i+1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]
            
    return pd.Series(values, index=x.index)

# seperate EPA in to rushing offense, rushing defense, passing offense, passing defense for each team
rushing_offense_epa = pbp.loc[pbp['rush_attempt'] == 1, :].groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()
rushing_defense_epa = pbp.loc[pbp['rush_attempt'] == 1, :].groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()
passing_offense_epa = pbp.loc[pbp['pass_attempt'] == 1, :].groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()
passing_defense_epa = pbp.loc[pbp['pass_attempt'] == 1, :].groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

rushing_offense_epa =  pd.merge(melted_schedule_teams.rename(columns={"team": "posteam"}), rushing_offense_epa, on=['season', 'week', 'posteam'], how='outer').sort_values(by=['posteam', 'season', 'week']).reset_index(drop=True)
rushing_defense_epa =  pd.merge(melted_schedule_teams.rename(columns={"team": "defteam"}), rushing_defense_epa, on=['season', 'week', 'defteam'], how='outer').sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)
passing_offense_epa =  pd.merge(melted_schedule_teams.rename(columns={"team": "posteam"}), passing_offense_epa, on=['season', 'week', 'posteam'], how='outer').sort_values(by=['posteam', 'season', 'week']).reset_index(drop=True)
passing_defense_epa =  pd.merge(melted_schedule_teams.rename(columns={"team": "defteam"}), passing_defense_epa, on=['season', 'week', 'defteam'], how='outer').sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)

# # lag EPA one period back
rushing_offense_epa['epa_shifted'] = rushing_offense_epa.groupby('posteam')['epa'].shift()
rushing_defense_epa['epa_shifted'] = rushing_defense_epa.groupby('defteam')['epa'].shift()
passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['epa'].shift()
passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['epa'].shift()

# In each case, calculate EWMA with a static window and dynamic window and assign it as a column 
rushing_offense_epa['ewma'] = rushing_offense_epa.groupby('posteam')['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
rushing_offense_epa['ewma_dynamic_window'] = rushing_offense_epa.groupby('posteam').apply(dynamic_window_ewma).values
rushing_defense_epa['ewma'] = rushing_defense_epa.groupby('defteam')['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
rushing_defense_epa['ewma_dynamic_window'] = rushing_defense_epa.groupby('defteam').apply(dynamic_window_ewma).values
passing_offense_epa['ewma'] = passing_offense_epa.groupby('posteam')['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
passing_offense_epa['ewma_dynamic_window'] = passing_offense_epa.groupby('posteam').apply(dynamic_window_ewma).values
passing_defense_epa['ewma'] = passing_defense_epa.groupby('defteam')['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
passing_defense_epa['ewma_dynamic_window'] = passing_defense_epa.groupby('defteam').apply(dynamic_window_ewma).values

#Merge all the data together
offense_epa = rushing_offense_epa.merge(passing_offense_epa, on=['posteam', 'season', 'week'], suffixes=('_rushing', '_passing')).rename(columns={'posteam': 'team'})
defense_epa = rushing_defense_epa.merge(passing_defense_epa, on=['defteam', 'season', 'week'], suffixes=('_rushing', '_passing')).rename(columns={'defteam': 'team'})
epa = offense_epa.merge(defense_epa, on=['team', 'season', 'week'], suffixes=('_offense', '_defense'))

#remove the first season of data
epa = epa.loc[epa['season'] != epa['season'].unique()[0], :]

epa = epa.reset_index(drop=True)

epa.head()

# %% [markdown]
# ### Success Rates

# %%
pbp['custom_success'] = ((pbp['down'] == 1) & (pbp['yards_gained'] >= 0.4 * pbp['ydstogo'])) | \
                        ((pbp['down'] == 2) & (pbp['yards_gained'] >= 0.6 * pbp['ydstogo'])) | \
                        ((pbp['down'].isin([3, 4])) & (pbp['yards_gained'] >= pbp['ydstogo']))

rushing_offense_success = pbp.loc[pbp['rush_attempt'] == 1, :].groupby(['posteam', 'season', 'week'], as_index=False)['custom_success'].mean()
rushing_defense_success = pbp.loc[pbp['rush_attempt'] == 1, :].groupby(['defteam', 'season', 'week'], as_index=False)['custom_success'].mean()
passing_offense_success = pbp.loc[pbp['pass_attempt'] == 1, :].groupby(['posteam', 'season', 'week'], as_index=False)['custom_success'].mean()
passing_defense_success = pbp.loc[pbp['pass_attempt'] == 1, :].groupby(['defteam', 'season', 'week'], as_index=False)['custom_success'].mean()

rushing_offense_success =  pd.merge(melted_schedule_teams.rename(columns={"team": "posteam"}), rushing_offense_success, on=['season', 'week', 'posteam'], how='outer').sort_values(by=['posteam', 'season', 'week']).reset_index(drop=True)
rushing_defense_success =  pd.merge(melted_schedule_teams.rename(columns={"team": "defteam"}), rushing_defense_success, on=['season', 'week', 'defteam'], how='outer').sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)
passing_offense_success =  pd.merge(melted_schedule_teams.rename(columns={"team": "posteam"}), passing_offense_success, on=['season', 'week', 'posteam'], how='outer').sort_values(by=['posteam', 'season', 'week']).reset_index(drop=True)
passing_defense_success =  pd.merge(melted_schedule_teams.rename(columns={"team": "defteam"}), passing_defense_success, on=['season', 'week', 'defteam'], how='outer').sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)

rushing_offense_success['success_shifted'] = rushing_offense_success.groupby('posteam')['custom_success'].shift()
rushing_defense_success['success_shifted'] = rushing_defense_success.groupby('defteam')['custom_success'].shift()
passing_offense_success['success_shifted'] = passing_offense_success.groupby('posteam')['custom_success'].shift()
passing_defense_success['success_shifted'] = passing_defense_success.groupby('defteam')['custom_success'].shift()

rushing_offense_success['ewma_success_rate'] = rushing_offense_success.groupby('posteam')['success_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
rushing_defense_success['ewma_success_rate'] = rushing_defense_success.groupby('defteam')['success_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
passing_offense_success['ewma_success_rate'] = passing_offense_success.groupby('posteam')['success_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())
passing_defense_success['ewma_success_rate'] = passing_defense_success.groupby('defteam')['success_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())

offense_sucess = rushing_offense_success.merge(passing_offense_success, on=['posteam', 'season', 'week'], suffixes=('_rushing', '_passing')).rename(columns={'posteam': 'team'})
defense_sucess = rushing_defense_success.merge(passing_defense_success, on=['defteam', 'season', 'week'], suffixes=('_rushing', '_passing')).rename(columns={'defteam': 'team'})
success = offense_sucess.merge(defense_sucess, on=['team', 'season', 'week'], suffixes=('_offense', '_defense'))

#remove the first season of data
success = success.loc[success['season'] != success['season'].unique()[0], :]

success = success.reset_index(drop=True)
success.head(2)

# %% [markdown]
# ## Aggregate datasets

# %%
scores = pbp[['game_id','season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']]\
.drop_duplicates().reset_index(drop=True)\
.assign(home_team_win = lambda x: (x.home_score > x.away_score).astype(int))

schedule_scores = schedule.merge(scores[['game_id','home_score', 'away_score']], on='game_id', how='left')

df = schedule_scores.merge(epa.rename(columns={'team': 'home_team'}), on=['home_team', 'season', 'week'], how='left')\
.merge(epa.rename(columns={'team': 'away_team'}), on=['away_team', 'season', 'week'], how='left', suffixes=('_home', '_away'))

df = df.merge(success.rename(columns={'team': 'home_team'}), on=['home_team', 'season', 'week'], how='left')\
.merge(success.rename(columns={'team': 'away_team'}), on=['away_team', 'season', 'week'], how='left', suffixes=('_home', '_away'))

df = df.merge(starting_qbs.rename(columns={'team': 'home_team'}), on=['home_team', 'season', 'week'], how='left')\
.merge(starting_qbs.rename(columns={'team': 'away_team'}), on=['away_team', 'season', 'week'], how='left', suffixes=('_home', '_away'))

df = df.rename(columns={
    'home_rest': 'rest_home',
    'away_rest': 'rest_away',
    'home_moneyline': 'moneyline_home',
    'away_moneyline': 'moneyline_away',
    'home_spread_odds': 'spread_odds_home',
    'away_spread_odds': 'spread_odds_away'
})

df['pred_team'] = 'undefined'


# %%
print(list(df.columns))

# %%
df = df[~((df['season'] == 2010) & (df['week'] == 1))]

df['spread_line'] = df['spread_line'] * -1

df['year_week'] = df['season'].astype(str) + '_' + df['week'].astype(str)
df['date_time'] = df['gameday'].astype(str) + '-' + df['gametime'].astype(str)


#df.loc[(df['season'] == 2023) & (df['week'] == 3) & (df['weekday'] == "Monday"), ['home_score', 'away_score']] = np.nan

# %% [markdown]
# # Define Features

# %%
targets = ['home_score', 'away_score']
ewma_features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column] + [column for column in df.columns if 'ewma' in column and 'success_rate' in column]

cat_features = ['pred_team', 'roof', 'weekday']

betting_features = ['moneyline_home', 'spread_line', 'spread_odds_home', 'total_line', 'over_odds'] 

other_features = ['rest_away', 'rest_home', 'div_game', 'implied_points_home', 
                  'implied_points_away','ewma_qbr_home', 'ewma_qbr_away'] 

numeric_features = [x for x in (other_features + betting_features + ewma_features) if x != 'div_game'] # + betting_features
float_features = numeric_features.copy()
boolean_features = 'div_game'

features = other_features + cat_features + ewma_features + betting_features # + betting_features
for feature in features:
  print(feature)

# %%
train_df = df.dropna(subset = 'home_score', inplace=False)
train_df = train_df[~((train_df['season'] == current_year) & (df['week'] >= current_week))]

input_features = features + ['moneyline_away', 'spread_odds_away']

X = train_df[input_features]
y = train_df[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # Define and fit pipeline

# %%
class home_away_transformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):    
        return self
    
    def transform(self, X_, y=None):
        X = X_.copy()
        if 'pred_team' in X.columns:
            mask = X['pred_team'] == 'away'
            if mask.any():
                
                #switch columns
                home_cols = [col for col in X.columns if col.endswith('_home')]
                for home_col in home_cols:
                    base_col = home_col[:-5]  # Remove '_home'
                    away_col = base_col + '_away'
                    if away_col in X.columns:
                        # Swap values where team == 'away'
                        temp = X.loc[mask, home_col].copy()
                        X.loc[mask, home_col] = X.loc[mask, away_col]
                        X.loc[mask, away_col] = temp
                
                #flip spread
                X.loc[mask, 'spread_line'] = -1.0 * X.loc[mask, 'spread_line']

        return X
      
home_away_transformer = home_away_transformer()

# %%
test = X.copy()
test['game_id'] = train_df['game_id']

test['home_team'] = train_df['home_team']
test['away_team'] = train_df['away_team']

# %%
X_home = test.copy()
X_away = test.copy()

y_home = y.iloc[:, 0]
y_away = y.iloc[:,1]


X_home['pred_team'] = 'home'
X_away['pred_team'] = 'away'

test_X = pd.concat([X_home, X_away], ignore_index=True)
test_y = pd.concat([y_home, y_away], ignore_index=True)

trans_test = home_away_transformer.transform(test_X)

# %%
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

# %%
numeric_transformer = Pipeline(steps=[
    ("num_selector", ItemSelector(numeric_features)),
    ('num_imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


boolean_transformer = Pipeline(steps=[
    ("bool_selector", ItemSelector(boolean_features)),
    ('bool_imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())])


float_transformer = Pipeline(steps=[
    ("float_selector", ItemSelector(float_features)),
    ('float_imputer', SimpleImputer(strategy='mean')),
    #('qcut', KBinsDiscretizer(n_bins=3, strategy='quantile')),
    ('qcut', StandardScaler())])

# boolean_transformer = Pipeline(steps=[])


# float_transformer = Pipeline(steps=[])




categorical_transformer5 = Pipeline(steps = [    
  ("cat_selector", ItemSelector(cat_features)), #drop the actual action for inference and training
  ("cat_imputer", SimpleImputer(strategy='constant', fill_value = "missing_categorical")),
  ])



#cat_column_index = list(range(len(boolean_features)+len(float_features)+len(ts_features), len(boolean_features)+len(float_features)+len(ts_features)+len(categorical_features_no_option)))


preprocessor6 = ColumnTransformer(
    transformers=[
        ('bool', boolean_transformer, boolean_features),
        ('float', float_transformer, float_features),  
        ('cat', categorical_transformer5, cat_features)
      ])

# %%
# simple transfomer to convert categorical columns to category type for lightgbm auto encoding
class Cat_typechange_Transformer(BaseEstimator, TransformerMixin):

    # Class Constructor

    def __init__(self,feature_names=[],cat_names=[]):
        self.feature_names = feature_names
        self.cat_names = cat_names
    def fit(self, X, y=None):
        
                    
        return self
    
    def transform(self, X_, y=None):

        X = pd.DataFrame(X_)
        #print(X.head(5))
        org_names = X.columns
        #print(org_names)
        
        changes = dict(zip(org_names,self.feature_names))
        #print(changes)
        X.rename(columns=changes,inplace=True)
        
        
        #X.columns = self.feature_name
        #print(X.columns)
        for col in X.columns:
          if col in self.cat_names:
            X[col] = X[col].astype('category')
          else:
            X[col] = X[col].astype('float')

               
        return X 

# %%
class ScorePipeline(Pipeline):
    def __init__(self, steps, **kwargs):

        #call parents innit
        super().__init__(steps, **kwargs)

    def predict_scores(self, X, team):
        """Custom predict method to add team info before predicting."""
        X = X.copy()
        X['pred_team'] = team
        return super().predict(X)  # Use the parent's predict method

# %%
pipeline_template = make_pipeline(
                        home_away_transformer,
                        ItemSelector(features),
                        #preprocessor6,
                        Cat_typechange_Transformer(feature_names=features, cat_names=cat_features),
                        LGBMRegressor(verbose=-1, n_jobs=-1, random_state = 31)
                         )

score_pipeline =  ScorePipeline(steps=pipeline_template.steps)

# %%
def fit_score_model(_X, _y, score_pipeline):

    X_home = _X.copy()
    X_away = _X.copy()

    y_home = _y.iloc[:, 0]
    y_away = _y.iloc[:,1]


    X_home['pred_team'] = 'home'
    X_away['pred_team'] = 'away'

    X = pd.concat([X_home, X_away], ignore_index=True)
    y = pd.concat([y_home, y_away], ignore_index=True)
    
    
    param_grid = {
        'lgbmregressor__n_estimators': [300,400],
        'lgbmregressor__max_depth':[8,12],
        'lgbmregressor__learning_rate':[0.05,0.1]
    }
    
    search = GridSearchCV(score_pipeline, param_grid, cv =2, n_jobs=-1, scoring = "neg_mean_squared_error")

    search.fit(X,y)
    model = search.best_estimator_


    return model

# %%
def scores_to_bets(results):

    results['spread_pred']= results['away_score_pred'] - results['home_score_pred']
    results['total_pred'] = results['home_score_pred'] + results['away_score_pred']

    results['spread_play'] = results.apply(lambda row: row['home_team'] if row['spread_pred'] < row['spread_line'] else row['away_team'], axis=1)
    results['total_play'] = results.apply(lambda row: 'under' if row['total_pred'] < row['total_line'] else ('over' if row['total_pred'] > row['total_line'] else None), axis=1)

    results['spread_diff'] = abs(results['spread_line'] - results['spread_pred'])
    results['total_diff'] = abs(results['total_line'] - results['total_pred'])

    return results


# %%
def fit_eval(df, X_train, X_test, y_train, y_test, score_pipeline):

    score_model =  fit_score_model(X_train, y_train, score_pipeline)

    away_scores = score_model.predict_scores(X_test, 'away')
    home_scores = score_model.predict_scores(X_test, 'home')


    results = df.loc[X_test.index]
    results['home_score']= y_test.iloc[:,0]
    results['away_score']= y_test.iloc[:,1]
    results['away_score_pred'] = away_scores
    results['home_score_pred'] = home_scores
    
    results = scores_to_bets(results)
    results = calculate_wins(results)

    return results 

# %%

def fit_classifiers(results, spread_class_features, total_class_features):


    spread_X = results[spread_class_features]
    spread_y = results['spread_win']
    total_X = results[total_class_features]
    total_y = results['total_win']

    spread_X = spread_X[spread_y.notna()]
    spread_y =  spread_y.dropna()

    total_X = total_X[total_y.notna()]
    total_y = total_y.dropna()




    param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }

    # Create and train the classifiers LGBMClassifier RandomForestClassifier
    spread_clf = GridSearchCV(LGBMClassifier(random_state = 2,  verbose = -1), param_grid, cv=5)
    total_clf = GridSearchCV(LGBMClassifier(random_state = 2, verbose = -1), param_grid, cv=5)
    spread_clf.fit(spread_X, spread_y)
    total_clf.fit(total_X, total_y)

    return spread_clf, total_clf

# %% [markdown]
# ### Fit model on training set and get eval df

# %%
results  = fit_eval(train_df, X_train, X_test, y_train, y_test, score_pipeline)

# %% [markdown]
# ## Fit classifiers on test set

# %%
spread_class_features =  ewma_features+betting_features+other_features+['spread_diff']
total_class_features = ewma_features+betting_features+other_features+['total_diff']

spread_clf, total_clf = fit_classifiers(results, spread_class_features, total_class_features)

# %% [markdown]
# ## Retrain on entire dataset

# %%
print(train_df.shape)


score_model =  fit_score_model(X, y, score_pipeline)

# %% [markdown]
# # Model metrics and performance

# %%
from matplotlib import pyplot as plt
feature_importances = score_model.named_steps['lgbmregressor'].feature_importances_
import_df = pd.DataFrame(zip(features,feature_importances),columns=['feature_name','feature_importance']).sort_values(by='feature_importance',ascending=True)

feature_imp = pd.Series(import_df['feature_importance'].values,index=import_df['feature_name'])
feature_imp.plot(kind='barh',color='blue')
plt.show()

# %%
print(f"Total win percentage: {100.0*results['total_win'].mean()}\nSpread win percentage: {100*results['spread_win'].mean()}")
results [['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score', 'away_score_pred', 'away_score', 
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'correct_spread_play', 'spread_win',
          'total_pred', 'total_line', 'true_total', 'total_play', 'correct_total_play', 'total_win']]

# %%
from scipy.stats import pointbiserialr

corr, _ = pointbiserialr(results["spread_win"], results["moneyline_home"])
print('Pearsons correlation: %.3f' % corr)

corr, _ = pointbiserialr(results["spread_win"], results["spread_diff"])
print('Pearsons correlation: %.3f' % corr)

# %% [markdown]
# # This week's plays

# %%
this_week = df[(df['season'] == current_year) & (df['week'] == current_week)]
#this_week = this_week[~this_week['home_score'].notnull()]

this_week_away_scores = score_model.predict_scores(this_week[input_features], 'away')
this_week_home_scores = score_model.predict_scores(this_week[features], 'home')

this_week['away_score_pred'] = this_week_away_scores
this_week['home_score_pred'] = this_week_home_scores

plays = scores_to_bets(this_week)


# %%

from sklearn.preprocessing import LabelEncoder

def win_probability(df, classifier, features):
    
    win_probs = classifier.predict_proba(df[features])[:, 1] * 100

    return win_probs

plays['spread_win_prob'] = win_probability(plays, classifier = spread_clf, features = spread_class_features)
plays['total_win_prob'] = win_probability(plays, classifier = total_clf, features = total_class_features)


# %%
def determine_plays(df, dont_update=[]):
    df = df.copy()

    max_spreads_plays = 5
    max_total_plays = 5
    
    
    # Determine the top n spread and total plays
    df['is_top_n_spread'] = df['spread_win_prob'].rank(method='first', ascending=False) <= max_spreads_plays
    df['is_top_n_total'] = df['total_win_prob'].rank(method='first', ascending=False) <= max_total_plays

    # Apply conditions to set spread_lock
    df['new_spread_lock'] = (
        df['is_top_n_spread'] &
        (abs(df['spread_pred'] - df['spread_line']) >= 0.5) &
        (df['spread_win_prob'] > 55.0)
    ).astype(int)

    df['new_total_lock'] = (
        df['is_top_n_total'] &
        (abs(df['total_pred'] - df['total_line']) >= 0.5) &
        (df['total_win_prob'] > 55.0)
    ).astype(int)
    
    # Update spread_lock and total_lock only for game_ids not in dont_update
    df['spread_lock'] = df.apply(
        lambda row: row['spread_lock'] if row['game_id'] in dont_update else row['new_spread_lock'],
        axis=1
    )
    df['total_lock'] = df.apply(
        lambda row: row['total_lock'] if row['game_id'] in dont_update else row['new_total_lock'],
        axis=1
    )
    
    # Drop the helper columns
    df.drop(['is_top_n_spread', 'is_top_n_total'], axis=1, inplace=True)
    
    return df

def print_plays(df):
    # Print Spread plays
    print("Spread plays:")
    spread_plays = df[df['spread_lock'] == 1]
    spread_plays = spread_plays.sort_values(by='spread_win_prob', ascending=False)
    for _, row in spread_plays.iterrows():
        mult = -1 if row['spread_play'] == row['away_team'] else 1
        pref = '+' if row['spread_line'] * mult > 0 else ''
        pref2 = '+' if row['spread_pred'] * mult > 0 else ''
        print(f"{row['home_team']}/{row['away_team']}: {row['spread_play']} {pref}{row['spread_line']*mult} "
              f"(model {row['spread_play']} {pref2}{(row['spread_pred']*mult):.2f}, "
              f"{row['spread_win_prob']:.2f}% win probability)")

    # Print Total plays
    print("\nTotal plays:")
    total_plays = df[df['total_lock'] == 1]
    total_plays = total_plays.sort_values(by='total_win_prob', ascending=False)
    for _, row in total_plays.iterrows():
        print(f"{row['home_team']}/{row['away_team']}: {row['total_play']} {row['total_line']} "
              f"(model {row['total_pred']:.2f}, {row['total_win_prob']:.2f}% win probability)")

plays = determine_plays(plays)

# %% [markdown]
# ### Check differences with database

# %%
picks_table = dynamodb.Table('nfl_expected_points_picks')
original_picks = pd.DataFrame(picks_table.scan()['Items'])

# %% [markdown]
# If game has already happened, don't update picks table

# %%
def get_ny_timestamp():
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.datetime.now(ny_tz)
    #formatted_timestamp = now_ny.strftime('%Y-%m-%d-%H:%M')
    return now_ny

compare_date_time_ny = np.datetime64(get_ny_timestamp()) + np.timedelta64(30, 'm')

original_picks['date_time_temp'] = pd.to_datetime(original_picks['date_time'], format='%Y-%m-%d-%H:%M')
dont_update = original_picks[original_picks['date_time_temp'] < compare_date_time_ny].copy().drop(columns=['date_time_temp'], inplace=False)


plays.set_index('game_id', inplace=True)
dont_update.set_index('game_id', inplace=True)

num_replacements = int(plays.index.isin(dont_update.index).sum())
print(f"Skipping updates for {num_replacements} picks")
print(dont_update.index.values)

plays_copy = plays.copy()
plays.update(dont_update)
updated_rows = plays_copy[plays_copy.index.isin(dont_update.index)]

plays.reset_index(inplace=True)
dont_update.reset_index(inplace=True)
updated_rows.reset_index(inplace=True)
original_picks.drop(columns=['date_time_temp'], inplace=True)

plays = determine_plays(plays, dont_update=list(dont_update['game_id'].values))

# %%
one = [dict(PickResponse(**item)) for _, item in original_picks.iterrows()]
two = [dict(PickResponse(**{**item, 'week': str(item['week'])})) for _, item in plays.iterrows()]

df_one = pd.DataFrame(one)
df_two = pd.DataFrame(two)

df_one['source']="database"
df_two['source']="predictions"

dup_cols=list(df_one.columns)
dup_cols.remove("source")
dup_cols = ['spread_play', 'total_play', 'spread_lock', 'total_lock'] #overide so difference is only pick/play differences

# Find the differences between the two DataFrames
df_all_diff = pd.concat([df_one, df_two]).drop_duplicates(keep=False, subset=dup_cols).sort_values(by=['date_time', 'home_team', 'source']).reset_index(drop=True) 
df_picks_diff = df_all_diff.drop_duplicates(keep=False, subset=['game_id', 'spread_play', 'total_play']) 
df_plays_diff = df_all_diff.drop_duplicates(keep=False, subset=['game_id', 'spread_lock', 'total_lock'])  


pick_changes_num = int(len(df_picks_diff)/2)
pick_changes_games = df_picks_diff.sort_values(['source','date_time','home_team'])['game_id'].values[:pick_changes_num]
play_changes_num = int(len(df_plays_diff)/2)
play_changes_games = df_plays_diff.sort_values(['source','date_time','home_team'])['game_id'].values[:play_changes_num]


print(f"{pick_changes_num} picks changed.")
print(pick_changes_games, "\n")
print(f"{play_changes_num} plays changed")
print(play_changes_games)
df_all_diff

# %% [markdown]
# ### Display plays

# %%
print_plays(plays)
plays[['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'away_score_pred', 
          'spread_pred', 'spread_line',  'spread_play', 'spread_win_prob', 'spread_lock', 'total_pred', 'total_line', 'total_play', 'total_win_prob', 'total_lock']]

# %% [markdown]
# # Write picks to DB

# %%

def write_dynamo_update(result: dict, update_table):
    # Convert all floats in the result dictionary to Decimal
    result = convert_floats_to_decimal(result)

    update_table.put_item(
        Item={
            'year_week': result['year_week'],
            'write_time': result['write_time'],
            'week': str(result['week']),
            'season': result['season'],
            'environment': result['environment'],
            'client_name': result['client_name'],
            'runtime': result['runtime'],
            'pick_changes': result['pick_changes'],
            'pick_changes_games': result['pick_changes_games'],
            'play_changes': result['play_changes'],
            'play_changes_games': result['play_changes_games'],
            'updates_skipped': result['updates_skipped'],
            'picks_num': result['picks_num'],
            'difference_df': result['difference_df'],
            'picks_df': result['picks_df']
        }
    )
    return result['write_time']


def write_dynamo_picks(df, picks_table, write_time):
    for index, row in df.iterrows():
        picks_table.put_item(
            Item={
                'year_week': row['year_week'],
                'game_id': str(row['game_id']),
                'season': row['season'],
                'week': str(row['week']),  
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score_pred': convert_to_decimal(row['home_score_pred']),
                'away_score_pred': convert_to_decimal(row['away_score_pred']),
                'spread_pred': convert_to_decimal(row['spread_pred']),
                'spread_line': convert_to_decimal(row['spread_line']),
                'spread_play': row['spread_play'],
                'spread_win_prob': convert_to_decimal(row['spread_win_prob']),
                'spread_lock': row['spread_lock'],
                'total_pred': convert_to_decimal(row['total_pred']),
                'total_line': convert_to_decimal(row['total_line']),
                'total_play': row['total_play'],
                'total_win_prob': convert_to_decimal(row['total_win_prob']),
                'total_lock': row['total_lock'],
                'date_time': row['date_time'],
                'write_time': write_time
            }
        )

def write_dynamo_results(df, historical_results_table):
    for index, row in df.iterrows():
        historical_results_table.put_item(
            Item={
                'year_week': row['year_week'],
                'game_id': str(row['game_id']),
                'season': row['season'],
                'week': str(row['week']),
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score_pred': convert_to_decimal(row['home_score_pred']),
                'away_score_pred': convert_to_decimal(row['away_score_pred']),
                'home_score': convert_to_decimal(row['home_score']),
                'away_score': convert_to_decimal(row['away_score']),
                'spread_pred': convert_to_decimal(row['spread_pred']),
                'spread_line': convert_to_decimal(row['spread_line']),
                'true_spread': convert_to_decimal(row['true_spread']),
                'spread_play': row['spread_play'],
                'spread_win_prob': convert_to_decimal(row['spread_win_prob']),
                'spread_lock': row['spread_lock'],
                'correct_spread_play': row['correct_spread_play'],
                'spread_win': None if math.isnan(row['spread_win']) else convert_to_decimal(row['spread_win']),
                'total_pred': convert_to_decimal(row['total_pred']),
                'total_line': convert_to_decimal(row['total_line']),
                'true_total': convert_to_decimal(row['true_total']),
                'total_play': row['total_play'],
                'total_win_prob': convert_to_decimal(row['total_win_prob']),
                'correct_total_play': row['correct_total_play'],
                'total_win': None if math.isnan(row['total_win']) else convert_to_decimal(row['total_win']),
                'total_lock': row['total_lock'],
                'date_time': row['date_time'],
                'write_time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
        )

# %%
def convert_picks(df):
    
    converted_picks = [
        convert_floats_to_decimal(dict(PickResponse(**{**item, 'week': str(item.get('week'))})))
        for item in df.to_dict(orient='records')
    ]

    return converted_picks

write_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
end_time = time.time()
database_updated = False

update_result = {
    "year_week": current_year_week,
    "write_time": write_time,
    "week": current_week,
    "season": current_year,
    "environment": environment,
    "client_name": client_name,
    "runtime": end_time - start_time,
    "pick_changes": pick_changes_num,
    "pick_changes_games": list(pick_changes_games),
    "play_changes": play_changes_num,
    "play_changes_games": list(play_changes_games),
    "updates_skipped": num_replacements,
    "picks_num": len(plays),
    "difference_df": list(convert_picks(df_all_diff)),
    "picks_df": list(convert_picks(plays)),
    "database_updated": database_updated
    }

# Write this to a JSON file
with open("/tmp/variable_output.json", "w") as f:
    json.dump({key: value for key, value in update_result.items() if key not in {"difference_df", "picks_df"}}, f)

if client_name in ["aws"]: #,"notebook"
    
    historical_picks_table = dynamodb.Table('nfl_expected_points_historical_picks')
    pick_updates_table = dynamodb.Table('nfl_expected_points_pick_updates')

    if len(plays)>5:
        write_time = write_dynamo_update(update_result, pick_updates_table)
        write_dynamo_picks(plays, historical_picks_table, write_time)
        delete_dynamo_enteries(picks_table, {'PartitionKey': 'year_week', 'SortKey': 'game_id'})
        write_dynamo_picks(plays, picks_table, write_time)
        database_updated = True
        print("updated")

#overwrite with new database_updated value if picks were written
with open("/tmp/variable_output.json", "r") as f:
    update_result_response = json.load(f)

# Update the 'database_updated' key
update_result_response['database_updated'] = database_updated

# Write the updated result back to the file
with open("/tmp/variable_output.json", "w") as f:
    json.dump(update_result_response, f)

# %%
sys.exit(0)

# %% [markdown]
# ## Update model record

# %%
historical_picks_table = dynamodb.Table('nfl_expected_points_historical_picks')
historical_results_table = dynamodb.Table('nfl_expected_points_results')

historical_picks = historical_picks_table.scan().get('Items',[])
historical_results = historical_results_table.scan().get('Items',[])

picks_game_ids = {item['game_id'] for item in historical_picks}
results_game_ids = {item['game_id'] for item in historical_results}

no_results = list(picks_game_ids - results_game_ids)
print(f"Games not in picks that are in results (should be 0): {len(results_game_ids - picks_game_ids)}")
print(f"{len(no_results)} picks need to be updated")

# %%
picks_to_update = pd.DataFrame([item for item in historical_picks if item['game_id'] in no_results])
#picks_to_update = pd.DataFrame([item for item in historical_picks])

scores = {
    row['game_id']: {'home_score': row['home_score'], 'away_score': row['away_score']}
    for _, row in df[['game_id', 'home_score', 'away_score']].iterrows()
    if pd.notna(row['home_score']) and pd.notna(row['away_score'])
}

picks_to_update = picks_to_update[picks_to_update['game_id'].isin(scores.keys())].copy()

picks_to_update['home_score'] = picks_to_update['game_id'].apply(lambda x: scores[x]['home_score'])
picks_to_update['away_score'] = picks_to_update['game_id'].apply(lambda x: scores[x]['away_score'])


picks_to_update = calculate_wins(picks_to_update)
check = get_result_stats(picks_to_update, Verbose = True)

print(f"Updating {len(picks_to_update)} picks")

# %%
print(len(picks_to_update[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',  'home_score_pred', 'away_score_pred',
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_win_prob' , 'spread_lock', 
          'correct_spread_play', 'spread_win', 'total_pred', 'total_line', 'true_total', 'total_play', 'total_win_prob', 
          'total_lock', 'correct_total_play', 'total_win', 'year_week','game_id','date_time']]))

picks_to_update[['home_team', 'away_team', 'home_score', 'away_score', 'home_score_pred', 'away_score_pred',
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_lock', 
          'correct_spread_play', 'spread_win', 'total_pred', 'total_line', 'true_total', 'total_play', 
          'total_lock', 'correct_total_play', 'total_win']]

# need to add away score and away score preds

# %%
write_dynamo_results(picks_to_update, historical_results_table)



## # # #safety = dynamo_add_column(historical_results_table, df, 'away_score_pred', write_dynamo_results, delete_dynamo_enteries, {'PartitionKey': 'year_week', 'SortKey': 'game_id'})

# %% [markdown]
# # Results

# %%
historical_results_table = dynamodb.Table('nfl_expected_points_results')
results_df = pd.DataFrame(historical_results_table.scan()['Items'])

# %%
#esults_df=results_df[results_df['week']=='6']
catch = get_result_stats(results_df, Verbose = True)

# %% [markdown]
# # Model retroactive record

# %%
def season_stats(df, current_week, year, score_pipeline):
    season_results = {}
    results_list = []
    for i in range(1,current_week+1):

        train_df = df[((df['season'] == year) & (df['week'] < i)) | (df['season'] < year)]
        train_df = train_df.dropna(subset = 'home_score', inplace=False)

        X = train_df[input_features]
        y = train_df[targets]   

        #do and eval split to fit classifiers
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        eval_results  = fit_eval(train_df, X_train, X_test, y_train, y_test, score_pipeline)
        spread_clf_temp, total_clf_temp = fit_classifiers(eval_results, spread_class_features, total_class_features)

        score_model_temp =  fit_score_model(X, y, score_pipeline)

        #predict this week
        test_df = df[(df['season'] == year) & (df['week'] == i)]
        X_test = test_df.loc[:, features]

        away_scores = score_model_temp.predict_scores(X_test[features], 'away')
        home_scores = score_model_temp.predict_scores(X_test[features], 'home')
        test_df['away_score_pred'] = away_scores
        test_df['home_score_pred'] = home_scores

        plays = scores_to_bets(test_df)
        plays['spread_win_prob'] = win_probability(plays, classifier = spread_clf, features = spread_class_features)
        plays['total_win_prob'] = win_probability(plays, classifier = total_clf, features = total_class_features)
        plays = determine_plays(plays)


        #get results
        results = calculate_wins(plays)
        data = get_result_stats(results, Verbose = False)

        

        season_results[i] = data
        season_results[i]['df'] = results

        results_list.append(results)
        
        print(f"Week {i} done")
        
    combined_results = pd.concat(results_list, ignore_index=True)    
    return season_results, combined_results

# %%
season_results,season_results_df = season_stats(df, current_week = 18, year = 2023, score_pipeline=score_pipeline)

total_spread_wins = sum(val['spread_wins'] for val in season_results.values())
total_total_wins = sum(val['total_wins'] for val in season_results.values())
total_games = sum(val['predicted_games'] for val in season_results.values())

total_spread_ratio = 100 * total_spread_wins / total_games
total_total_ratio = 100 * total_total_wins / total_games

total_spread_lock_wins = sum(val['spread_lock_wins'] for val in season_results.values())
total_total_lock_wins = sum(val['total_lock_wins'] for val in season_results.values())
total_spread_lock_games = sum(val['spread_lock_predictions'] for val in season_results.values())
total_total_lock_games = sum(val['total_lock_predictions'] for val in season_results.values())


total_spread_lock_ratio = 100* total_spread_lock_wins / total_spread_lock_games 
total_total_lock_ratio = 100 * total_total_lock_wins / total_total_lock_games

print(f"Season spread win percentage: {total_spread_ratio:.2f}%")
print(f"Season total win percentage: {total_total_ratio:.2f}%\n")
print(f"Season spread lock win percentage: {total_spread_lock_ratio:.2f}%")
print(f"Season total lock win percentage: {total_total_lock_ratio:.2f}%")

# %%
spread_clf.predict_proba(dd)

# %%
dd

# %%
season_results[1]['df']

# %%
win_probability(season_results[1]['df'], classifier = total_clf, features = total_class_features)

# %%
season_results[1]['df'][spread_class_features]

# %%
get_result_stats(season_results_df, Verbose=True)

# %%
season_results_df[['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score',
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_win_prob' , 'spread_lock', 
          'correct_spread_play', 'spread_win', 'total_pred', 'total_line', 'true_total', 'total_play', 'total_win_prob', 
          'total_lock', 'correct_total_play', 'total_win', 'year_week','game_id','date_time']]

# %%
check_week = 2
print(f"Week {check_week} spread win percentage: {season_results[check_week]['spread_perc']:.2f}%")
print(f"Week {check_week} total win percentage: {season_results[check_week]['total_perc']:.2f}%\n")
print(f"Week {check_week} spread lock win percentage: {season_results[check_week]['spread_lock_perc']:.2f}%")
print(f"Week {check_week} total lock win percentage: {season_results[check_week]['total_lock_perc']:.2f}%")

# %%
season_results[4]['df'][['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score',
          'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_win_prob' , 'correct_spread_play', 'spread_win',
          'total_pred', 'total_line', 'true_total', 'total_play', 'total_win_prob', 'correct_total_play', 'total_win']]

# %%
season_results[check_week]['spread_locks_df'][['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score', 'away_score_pred', 'away_score', 
           'spread_pred', 'spread_line', 'true_spread', 'spread_play', 'spread_win_prob', 'correct_spread_play', 'spread_win']]

# %%
season_results[check_week]['total_locks_df'][['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'home_score', 'away_score_pred', 'away_score', 
          'total_pred', 'total_line', 'true_total', 'total_play', 'total_win_prob', 'correct_total_play', 'total_win', 'spread_play']]

# %%
# season_results[3]['df'][['season', 'week', 'home_team', 'away_team', 'home_score_pred', 'away_score_pred', 
#           'spread_pred', 'spread_line',  'spread_play', 'spread_win_prob',
#           'total_pred', 'total_line', 'total_play', 'total_win_prob', 
#           'spread_win','total_win']].to_csv('my_data.csv', index=False)

# %% [markdown]
# # Supplemental analytics and vizualizations

# %%
team = 'DAL'
tm = epa.loc[epa['team'] == team, :].assign(
    season_week = lambda x: 'w' + x.week.astype(str) + ' (' + x.season.astype(str) + ')'
).set_index('season_week')

fig, ax = plt.subplots()

loc = plticker.MultipleLocator(base=16) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.tick_params(axis='x', rotation=75) #rotate the x-axis labels a bit

ax.plot(tm['epa_shifted_passing_offense'], lw=1, alpha=0.5)
ax.plot(tm['ewma_dynamic_window_passing_offense'], lw=2)
ax.plot(tm['ewma_passing_offense'], lw=2);
plt.axhline(y=0, color='red', lw=1.5, alpha=0.5)

ax.legend(['Passing EPA', 'EWMA on EPA with dynamic window', 'Static 10 EWMA on EPA'])
ax.set_title(f'{team} Passing EPA per play')
plt.show()

# %% [markdown]
# ### 2023 Season - EPA

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
import numpy as np


# Assuming nflreadr::load_pbp(2020) loads a pandas DataFrame
pbp_viz = pbp[(pbp['season'] == 2023) & (pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & ((pbp['rush'] == 1) | (pbp['pass'] == 1))] #(pbp['rz'] == 1) &

offense = pbp_viz.groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa'})
pass_offense = pbp_viz[pbp_viz['pass_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_pass_epa'})
run_offense = pbp_viz[pbp_viz['rush_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_run_epa'})
rz_offense =pbp_viz[pbp_viz['rz']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'rz_off_epa'})
non_rz_offense=pbp_viz[pbp_viz['rz']==0].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'non_rz_off_epa'})



defense = pbp_viz.groupby('defteam').agg({'epa': np.mean}).reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa'})

combined = pd.merge(offense, defense, on='team')

qbs = pbp_viz[(pbp_viz['pass'] == 1) | (pbp_viz['rush'] == 1) & pbp_viz['down'].isin(range(1, 5))].groupby('id').agg({
    'name': 'first',
    'posteam': 'last',
    'id': 'count',
    'qb_epa': np.mean
}).rename(columns={'id': 'plays', 'posteam': 'team'}).query('plays > 200').nlargest(10, 'qb_epa')

# Assuming that 'team_logos' is a dictionary with team names as keys and URLs/paths of logos as values
team_logos = teams.set_index('team_abbr')['team_logo_espn'].to_dict()

fig, ax = plt.subplots()
ax.scatter(combined['off_epa'], combined['def_epa'])

for x, y, team in zip(combined['off_epa'], combined['def_epa'], combined['team']):

    with urllib.request.urlopen(team_logos[team]) as url:
        img = Image.open(url)
        img = np.array(img)
    
    # Adjust zoom size only for Jets and Packers
    if team in ['NYJ', 'GB']:  # Adjust based on the correct team abbreviation in your dataset
        zoom_factor = 0.007  # Smaller size for Jets and Packers
    else:
        zoom_factor = 0.05  # Default size for other teams
    
    # Create offset image with the appropriate zoom
    oi = OffsetImage(img, zoom=zoom_factor)
    # Create annotation box
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    # Add annotation box to the axes
    ax.add_artist(ab)
    
ax.set_xlabel('Offensive EPA/play')
ax.set_ylabel('Defensive EPA/play')
ax.set_title('2023 NFL EPA per Play')
ax.grid(alpha=0.2)
ax.invert_yaxis()
plt.show()

# %% [markdown]
# ### Model sees - Offense EPA

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
import numpy as np


m = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['home_team', 'away_team'], value_name='team').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'team']]
m1 = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['ewma_dynamic_window_rushing_offense_home', 'ewma_dynamic_window_rushing_offense_away'], value_name='ewma_rushing_offense').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'ewma_rushing_offense']]
m2 = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['ewma_dynamic_window_passing_offense_home', 'ewma_dynamic_window_passing_offense_away'], value_name='ewma_passing_offense').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'ewma_passing_offense']]
combined = pd.concat([m, m1['ewma_rushing_offense'], m2['ewma_passing_offense']], axis=1)

fig, ax = plt.subplots()
ax.scatter(combined['ewma_rushing_offense'], combined['ewma_passing_offense'])

# Add logos to the plot
for x, y, team in zip(combined['ewma_rushing_offense'], combined['ewma_passing_offense'], combined['team']):
    # Read image from URL
    with urllib.request.urlopen(team_logos[team]) as url:
        img = Image.open(url)
        img = np.array(img)
    
    # Adjust zoom size only for Jets and Packers
    if team in ['NYJ', 'GB']:  # Adjust based on the correct team abbreviation in your dataset
        zoom_factor = 0.008  # Smaller size for Jets and Packers
    else:
        zoom_factor = 0.05  # Default size for other teams
    
    # Create offset image with the appropriate zoom
    oi = OffsetImage(img, zoom=zoom_factor)
    # Create annotation box
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    # Add annotation box to the axes
    ax.add_artist(ab)
    
ax.set_xlabel('Rush Offense EPA/play')
ax.set_ylabel('Pass Offense EPA/play')
ax.set_title('NFL Exponentially Weighted Average Offensive EPA per Play')
ax.grid(alpha=0.2)
#ax.invert_yaxis()
plt.show()

# %% [markdown]
# ### Model sees - Defense EPA

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
import numpy as np


m = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['home_team', 'away_team'], value_name='team').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'team']]
m1 = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['ewma_dynamic_window_rushing_defense_home', 'ewma_dynamic_window_rushing_defense_away'], value_name='ewma_rushing_defense').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'ewma_rushing_defense']]
m2 = pd.melt(this_week, id_vars=['season', 'week'], value_vars=['ewma_dynamic_window_passing_defense_home', 'ewma_dynamic_window_passing_defense_away'], value_name='ewma_passing_defense').drop('variable', axis=1).reset_index(drop=True)[['season', 'week', 'ewma_passing_defense']]
combined = pd.concat([m, m1['ewma_rushing_defense'], m2['ewma_passing_defense']], axis=1)
fig, ax = plt.subplots()
ax.scatter(combined['ewma_rushing_defense'], combined['ewma_passing_defense'])

# Add logos to the plot
for x, y, team in zip(combined['ewma_rushing_defense'], combined['ewma_passing_defense'], combined['team']):
    # Read image from URL
    with urllib.request.urlopen(team_logos[team]) as url:
        img = Image.open(url)
        img = np.array(img)
    
    # Adjust zoom size only for Jets and Packers
    if team in ['NYJ', 'GB', 'PHI']:  # Adjust based on the correct team abbreviation in your dataset
        zoom_factor = 0.008  # Smaller size for Jets and Packers
    else:
        zoom_factor = 0.05  # Default size for other teams
    
    # Create offset image with the appropriate zoom
    oi = OffsetImage(img, zoom=zoom_factor)
    # Create annotation box
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    # Add annotation box to the axes
    ax.add_artist(ab)
    
ax.set_xlabel('Rush Defense EPA/play')
ax.set_ylabel('Pass Defense EPA/play')
ax.set_title('NFL Exponentially Weighted Average Defensive EPA per Play')
ax.grid(alpha=0.2)
ax.invert_yaxis()
ax.invert_xaxis()
plt.show()

# %% [markdown]
# ### 2023 Season - RZ Offense and Defense EPA

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
import numpy as np


# Assuming nflreadr::load_pbp(2020) loads a pandas DataFrame
pbp_viz = pbp[(pbp['season'] == 2023) & (pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & ((pbp['rush'] == 1) | (pbp['pass'] == 1))] #(pbp['rz'] == 1) &

offense = pbp_viz.groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa'})
pass_offense = pbp_viz[pbp_viz['pass_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_pass_epa'})
run_offense = pbp_viz[pbp_viz['rush_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_run_epa'})
rz_offense =pbp_viz[pbp_viz['rz']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'rz_off_epa'})
non_rz_offense=pbp_viz[pbp_viz['rz']==0].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'non_rz_off_epa'})



defense = pbp_viz.groupby('defteam').agg({'epa': np.mean}).reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa'})

combined = pd.merge(rz_offense, non_rz_offense, on='team')

qbs = pbp_viz[(pbp_viz['pass'] == 1) | (pbp_viz['rush'] == 1) & pbp_viz['down'].isin(range(1, 5))].groupby('id').agg({
    'name': 'first',
    'posteam': 'last',
    'id': 'count',
    'qb_epa': np.mean
}).rename(columns={'id': 'plays', 'posteam': 'team'}).query('plays > 200').nlargest(10, 'qb_epa')

# Assuming that 'team_logos' is a dictionary with team names as keys and URLs/paths of logos as values
team_logos = teams.set_index('team_abbr')['team_logo_espn'].to_dict()

fig, ax = plt.subplots()
ax.scatter(combined['non_rz_off_epa'], combined['rz_off_epa'])

# Add logos to the plot
for x, y, team in zip(combined['non_rz_off_epa'], combined['rz_off_epa'], combined['team']):
    # Read image from URL
    with urllib.request.urlopen(team_logos[team]) as url:
        img = Image.open(url)
        img = np.array(img)
    # Create offset image
    oi = OffsetImage(img, zoom=0.05)
    # Create annotation box
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    # Add annotation box to the axes
    ax.add_artist(ab)
    
ax.set_xlabel('Offense EPA/play (Non Red Zone)')
ax.set_ylabel('Offense EPA/play (Red Zone)')
ax.set_title('2023 NFL Offensive EPA per Play')
ax.grid(alpha=0.2)
#ax.invert_yaxis()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
import numpy as np


# Assuming nflreadr::load_pbp(2020) loads a pandas DataFrame
pbp_viz = pbp[(pbp['rz'] == 1) & (pbp['season'] == 2023) & (pbp['season_type'] == 'REG') & (pbp['posteam'].notna()) & ((pbp['rush'] == 1) | (pbp['pass'] == 1))] #(pbp['rz'] == 1) &

offense = pbp_viz.groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa'})
pass_offense = pbp_viz[pbp_viz['pass_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_pass_epa'})
run_offense = pbp_viz[pbp_viz['rush_attempt']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_run_epa'})
rz_offense =pbp_viz[pbp_viz['rz']==1].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'rz_off_epa'})
non_rz_offense=pbp_viz[pbp_viz['rz']==0].groupby('posteam').agg({'epa': np.mean}).reset_index().rename(columns={'posteam': 'team', 'epa': 'non_rz_off_epa'})



defense = pbp_viz.groupby('defteam').agg({'epa': np.mean}).reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa'})

combined = pd.merge(run_offense, pass_offense, on='team')

qbs = pbp_viz[(pbp_viz['pass'] == 1) | (pbp_viz['rush'] == 1) & pbp_viz['down'].isin(range(1, 5))].groupby('id').agg({
    'name': 'first',
    'posteam': 'last',
    'id': 'count',
    'qb_epa': np.mean
}).rename(columns={'id': 'plays', 'posteam': 'team'}).query('plays > 200').nlargest(10, 'qb_epa')

# Assuming that 'team_logos' is a dictionary with team names as keys and URLs/paths of logos as values
team_logos = teams.set_index('team_abbr')['team_logo_espn'].to_dict()

fig, ax = plt.subplots()
ax.scatter(combined['off_run_epa'], combined['off_pass_epa'])

# Add logos to the plot
for x, y, team in zip(combined['off_run_epa'], combined['off_pass_epa'], combined['team']):
    # Read image from URL
    with urllib.request.urlopen(team_logos[team]) as url:
        img = Image.open(url)
        img = np.array(img)
    # Create offset image
    oi = OffsetImage(img, zoom=0.05)
    # Create annotation box
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    # Add annotation box to the axes
    ax.add_artist(ab)
    
ax.set_xlabel('Run Offense EPA/play (Non Red Zone)')
ax.set_ylabel('Pass Offense EPA/play (Red Zone)')
ax.set_title('2023 NFL Offensive EPA per Play (Red Zone)')
ax.grid(alpha=0.2)
#ax.invert_yaxis()
plt.show()

# %%


# %%
data_1 = [
    {
        "date": "2024-10-16",
        "player_name": "Andrew Nembhard",
        "sportsbook": "BetRivers",
        "odds": 1300.0,
        "units": 80.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Andrew Wiggins",
        "sportsbook": "BetMGM",
        "odds": 900.0,
        "units": 108.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Austin Reaves",
        "sportsbook": "BetRivers",
        "odds": 1200.0,
        "units": 59.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Caleb Houstan",
        "sportsbook": "BetMGM",
        "odds": 1400.0,
        "units": 58.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Cody Martin",
        "sportsbook": "BetRivers",
        "odds": 1200.0,
        "units": 53.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "D'Angelo Russell",
        "sportsbook": "BetRivers",
        "odds": 1050.0,
        "units": 51.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Gary Harris",
        "sportsbook": "BetMGM",
        "odds": 1200.0,
        "units": 75.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Jalen Suggs",
        "sportsbook": "DraftKings",
        "odds": 1100.0,
        "units": 92.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Julian Champagnie",
        "sportsbook": "BetMGM",
        "odds": 1250.0,
        "units": 88.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Kris Murray",
        "sportsbook": "BetMGM",
        "odds": 1150.0,
        "units": 82.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Max Strus",
        "sportsbook": "BetMGM",
        "odds": 950.0,
        "units": 64.0,
        "bankroll": 5000
    },
    {
        "date": "2024-10-16",
        "player_name": "Wendell Carter Jr.",
        "sportsbook": "DraftKings",
        "odds": 800.0,
        "units": 53.0,
        "bankroll": 5000
    }
]

# %%



