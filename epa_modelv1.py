import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------
# CONFIG
# ---------------------------
PATH = "whl_2025_ordered.csv"          # or "whl_2025_ordered.csv"
ALPHA = 5.0                    # ridge (tune later)
USE_RECENCY = True
HALF_LIFE_GAMES = 300          # larger = slower decay; smaller = more recency
TRAIN_FRAC = 0.80              # chronological split

# ---------------------------
# 0) Load
# ---------------------------
df = pd.read_csv(PATH)

req = ["game_id", "home_team", "away_team", "home_xg", "away_xg", "home_goalie", "away_goalie"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------------------------
# 1) Aggregate to GAME level
#    (keeps original CSV order by first occurrence index)
# ---------------------------
first_idx = (
    df.reset_index()
      .groupby(["game_id", "home_team", "away_team"], as_index=False)["index"]
      .min()
      .rename(columns={"index": "first_row_index"})
)

games = (
    df.groupby(["game_id", "home_team", "away_team"], as_index=False)
      .agg(
          home_xg=("home_xg", "sum"),
          away_xg=("away_xg", "sum"),
          home_goalie=("home_goalie", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
          away_goalie=("away_goalie", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
      )
)

games = games.merge(first_idx, on=["game_id", "home_team", "away_team"], how="left")
games = games.sort_values("first_row_index").reset_index(drop=True)

games["xgd"] = games["home_xg"] - games["away_xg"]

# Optional: if goals exist, aggregate for winner accuracy
has_goals = ("home_goals" in df.columns) and ("away_goals" in df.columns)
if has_goals:
    goals_g = (
        df.groupby(["game_id","home_team","away_team"], as_index=False)
          .agg(home_goals=("home_goals","sum"), away_goals=("away_goals","sum"))
    ).merge(first_idx, on=["game_id","home_team","away_team"], how="left").sort_values("first_row_index").reset_index(drop=True)
    games = games.merge(goals_g[["game_id","home_team","away_team","home_goals","away_goals"]], on=["game_id","home_team","away_team"], how="left")

# ---------------------------
# 2) Build design matrix (TEAM + GOALIE), FRC-style differences
#    xGD = (team_home - team_away) + (goalie_home - goalie_away) + H + noise
# ---------------------------
teams = sorted(pd.unique(pd.concat([games["home_team"], games["away_team"]], ignore_index=True)))
goalies = sorted(pd.unique(pd.concat([games["home_goalie"], games["away_goalie"]], ignore_index=True).dropna()))

t2i = {t:i for i,t in enumerate(teams)}
g2i = {g:i for i,g in enumerate(goalies)}

n_games = len(games)
n_team = len(teams)
n_goalie = len(goalies)

A = np.zeros((n_games, n_team + n_goalie), dtype=float)

for i, r in games.iterrows():
    # teams
    A[i, t2i[r["home_team"]]] = 1.0
    A[i, t2i[r["away_team"]]] = -1.0

    # goalies (skip if missing)
    hg = r["home_goalie"]
    ag = r["away_goalie"]
    if pd.notna(hg) and hg in g2i:
        A[i, n_team + g2i[hg]] = 1.0
    if pd.notna(ag) and ag in g2i:
        A[i, n_team + g2i[ag]] = -1.0

y = games["xgd"].values

# ---------------------------
# 3) Recency weights (CSV already ordered)
#    exponential decay by game index distance from most recent
# ---------------------------
if USE_RECENCY:
    # weight = 0.5 ** (age / half_life)
    ages = (n_games - 1) - np.arange(n_games)
    w_all = 0.5 ** (ages / HALF_LIFE_GAMES)
else:
    w_all = np.ones(n_games, dtype=float)

# ---------------------------
# 4) Chronological train/test split (recommended)
# ---------------------------
cut = int(TRAIN_FRAC * n_games)

A_train, y_train, w_train = A[:cut], y[:cut], w_all[:cut]
A_test,  y_test,  w_test  = A[cut:], y[cut:], w_all[cut:]

model = Ridge(alpha=ALPHA, fit_intercept=True, random_state=0)
model.fit(A_train, y_train, sample_weight=w_train)

yhat = model.predict(A_test)

def wmae(y, yhat, w):
    return float(np.sum(w * np.abs(y - yhat)) / np.sum(w))

def wrmse(y, yhat, w):
    return float(np.sqrt(np.sum(w * (y - yhat) ** 2) / np.sum(w)))

print("TEST wMAE (xG diff per game):", round(wmae(y_test, yhat, w_test), 4))
print("TEST wRMSE (xG diff per game):", round(wrmse(y_test, yhat, w_test), 4))

if has_goals:
    pred_home_win = (yhat > 0).astype(int)
    actual_home_win = ((games.loc[cut:, "home_goals"] - games.loc[cut:, "away_goals"]) > 0).astype(int).values
    acc = float((pred_home_win == actual_home_win).mean())
    print("TEST winner accuracy (%):", round(acc * 100, 2))

# ---------------------------
# 5) Fit on ALL games for final ratings (still using recency weights)
# ---------------------------
model_full = Ridge(alpha=ALPHA, fit_intercept=True, random_state=0)
model_full.fit(A, y, sample_weight=w_all)

coef = model_full.coef_.copy()
home_adv = float(model_full.intercept_)   # H in xG diff per game

team_epa = coef[:n_team]
goalie_epa = coef[n_team:]

# Anchor for identifiability: center TEAM EPAs to mean 0
team_epa = team_epa - team_epa.mean()

team_ratings = pd.DataFrame({"team": teams, "team_epa_xgd": team_epa})
team_ratings = team_ratings.sort_values("team_epa_xgd", ascending=False).reset_index(drop=True)
team_ratings.insert(0, "rank", team_ratings.index + 1)

goalie_ratings = pd.DataFrame({"goalie": goalies, "goalie_epa_xgd": goalie_epa})
goalie_ratings = goalie_ratings.sort_values("goalie_epa_xgd", ascending=False).reset_index(drop=True)
goalie_ratings.insert(0, "rank", goalie_ratings.index + 1)

print("\nEstimated home advantage H (xG diff per game):", round(home_adv, 4))
print("\nTEAM POWER RANKINGS (recency-weighted, goalie-adjusted):")
print(team_ratings)

print("\nGOALIE EPA (xG-diff impact term in model):")
print(goalie_ratings)

# team_ratings.to_csv("whl_team_epa_recency_goalie.csv", index=False)
# goalie_ratings.to_csv("whl_goalie_epa_recency.csv", index=False)