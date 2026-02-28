import pandas as pd
import numpy as np

from collections import defaultdict, deque
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# -----------------------------
# CONFIG
# -----------------------------
PATH = "whl_2025_ordered.csv"   # your file
TRAIN_FRAC = 0.80               # chronological split
ROLL_N = 10                     # rolling window for "form"
MIN_HISTORY = 3                 # if fewer than this, fall back to league priors
SEED = 42

# -----------------------------
# 0) Load
# -----------------------------
df = pd.read_csv(PATH)

need = [
    "game_id","record_id","home_team","away_team","went_ot",
    "home_off_line","home_def_pairing","away_off_line","away_def_pairing",
    "home_goalie","away_goalie","toi",
    "home_assists","home_shots","home_xg","home_max_xg","home_goals",
    "away_assists","away_shots","away_xg","away_max_xg","away_goals",
    "home_penalties_committed","home_penalty_minutes",
    "away_penalties_committed","away_penalty_minutes",
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# preserve chronological order as it appears in CSV
first_idx = (
    df.reset_index()
      .groupby(["game_id","home_team","away_team"], as_index=False)["index"]
      .min()
      .rename(columns={"index":"first_row_index"})
)

def mode_or_first(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]

# -----------------------------
# 1) Aggregate to game level
# -----------------------------
agg_num = {
    "toi": "sum",
    "home_assists": "sum",
    "home_shots": "sum",
    "home_xg": "sum",
    "home_max_xg": "max",
    "home_goals": "sum",
    "away_assists": "sum",
    "away_shots": "sum",
    "away_xg": "sum",
    "away_max_xg": "max",
    "away_goals": "sum",
    "home_penalties_committed": "sum",
    "home_penalty_minutes": "sum",
    "away_penalties_committed": "sum",
    "away_penalty_minutes": "sum",
}

g_num = (
    df.groupby(["game_id","home_team","away_team"], as_index=False)
      .agg(**{k: (k, v) for k, v in agg_num.items()})
)

g_cat = (
    df.groupby(["game_id","home_team","away_team"], as_index=False)
      .agg(
          went_ot=("went_ot", "max"),  # postgame; will not be used directly as feature for same game
          home_off_line=("home_off_line", mode_or_first),
          home_def_pairing=("home_def_pairing", mode_or_first),
          away_off_line=("away_off_line", mode_or_first),
          away_def_pairing=("away_def_pairing", mode_or_first),
          home_goalie=("home_goalie", mode_or_first),
          away_goalie=("away_goalie", mode_or_first),
      )
)

g = (
    g_num.merge(g_cat, on=["game_id","home_team","away_team"], how="left")
         .merge(first_idx, on=["game_id","home_team","away_team"], how="left")
         .sort_values("first_row_index")
         .reset_index(drop=True)
)

# target
g["home_win"] = (g["home_goals"] > g["away_goals"]).astype(int)

# -----------------------------
# 2) Leakage-safe rolling feature builder
# -----------------------------
TEAM_METRICS = ["gf","ga","sf","sa","xgf","xga","pim","win","ot"]
GOALIE_METRICS = ["sa","ga","xga"]

def make_team_state(roll_n: int):
    d = {}
    d["games"] = 0
    d["tot"] = dict((m, 0.0) for m in TEAM_METRICS)
    d["roll"] = dict((m, deque(maxlen=roll_n)) for m in TEAM_METRICS)
    return d

def make_goalie_state(roll_n: int):
    d = {}
    d["games"] = 0
    d["tot"] = dict((m, 0.0) for m in GOALIE_METRICS)
    d["roll"] = dict((m, deque(maxlen=roll_n)) for m in GOALIE_METRICS)
    return d

team_state = defaultdict(lambda: make_team_state(ROLL_N))
goalie_state = defaultdict(lambda: make_goalie_state(ROLL_N))

league_games = 0
league_tot = dict((m, 0.0) for m in TEAM_METRICS)

def safe_div(a, b, default=0.0):
    return float(a / b) if b else float(default)

def mean_deque(dq, default=0.0):
    return float(np.mean(dq)) if len(dq) else float(default)

def team_features(team: str, prefix: str):
    st = team_state[team]
    # league priors (so early-season teams aren’t all-zero)
    lg = max(league_games, 1)

    pri_gf = league_tot["gf"] / lg
    pri_ga = league_tot["ga"] / lg
    pri_sf = league_tot["sf"] / lg
    pri_sa = league_tot["sa"] / lg
    pri_xgf = league_tot["xgf"] / lg
    pri_xga = league_tot["xga"] / lg
    pri_pim = league_tot["pim"] / lg
    pri_win = league_tot["win"] / lg
    pri_ot = league_tot["ot"] / lg

    use_priors = st["games"] < MIN_HISTORY

    # rolling (form)
    r_gf = mean_deque(st["roll"]["gf"], pri_gf if use_priors else 0.0)
    r_ga = mean_deque(st["roll"]["ga"], pri_ga if use_priors else 0.0)
    r_sf = mean_deque(st["roll"]["sf"], pri_sf if use_priors else 0.0)
    r_sa = mean_deque(st["roll"]["sa"], pri_sa if use_priors else 0.0)
    r_xgf = mean_deque(st["roll"]["xgf"], pri_xgf if use_priors else 0.0)
    r_xga = mean_deque(st["roll"]["xga"], pri_xga if use_priors else 0.0)
    r_pim = mean_deque(st["roll"]["pim"], pri_pim if use_priors else 0.0)
    r_win = mean_deque(st["roll"]["win"], pri_win if use_priors else 0.0)
    r_ot = mean_deque(st["roll"]["ot"], pri_ot if use_priors else 0.0)

    # season-to-date
    gms = max(st["games"], 1)
    s_gf = st["tot"]["gf"] / gms if not use_priors else pri_gf
    s_ga = st["tot"]["ga"] / gms if not use_priors else pri_ga
    s_sf = st["tot"]["sf"] / gms if not use_priors else pri_sf
    s_sa = st["tot"]["sa"] / gms if not use_priors else pri_sa
    s_xgf = st["tot"]["xgf"] / gms if not use_priors else pri_xgf
    s_xga = st["tot"]["xga"] / gms if not use_priors else pri_xga
    s_pim = st["tot"]["pim"] / gms if not use_priors else pri_pim
    s_win = st["tot"]["win"] / gms if not use_priors else pri_win
    s_ot = st["tot"]["ot"] / gms if not use_priors else pri_ot

    out = {
        f"{prefix}_games_played": float(st["games"]),
        f"{prefix}_roll_gf": r_gf,
        f"{prefix}_roll_ga": r_ga,
        f"{prefix}_roll_sf": r_sf,
        f"{prefix}_roll_sa": r_sa,
        f"{prefix}_roll_xgf": r_xgf,
        f"{prefix}_roll_xga": r_xga,
        f"{prefix}_roll_pim": r_pim,
        f"{prefix}_roll_winrate": r_win,
        f"{prefix}_roll_otrate": r_ot,
        f"{prefix}_s_gf": s_gf,
        f"{prefix}_s_ga": s_ga,
        f"{prefix}_s_sf": s_sf,
        f"{prefix}_s_sa": s_sa,
        f"{prefix}_s_xgf": s_xgf,
        f"{prefix}_s_xga": s_xga,
        f"{prefix}_s_pim": s_pim,
        f"{prefix}_s_winrate": s_win,
        f"{prefix}_s_otrate": s_ot,
        f"{prefix}_s_gd": s_gf - s_ga,
        f"{prefix}_s_sd": s_sf - s_sa,
        f"{prefix}_s_xgd": s_xgf - s_xga,
        f"{prefix}_roll_gd": r_gf - r_ga,
        f"{prefix}_roll_sd": r_sf - r_sa,
        f"{prefix}_roll_xgd": r_xgf - r_xga,
    }
    return out

def goalie_features(goalie: str, prefix: str):
    st = goalie_state[goalie]
    # goalie priors from league-ish: use league per-game against rates
    lg = max(league_games, 1)
    pri_sa = league_tot["sa"] / lg
    pri_ga = league_tot["ga"] / lg
    pri_xga = league_tot["xga"] / lg

    use_priors = st["games"] < MIN_HISTORY

    r_sa = mean_deque(st["roll"]["sa"], pri_sa if use_priors else 0.0)
    r_ga = mean_deque(st["roll"]["ga"], pri_ga if use_priors else 0.0)
    r_xga = mean_deque(st["roll"]["xga"], pri_xga if use_priors else 0.0)

    gms = max(st["games"], 1)
    s_sa = st["tot"]["sa"] / gms if not use_priors else pri_sa
    s_ga = st["tot"]["ga"] / gms if not use_priors else pri_ga
    s_xga = st["tot"]["xga"] / gms if not use_priors else pri_xga

    # save% proxy; and "goals - xg" proxy (lower is better)
    out = {
        f"{prefix}_games": float(st["games"]),
        f"{prefix}_roll_sv": 1.0 - safe_div(r_ga, r_sa, default=safe_div(pri_ga, pri_sa, default=0.0)),
        f"{prefix}_s_sv": 1.0 - safe_div(s_ga, s_sa, default=safe_div(pri_ga, pri_sa, default=0.0)),
        f"{prefix}_roll_g_minus_xg": r_ga - r_xga,
        f"{prefix}_s_g_minus_xg": s_ga - s_xga,
    }
    return out

rows = []
y = []

# categorical columns to keep (pregame identities)
cat_cols = [
    "home_team","away_team",
    "home_goalie","away_goalie",
    "home_off_line","away_off_line",
    "home_def_pairing","away_def_pairing",
]

for _, r in g.iterrows():
    ht = r["home_team"]
    at = r["away_team"]

    hg = r["home_goals"]
    ag = r["away_goals"]

    hs = r["home_shots"]
    a_s = r["away_shots"]

    hxg = r["home_xg"]
    axg = r["away_xg"]

    hpim = r["home_penalty_minutes"]
    apim = r["away_penalty_minutes"]

    went_ot = int(r["went_ot"]) if pd.notna(r["went_ot"]) else 0

    # build features from history BEFORE updating with this game
    f = {}
    f.update(team_features(ht, "home"))
    f.update(team_features(at, "away"))

    # differentials (home - away)
    for k in ["roll_gd","roll_sd","roll_xgd","roll_winrate","roll_otrate",
              "s_gd","s_sd","s_xgd","s_winrate","s_otrate","s_pim"]:
        f[f"diff_{k}"] = f[f"home_{k}"] - f[f"away_{k}"]

    # goalie history features (pre-game)
    f.update(goalie_features(r["home_goalie"], "hg"))
    f.update(goalie_features(r["away_goalie"], "ag"))
    f["diff_goalie_sv_roll"] = f["hg_roll_sv"] - f["ag_roll_sv"]
    f["diff_goalie_sv_s"] = f["hg_s_sv"] - f["ag_s_sv"]
    f["diff_goalie_g_minus_xg_roll"] = f["hg_roll_g_minus_xg"] - f["ag_roll_g_minus_xg"]
    f["diff_goalie_g_minus_xg_s"] = f["hg_s_g_minus_xg"] - f["ag_s_g_minus_xg"]

    # categorical IDs (as-is)
    f["home_team"] = ht
    f["away_team"] = at
    f["home_goalie"] = r["home_goalie"]
    f["away_goalie"] = r["away_goalie"]
    f["home_off_line"] = r["home_off_line"]
    f["away_off_line"] = r["away_off_line"]
    f["home_def_pairing"] = r["home_def_pairing"]
    f["away_def_pairing"] = r["away_def_pairing"]

    rows.append(f)
    y.append(int(r["home_win"]))

    # UPDATE states AFTER feature creation (no leakage)
    # team updates
    for team, gf, ga, sf, sa, xgf, xga, pim, win, ot in [
        (ht, hg, ag, hs, a_s, hxg, axg, hpim, int(hg > ag), went_ot),
        (at, ag, hg, a_s, hs, axg, hxg, apim, int(ag > hg), went_ot),
    ]:
        st = team_state[team]
        st["games"] += 1
        st["tot"]["gf"] += gf
        st["tot"]["ga"] += ga
        st["tot"]["sf"] += sf
        st["tot"]["sa"] += sa
        st["tot"]["xgf"] += xgf
        st["tot"]["xga"] += xga
        st["tot"]["pim"] += pim
        st["tot"]["win"] += win
        st["tot"]["ot"] += ot

        st["roll"]["gf"].append(gf)
        st["roll"]["ga"].append(ga)
        st["roll"]["sf"].append(sf)
        st["roll"]["sa"].append(sa)
        st["roll"]["xgf"].append(xgf)
        st["roll"]["xga"].append(xga)
        st["roll"]["pim"].append(pim)
        st["roll"]["win"].append(win)
        st["roll"]["ot"].append(ot)

    # goalie updates (goalie faces opponent shots/xg, allows opponent goals)
    hg_name = r["home_goalie"]
    ag_name = r["away_goalie"]

    # home goalie: faces away shots/xg, allows away goals
    gst = goalie_state[hg_name]
    gst["games"] += 1
    gst["tot"]["sa"] += a_s
    gst["tot"]["ga"] += ag
    gst["tot"]["xga"] += axg
    gst["roll"]["sa"].append(a_s)
    gst["roll"]["ga"].append(ag)
    gst["roll"]["xga"].append(axg)

    # away goalie: faces home shots/xg, allows home goals
    gst = goalie_state[ag_name]
    gst["games"] += 1
    gst["tot"]["sa"] += hs
    gst["tot"]["ga"] += hg
    gst["tot"]["xga"] += hxg
    gst["roll"]["sa"].append(hs)
    gst["roll"]["ga"].append(hg)
    gst["roll"]["xga"].append(hxg)

    # league updates (count both teams as two entries per game to keep priors consistent with per-team game rate)
    league_games += 2
    league_tot["gf"] += (hg + ag)
    league_tot["ga"] += (ag + hg)
    league_tot["sf"] += (hs + a_s)
    league_tot["sa"] += (a_s + hs)
    league_tot["xgf"] += (hxg + axg)
    league_tot["xga"] += (axg + hxg)
    league_tot["pim"] += (hpim + apim)
    league_tot["win"] += (int(hg > ag) + int(ag > hg))
    league_tot["ot"] += (went_ot + went_ot)

X = pd.DataFrame(rows)
y = np.array(y, dtype=int)

# cast categoricals (LightGBM handles category dtype)
for c in cat_cols:
    X[c] = X[c].astype("category")

# chronological split
N = len(X)
cut = int(TRAIN_FRAC * N)

X_train, y_train = X.iloc[:cut].copy(), y[:cut]
X_test, y_test = X.iloc[cut:].copy(), y[cut:]

# optional: use a small tail of train as validation for early stopping
val_cut = int(0.90 * len(X_train))
X_tr, y_tr = X_train.iloc[:val_cut], y_train[:val_cut]
X_val, y_val = X_train.iloc[val_cut:], y_train[val_cut:]

# -----------------------------
# 3) Train LightGBM
# -----------------------------
clf = LGBMClassifier(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
)

clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    categorical_feature=cat_cols,
    callbacks=[],
)

# If early stopping is available in your lightgbm build:
try:
    from lightgbm import early_stopping, log_evaluation
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=50)],
    )
except Exception:
    pass

# -----------------------------
# 4) Evaluate
# -----------------------------
probs = clf.predict_proba(X_test)[:, 1]
pred = (probs >= 0.5).astype(int)

acc = (pred == y_test).mean()
brier = np.mean((probs - y_test) ** 2)
ll = log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))

print("N games:", N, "Train:", cut, "Test:", N - cut)
print("TEST accuracy (%):", round(acc * 100, 2))
print("TEST Brier:", round(brier, 4))
print("TEST LogLoss:", round(ll, 4))

# -----------------------------
# 5) Feature importance (top 25)
# -----------------------------
imp = pd.DataFrame({
    "feature": X.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP 25 FEATURES:")
print(imp.head(25).to_string(index=False))