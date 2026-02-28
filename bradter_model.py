import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

PATH = "whl_2025_ordered.csv"
TEST_SIZE = 0.20
SEED = 42
L2_C = 1.0
MAX_ITER = 5000

df = pd.read_csv(PATH)

need = [
    "game_id","record_id","home_team","away_team",
    "home_off_line","home_def_pairing","away_off_line","away_def_pairing",
    "home_goalie","away_goalie",
    "home_goals","away_goals",
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

def mode_or_first(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]

first_idx = (
    df.reset_index()
      .groupby(["game_id","home_team","away_team"], as_index=False)["index"]
      .min()
      .rename(columns={"index":"first_row_index"})
)

g = (
    df.groupby(["game_id","home_team","away_team"], as_index=False)
      .agg(
          home_goals=("home_goals","sum"),
          away_goals=("away_goals","sum"),
          home_off_line=("home_off_line", mode_or_first),
          away_off_line=("away_off_line", mode_or_first),
          home_def_pairing=("home_def_pairing", mode_or_first),
          away_def_pairing=("away_def_pairing", mode_or_first),
          home_goalie=("home_goalie", mode_or_first),
          away_goalie=("away_goalie", mode_or_first),
      )
      .merge(first_idx, on=["game_id","home_team","away_team"], how="left")
      .sort_values("first_row_index")
      .reset_index(drop=True)
)

y = (g["home_goals"] > g["away_goals"]).astype(int).to_numpy()

def diff_onehot(home_col: pd.Series, away_col: pd.Series, prefix: str) -> pd.DataFrame:
    home_s = home_col.astype("string")
    away_s = away_col.astype("string")
    all_vals = pd.concat([home_s, away_s], ignore_index=True)

    all_dum = pd.get_dummies(all_vals, prefix=prefix, dummy_na=True).astype(np.int8)

    H = all_dum.iloc[:len(home_s)].reset_index(drop=True)
    A = all_dum.iloc[len(home_s):].reset_index(drop=True)

    D = H.sub(A)

    if D.shape[1] > 0:
        D = D.iloc[:, 1:]
    return D

X_team = diff_onehot(g["home_team"], g["away_team"], "team")
X_off  = diff_onehot(g["home_off_line"], g["away_off_line"], "off")
X_def  = diff_onehot(g["home_def_pairing"], g["away_def_pairing"], "def")
X_g    = diff_onehot(g["home_goalie"], g["away_goalie"], "g")

X = pd.concat([X_team, X_off, X_def, X_g], axis=1).astype(float)
X.insert(0, "home_adv", 1.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y
)

clf = LogisticRegression(
    penalty="l2",
    C=L2_C,
    solver="lbfgs",
    max_iter=MAX_ITER
)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
pred = (probs >= 0.5).astype(int)

acc = (pred == y_test).mean()
brier = np.mean((probs - y_test) ** 2)
ll = log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))

print("N games:", len(X), "Train:", len(X_train), "Test:", len(X_test))
print("TEST accuracy (%):", round(acc * 100, 2))
print("TEST Brier:", round(brier, 4))
print("TEST LogLoss:", round(ll, 4))

coef = pd.Series(clf.coef_[0], index=X.columns)
print("\nEstimated home advantage (log-odds):", round(float(coef["home_adv"]), 4))