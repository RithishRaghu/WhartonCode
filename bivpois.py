import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import skellam

EPS = 1e-12

def safe_exp(x):
    return np.exp(np.clip(x, -20.0, 20.0))

# -----------------------------
# CONFIG
# -----------------------------
PATH = "whl_2025_ordered.csv"
TRAIN_FRAC = 0.80
HALF_LIFE_GAMES = 300
L2 = 0.5

# -----------------------------
# 0) Load + aggregate to game level in CSV order
# -----------------------------
df = pd.read_csv(PATH)

need = ["game_id","home_team","away_team","home_goals","away_goals"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

first_idx = (
    df.reset_index()
      .groupby(["game_id","home_team","away_team"], as_index=False)["index"]
      .min()
      .rename(columns={"index":"first_row_index"})
)

g = (
    df.groupby(["game_id","home_team","away_team"], as_index=False)
      .agg(home_goals=("home_goals","sum"),
           away_goals=("away_goals","sum"))
).merge(first_idx, on=["game_id","home_team","away_team"], how="left")

g = g.sort_values("first_row_index").reset_index(drop=True)

teams = sorted(pd.unique(pd.concat([g["home_team"], g["away_team"]], ignore_index=True)))
T = len(teams)
t2i = {t:i for i,t in enumerate(teams)}

home_idx = g["home_team"].map(t2i).to_numpy()
away_idx = g["away_team"].map(t2i).to_numpy()
Gh = g["home_goals"].to_numpy().astype(int)
Ga = g["away_goals"].to_numpy().astype(int)

N = len(g)
cut = int(TRAIN_FRAC * N)

# -----------------------------
# 1) Recency weights (chronological)
# -----------------------------
ages = (cut - 1) - np.arange(cut)
w_train = 0.5 ** (np.clip(ages, 0, None) / HALF_LIFE_GAMES)

# -----------------------------
# 2) Bivariate Poisson log PMF
# -----------------------------
def bivpois_logpmf(g1, g2, l1, l2, l3):
    l1 = max(float(l1), EPS)
    l2 = max(float(l2), EPS)
    l3 = max(float(l3), EPS)

    m = min(g1, g2)
    ks = np.arange(m + 1)

    term = (
        (g1 - ks) * np.log(l1) +
        (g2 - ks) * np.log(l2) +
        ks * np.log(l3) -
        gammaln(g1 - ks + 1) -
        gammaln(g2 - ks + 1) -
        gammaln(ks + 1)
    )

    if np.all(~np.isfinite(term)):
        return -np.inf

    return -(l1 + l2 + l3) + logsumexp(term)

# -----------------------------
# 3) Negative log posterior
# -----------------------------
def neg_logpost(theta, home_i, away_i, gh, ga, w, l2_pen):
    alpha = theta[0]
    home_adv = theta[1]
    log_l3 = theta[2]

    atk = np.zeros(T)
    dfn = np.zeros(T)

    atk[:-1] = theta[3:3 + (T - 1)]
    dfn[:-1] = theta[3 + (T - 1): 3 + 2 * (T - 1)]

    atk[:-1] -= atk[:-1].mean()
    dfn[:-1] -= dfn[:-1].mean()

    l3 = safe_exp(log_l3)

    nll = 0.0
    for i in range(len(gh)):
        h = home_i[i]
        a = away_i[i]

        eta1 = alpha + home_adv + atk[h] - dfn[a]
        eta2 = alpha + atk[a] - dfn[h]

        l1 = safe_exp(eta1)
        l2 = safe_exp(eta2)

        lp = bivpois_logpmf(int(gh[i]), int(ga[i]), l1, l2, l3)
        nll -= w[i] * lp

    pen = l2_pen * (np.sum(atk[:-1] ** 2) + np.sum(dfn[:-1] ** 2))
    return nll + pen

# -----------------------------
# 4) Fit on TRAIN
# -----------------------------
alpha0 = np.log(max((Gh[:cut].mean() + Ga[:cut].mean()) / 2, 1e-6)) - np.log(1.0 + 0.2)
home0 = 0.10
logl30 = np.log(0.10)

theta0 = np.zeros(3 + 2 * (T - 1))
theta0[0] = alpha0
theta0[1] = home0
theta0[2] = logl30

bounds = []
bounds.append((-5.0, 5.0))
bounds.append((-1.0, 1.0))
bounds.append((-5.0, 2.0))
bounds += [(-2.0, 2.0)] * (2 * (T - 1))

res = minimize(
    neg_logpost,
    theta0,
    args=(home_idx[:cut], away_idx[:cut], Gh[:cut], Ga[:cut], w_train, L2),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 500}
)

if not res.success:
    print("WARNING: optimizer did not fully converge:", res.message)

theta_hat = res.x

alpha = theta_hat[0]
home_adv = theta_hat[1]
l3 = np.exp(theta_hat[2])

atk = np.zeros(T)
dfn = np.zeros(T)
atk[:-1] = theta_hat[3:3 + (T - 1)]
dfn[:-1] = theta_hat[3 + (T - 1): 3 + 2 * (T - 1)]
atk[:-1] -= atk[:-1].mean()
dfn[:-1] -= dfn[:-1].mean()

# -----------------------------
# 5) Fast win probability (Skellam)
# -----------------------------
def win_prob(l1, l2):
    return float(1.0 - skellam.cdf(0, mu1=l1, mu2=l2))

# -----------------------------
# 6) Predict on TEST
# -----------------------------
probs = []
pred_win = []

for i in range(cut, N):
    h = home_idx[i]
    a = away_idx[i]

    l1 = np.exp(alpha + home_adv + atk[h] - dfn[a])
    l2 = np.exp(alpha + atk[a] - dfn[h])

    p = win_prob(l1, l2)
    probs.append(p)
    pred_win.append(1 if p >= 0.5 else 0)

probs = np.array(probs)
pred_win = np.array(pred_win)
actual_win = ((Gh[cut:] - Ga[cut:]) > 0).astype(int)

acc = (pred_win == actual_win).mean()
brier = np.mean((probs - actual_win) ** 2)
ll = log_loss(actual_win, np.clip(probs, 1e-6, 1 - 1e-6))

print("TEST accuracy (%):", round(acc * 100, 2))
print("TEST Brier:", round(brier, 4))
print("TEST LogLoss:", round(ll, 4))
print("Fitted home_adv (log-scale term):", round(home_adv, 4))
print("Fitted lambda3 (shared covariance mean):", round(l3, 4))

# -----------------------------
# 7) Power rankings
# -----------------------------
base = np.exp(alpha)

eg_for = base * np.exp(atk)
eg_against = base * np.exp(-dfn)
strength = eg_for - eg_against

rank = pd.DataFrame({
    "team": teams,
    "attack": atk,
    "defense": dfn,
    "eg_for_neutral": eg_for,
    "eg_against_neutral": eg_against,
    "strength_xgd_neutral": strength
}).sort_values("strength_xgd_neutral", ascending=False).reset_index(drop=True)

rank.insert(0, "rank", rank.index + 1)

print("\nPOWER RANKINGS (bivariate Poisson attack/defense; neutral strength):")
print(rank[["rank","team","strength_xgd_neutral","eg_for_neutral","eg_against_neutral"]])