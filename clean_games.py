import pandas as pd

# load file
df = pd.read_csv("whl_2025.csv")

# columns to remove
drop_cols = [
    "record_id",
    "home_off_line",
    "home_def_pairing",
    "away_off_line",
    "away_def_pairing",
    "home_goalie",
    "away_goalie"
]

df = df.drop(columns=drop_cols)

# numeric stat columns
stat_cols = df.loc[:, "toi":"away_penalty_minutes"].columns.tolist()

# columns that should use MAX instead of SUM
max_cols = ["home_max_xg", "away_max_xg"]

# aggregation rules
agg = {}

# keep first value for these
for col in ["home_team", "away_team", "went_ot"]:
    agg[col] = "first"

# numeric columns
for col in stat_cols:
    if col in max_cols:
        agg[col] = "max"
    else:
        agg[col] = "sum"

# group by game
grouped = df.groupby("game_id", as_index=False).agg(agg)

# replace game_id with sequential numbers
grouped["game_id"] = range(1, len(grouped) + 1)

# save new file
grouped.to_csv("whl_2025_cleaned.csv", index=False)

print("Done. Cleaned file saved as whl_2025_cleaned.csv")
