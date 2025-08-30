import pandas as pd
from fantasy.scoring import add_calc_pts   # <-- import your function


# Load projections
df = pd.read_csv("ffa_season_projections_2025.csv")
df = df[df["data_src"] == "ESPN"]

# Add calc_pts column
df = add_calc_pts(df)

# Define fantasy-relevant positions
positions = ["QB", "RB", "WR", "TE", "K", "DST"]

# For each position, get the top 3 by calc_pts
for pos in positions:
    top3 = (
        df[df["pos"] == pos]
        .sort_values("calc_pts", ascending=False)
        .head(3)
    )
    print(f"\nTop 3 {pos}s:")
    print(top3[["player", "team", "data_src", "calc_pts"]].to_string(index=False))
