import pandas as pd
import numpy as np

from fantasy.scoring import add_calc_pts

# --- CONFIG: set your scoring here ---
SCORING = {
    "ppr": 1.0,                  # set 0.5 for half, 0 for standard
    "pass_td": 4.0,
    "pass_yd": 1.0/25.0,         # 0.04
    "pass_int": -2.0,
    "rush_td": 6.0,
    "rush_yd": 1.0/10.0,         # 0.1
    "rec_td": 6.0,
    "rec_yd": 1.0/10.0,          # 0.1
    "fumbles_lost": -2.0,
    "two_pt": 2.0,               # pass/rush/rec 2-pt all worth 2
    # Optional bonuses (toggle on/off)
    "bonus_pass_300": 0.0,       # e.g., 3.0 if your league gives +3 for 300+ pass yds
    "bonus_rush_100": 0.0,       # e.g., 3.0 for 100+ rush yds
    "bonus_rec_100": 0.0,        # e.g., 3.0 for 100+ rec yds
}

# Load
df = pd.read_csv("ffa_season_projections_2025.csv")

# ESPN rows only (others already have site_pts)
espn = df[df["data_src"] == "ESPN"].copy()

espn = add_calc_pts(espn)


# # Fill NaNs with 0 so QBs don't produce NaN from missing receiving fields
# for col in ["pass_yds","pass_tds","pass_int",
#             "rush_yds","rush_tds",
#             "rec","rec_yds","rec_tds",
#             "fumbles_lost",
#             # common 2-pt fields (names vary across sources; these are defensive)
#             "pass_2pt","rush_2pt","rec_2pt","two_pt_conv"]:
#     if col in espn.columns:
#         espn[col] = espn[col].fillna(0)
#     else:
#         espn[col] = 0.0

# # Compute fantasy points (season)
# espn["calc_pts"] = (
#     espn["pass_yds"] * SCORING["pass_yd"] +
#     espn["pass_tds"] * SCORING["pass_td"] +
#     espn["pass_int"] * SCORING["pass_int"] +
#     espn["rush_yds"] * SCORING["rush_yd"] +
#     espn["rush_tds"] * SCORING["rush_td"] +
#     espn["rec"] * SCORING["ppr"] +
#     espn["rec_yds"] * SCORING["rec_yd"] +
#     espn["rec_tds"] * SCORING["rec_td"] +
#     espn["fumbles_lost"] * SCORING["fumbles_lost"] +
#     # 2-pt conversions (sum any available fields)
#     (espn["pass_2pt"] + espn["rush_2pt"] + espn["rec_2pt"] + espn["two_pt_conv"]) * SCORING["two_pt"]
# )

# # Optional one-time bonuses (applied to the season totals ESPN lists)
# # NOTE: These are ON/OFF thresholds; if your league uses per-game bonuses, compute weekly instead.
# if SCORING["bonus_pass_300"]:
#     espn["calc_pts"] += np.where(espn["pass_yds"] >= 300, SCORING["bonus_pass_300"], 0)
# if SCORING["bonus_rush_100"]:
#     espn["calc_pts"] += np.where(espn["rush_yds"] >= 100, SCORING["bonus_rush_100"], 0)
# if SCORING["bonus_rec_100"]:
#     espn["calc_pts"] += np.where(espn["rec_yds"] >= 100, SCORING["bonus_rec_100"], 0)

# Show top-10 ESPN by our computed points
cols_to_show = ["player","team","pos","pass_yds","pass_tds","pass_int","rush_yds","rush_tds","rec","rec_yds","rec_tds","fumbles_lost","calc_pts"]
print(espn.sort_values("calc_pts", ascending=False).head(10)[cols_to_show])

# If you want to directly compare with ESPN FPTS you read off the site,
# add a column with your manual entry to sanity-check rounding/bonuses.
