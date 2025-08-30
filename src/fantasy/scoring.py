import pandas as pd

def add_calc_pts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fantasy points calculation (calc_pts) to projections DataFrame.
    Assumes ESPN standard scoring rules, handles NaNs as 0.
    Works across all positions (QB, RB, WR, TE, K, DST).
    """
    df = df.copy()
    
    # Fill NaN with 0 for all numeric fields
    stat_cols = [
        "pass_yds", "pass_tds", "pass_int",
        "rush_yds", "rush_tds",
        "rec", "rec_yds", "rec_tds",
        "fumbles_lost",
        "xp", "xp_att", "fg", "fg_att",  # kickers
        "dst_int", "dst_fum_rec", "dst_sacks", "dst_td", "dst_safety",
        "dst_pts_allowed"
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Scoring rules
    calc_pts = (
        # QB / RB / WR / TE
        df.get("pass_yds", 0) * 0.04 +
        df.get("pass_tds", 0) * 4 +
        df.get("pass_int", 0) * -2 +
        df.get("rush_yds", 0) * 0.1 +
        df.get("rush_tds", 0) * 6 +
        df.get("rec", 0) * 1 +                   # full PPR
        df.get("rec_yds", 0) * 0.1 +
        df.get("rec_tds", 0) * 6 +
        df.get("fumbles_lost", 0) * -2 +

        # Kickers (basic FG = 3, XP = 1; could be extended if split by distance is available)
        df.get("fg", 0) * 3 +
        df.get("xp", 0) * 1 +

        # DST
        df.get("dst_td", 0) * 6 +
        df.get("dst_int", 0) * 2 +
        df.get("dst_fum_rec", 0) * 2 +
        df.get("dst_sacks", 0) * 1 +
        df.get("dst_safety", 0) * 2 +
        0
    )

    # Simple defense points allowed adjustment (optional, approximate ESPN defaults)
    # if "dst_pts_allowed" in df.columns:
    #     def dst_points_allowed_score(pts):
    #         if pts == 0: return 10
    #         elif pts <= 6: return 7
    #         elif pts <= 13: return 4
    #         elif pts <= 20: return 1
    #         elif pts <= 27: return 0
    #         elif pts <= 34: return -1
    #         elif pts <= 45: return -4
    #         else: return -5
    #     calc_pts += df["dst_pts_allowed"].apply(dst_points_allowed_score)

    df["calc_pts"] = calc_pts
    return df
