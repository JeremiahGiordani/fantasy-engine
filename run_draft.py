# run_draft.py
"""
Run-time helper for a LIVE or PRACTICE draft.
- Loads a projections CSV into a PlayerPool
- Builds a league + engine config
- Applies any manual picks (what other teams already drafted)
- Prints a recommendation for YOUR next pick at high verbosity
- (Optionally) lets you execute the pick to mutate state

HOW TO USE:
1) Edit the CONFIG BLOCK below:
   - CSV_PATH: path to your projections CSV (e.g., "projections_2025_wk0.csv")
   - LEAGUE_SIZE: number of teams
   - MY_TEAM_INDEX: your team index (0-based)
   - ROSTER_RULES: starting lineup + bench + flex config for your league
   - MANUAL_PICKS: list of what has ALREADY happened in the real draft, in order.
        Each item is dict(name=..., team=..., pos=...) OR dict(uid=...)
        Only one of (uid) or (name[, team, pos]) is needed.
        Examples:
            {"name": "Bijan Robinson", "team": "ATL", "pos": "RB"}
            {"uid": "Bijan Robinson|ATL|RB"}  # if you know the uid format

2) Run:
   python run_draft.py

3) It will print:
   - current state (pick number, who's on the clock)
   - a recommendation for your next pick (top-5, plus next-turn per-position breakdown)
   - You can set VERBOSITY to 0/1/2; this script forces 2 for clarity.

CSV REQUIREMENTS:
- Flexible; we try to map common column names:
    player/name, position/pos, team, points/mu, vor, adp, floor, ceiling, sigma/sd
- Unknown columns are ignored. Missing values default to None.

TIP:
If you’re mid-draft, just append picks in MANUAL_PICKS to reflect what already happened,
then re-run this script. It will recompute from the CSV and print a fresh recommendation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.fantasy.models import (
    DraftEngineConfig,
    DraftState,
    EngineParams,
    LeagueConfig,
    Player,
    PlayerPool,
    Position,
    Projection,
    RosterRules,
    StartShareCurves,
    Verbosity,
)
from src.fantasy.draft_engine import DraftEngine
from src.fantasy.marginal_value import value_now_for_candidates  # for optional extra checks


# ==========================
# ===== CONFIG BLOCK =======
# ==========================

CSV_PATH = "data/projections_2025_wk0.csv"  # <-- set this to your CSV file

LEAGUE_SIZE = 10
MY_TEAM_INDEX = 0

# Starters/bench — adjust to your league
ROSTER_RULES = RosterRules(
    starters={
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "DST": 1,
        "K": 1,
    },
    flex_positions={Position.RB, Position.WR, Position.TE},
    bench=6,
    caps={"QB": 3, "TE": 3, "K": 2, "DST": 2},  # soft safety caps
)

# Engine params — adjust if you’d like
ENGINE_PARAMS = EngineParams(
    start_share_priors=StartShareCurves(),
    # use_vor=True # uses VOR if present; otherwise falls back to μ
    # risk_lambda shrinks value for larger sigma (risk aversion)
    # opponent model controls how we expect others to draft
)

# Script verbosity — we’ll force engine verbosity=DEBUG so you get rich prints
VERBOSITY = Verbosity.DEBUG

# MANUAL PICKS: what has ALREADY been drafted (absolute order from pick #1).
# You only need to fill the fields that identify the player.
# Examples:
#   {"name": "Bijan Robinson", "team": "ATL", "pos": "RB"}
#   {"uid": "Josh Allen|BUF|QB"}
MANUAL_PICKS: List[Dict[str, str]] = [
    # {"name": "Bijan Robinson", "team": "ATL", "pos": "RB"},
    # {"name": "Saquon Barkley", "team": "PHI", "pos": "RB"},
    # {"name": "Ja'Marr Chase", "team": "CIN", "pos": "WR"},
]


# ==========================
# ===== LOADER UTILS =======
# ==========================

def _coalesce(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.upper() == "NA":
            return None
        return float(s)
    except Exception:
        return None


def load_player_pool_from_csv(path: str) -> PlayerPool:
    by_uid: Dict[str, Player] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _coalesce(row, ["player", "name", "Player", "Name"])
            pos = _coalesce(row, ["position", "pos", "Position", "Pos"])
            team = _coalesce(row, ["team", "Team"])
            if not name or not pos:
                continue

            # Map/clean position
            pos_clean = str(pos).strip().upper()
            if pos_clean not in {"QB", "RB", "WR", "TE", "K", "DST"}:
                # Skip IDP or unsupported positions silently
                continue

            try:
                position = Position(pos_clean)
            except Exception:
                continue

            mu = _to_float(_coalesce(row, ["points", "mu", "proj", "fpts", "FPTS"]))
            floor = _to_float(_coalesce(row, ["floor", "Floor"]))
            ceiling = _to_float(_coalesce(row, ["ceiling", "Ceiling"]))
            sigma = _to_float(_coalesce(row, ["sigma", "sd", "sd_pts"]))
            vor = _to_float(_coalesce(row, ["points_vor", "VOR", "vor"]))
            adp = _to_float(_coalesce(row, ["adp", "ADP"]))

            proj = Projection(
                mu=mu or 0.0,
                floor=floor,
                ceiling=ceiling,
                sigma=sigma,
                vor=vor,
                adp=adp,
                source=_coalesce(row, ["source", "Source"]),
            )
            uid = f"{name}|{team or 'FA'}|{position.value}"
            by_uid[uid] = Player(
                uid=uid,
                name=name,
                position=position,
                team=team,
                proj=proj,
            )
    return PlayerPool(by_uid=by_uid)


def find_uid(
    pool: PlayerPool,
    name: Optional[str] = None,
    team: Optional[str] = None,
    pos: Optional[str] = None,
    uid: Optional[str] = None,
) -> Optional[str]:
    """Resolve a player selection into a uid."""
    if uid:
        return uid if uid in pool.by_uid else None
    if not name:
        return None
    name_norm = name.strip().lower()
    team_norm = team.strip().upper() if team else None
    pos_norm = pos.strip().upper() if pos else None
    candidates = []
    for u, p in pool.by_uid.items():
        if p.name.strip().lower() != name_norm:
            continue
        if team_norm and (p.team or "").upper() != team_norm:
            continue
        if pos_norm and p.position.value != pos_norm:
            continue
        candidates.append(u)
    if len(candidates) == 1:
        return candidates[0]
    # If ambiguous, try best guess by ADP (earlier is “more prominent”)
    if candidates:
        candidates.sort(key=lambda u: (pool.by_uid[u].proj.adp or 9999.0))
        return candidates[0]
    return None


# ==========================
# ===== MAIN PROGRAM  ======
# ==========================

def main():
    # 1) Load pool
    pool = load_player_pool_from_csv(CSV_PATH)

    # 2) Build config + state + engine
    league = LeagueConfig(league_size=LEAGUE_SIZE, rules=ROSTER_RULES)
    cfg = DraftEngineConfig(league=league, engine=ENGINE_PARAMS, verbosity=VERBOSITY)
    state = DraftState(league=league)
    engine = DraftEngine(config=cfg, pool=pool)

    # 3) Apply manual picks (what already happened)
    for pick_idx, sel in enumerate(MANUAL_PICKS, start=1):
        # Ensure state.pick_number == pick_idx
        if state.pick_number != pick_idx:
            # Fast-forward if needed (shouldn't happen if you keep list contiguous)
            state.pick_number = pick_idx
        on_clock = state.team_on_the_clock()
        uid = find_uid(pool,
                       name=sel.get("name"),
                       team=sel.get("team"),
                       pos=sel.get("pos"),
                       uid=sel.get("uid"))
        if uid is None:
            raise ValueError(f"Could not resolve player in MANUAL_PICKS at pick {pick_idx}: {sel}")
        pl = pool.by_uid[uid]
        # Apply
        state.drafted_uids.add(uid)
        state.rosters[on_clock].add(pl.position, uid)
        if VERBOSITY >= Verbosity.PICKS:
            print(f"{pick_idx:3d}. Team {on_clock} (manual) -> {pl.name} ({pl.position.value}, {pl.team or 'FA'})")
        state.advance_one_pick()

    # 4) Show current draft pointer
    on_clock = state.team_on_the_clock()
    rnd = state.round_index() + 1
    pos_in_round = state.index_in_round() + 1
    print("\n=== Draft Status ===")
    print(f"Pick #{state.pick_number}  (Round {rnd}, Pick {pos_in_round} in round)")
    print(f"Team on the clock: {on_clock}")

    # 5) If it's our turn, print RECOMMENDATION (without auto-picking)
    if on_clock == MY_TEAM_INDEX:
        # Use engine's internal evaluator to get best + top5 + breakdown
        best, breakdown, top5 = engine._recommend_for_team(state, MY_TEAM_INDEX)  # type: ignore

        if best is None:
            print("No feasible candidates. (Check caps/gates and CSV columns.)")
            return

        pl = pool.by_uid[best.uid]
        print("\n=== Recommendation (do NOT auto-pick) ===")
        print(
            f"-> {pl.name} ({pl.position.value}, {pl.team or 'FA'})  "
            f"[Δ_now={best.value_now:.2f}, Δ_next={best.exp_next:.2f}, "
            f"U={best.utility:.2f}, ADP={best.adp if best.adp is not None else 'NA'}]"
        )

        print("\nTop 5 candidates:")
        for r in top5:
            print(
                f"   - {r.name:22s} {r.position.value:>3s}  "
                f"Δ_now={r.value_now:7.2f}  Δ_next={r.exp_next:7.2f}  "
                f"U={r.utility:7.2f}  ADP={r.adp if r.adp is not None else 'NA'}"
            )

        if breakdown:
            print("\nNext-turn breakdown by position (conditioned on picking RECOMMENDED):")
            for p in sorted(breakdown.exp_by_pos.keys(), key=lambda x: x.value):
                val = breakdown.exp_by_pos[p]
                rnk = breakdown.rank_by_pos[p]
                print(f"   * {p.value:>3s}: exp={val:7.2f}, r_p(a)={rnk:.2f}")

        # If you want to actually lock in the pick, uncomment:
        # print("\nLocking in recommendation...")
        # engine.make_pick(state)

    else:
        # Not our turn; optionally print who is on the clock and suggest what THEY might do
        print(f"\nIt is NOT your turn (your team index = {MY_TEAM_INDEX}).")
        print("Add more MANUAL_PICKS to catch up to your turn, then re-run this script.")

    # 6) Optional: print roster tallies for sanity
    print("\n=== Roster tallies by team ===")
    for t, r in enumerate(state.rosters):
        print(f"Team {t}: {{"
              + ", ".join([f"{p.value}: {r.count(p)}" for p in Position if r.count(p) > 0])
              + "}}")


if __name__ == "__main__":
    main()
