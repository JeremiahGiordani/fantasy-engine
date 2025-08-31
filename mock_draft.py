# mock_draft.py
"""
Simulate a multi-team snake draft with bot drafters using the DraftEngine.

What this does
--------------
- Loads a projections CSV into a PlayerPool
- Builds a league + engine config
- Runs a snake draft for NUM_ROUNDS rounds and LEAGUE_SIZE teams
- Uses the same DraftEngine for all teams (bots)
- Prints pick-by-pick at medium verbosity (one line per pick)
- Prints final roster tallies and (optionally) rosters by position

How to use
---------
1) Set CSV_PATH to your projections file (e.g., "projections_2025_wk0.csv").
2) Adjust LEAGUE_SIZE, NUM_ROUNDS, and roster rules if needed.
3) Run:  python mock_draft.py
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

from fantasy.models import (
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
    ValueModelParams
)
from src.fantasy.draft_engine import DraftEngine


# ==========================
# ===== CONFIG BLOCK =======
# ==========================

CSV_PATH = "projections_2025_wk0.csv"  # <-- set to your projections CSV

LEAGUE_SIZE = 5
NUM_ROUNDS = 16  # total rounds to simulate

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
    caps={"QB": 3, "TE": 3, "K": 2, "DST": 2},  # safety caps to avoid weird hoarding
)

ENGINE_PARAMS = EngineParams(
    start_share_priors=StartShareCurves(),
    value_model=ValueModelParams(use_vor=False)
)

ENGINE_VERBOSITY = Verbosity.PICKS  # medium verbosity (one line per pick)

PRINT_FULL_ROSTERS = True  # set False to only print tallies


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

            pos_clean = str(pos).strip().upper()
            if pos_clean not in {"QB", "RB", "WR", "TE", "K", "DST"}:
                # Skip unsupported (e.g., IDP)
                continue

            position = Position(pos_clean)

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


# ==========================
# ===== PRETTY PRINT  ======
# ==========================

def _print_roster_tallies(state: DraftState) -> None:
    print("\n=== Roster tallies by team ===")
    for t, r in enumerate(state.rosters):
        parts = []
        for p in Position:
            c = r.count(p)
            if c > 0:
                parts.append(f"{p.value}: {c}")
        print(f"Team {t}: "+"{"+", ".join(parts)+"}")


def _print_full_rosters(state: DraftState, pool: PlayerPool) -> None:
    print("\n=== Rosters ===")
    for t, r in enumerate(state.rosters):
        print(f"\nTeam {t}:")
        by_pos: Dict[Position, List[str]] = {p: [] for p in Position}
        for uid in r.players:
            pl = pool.by_uid[uid]
            by_pos[pl.position].append(f"{pl.name} ({pl.team or 'FA'})")
        for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
            if by_pos[p]:
                print(f"  {p.value}: " + ", ".join(by_pos[p]))


# ==========================
# ===== MAIN PROGRAM  ======
# ==========================

def main():
    # Load player pool
    pool = load_player_pool_from_csv(CSV_PATH)

    # League + config + state + engine
    league = LeagueConfig(league_size=LEAGUE_SIZE, rules=ROSTER_RULES)
    cfg = DraftEngineConfig(league=league, engine=ENGINE_PARAMS, verbosity=ENGINE_VERBOSITY)
    state = DraftState(league=league)
    engine = DraftEngine(config=cfg, pool=pool)

    total_picks = LEAGUE_SIZE * NUM_ROUNDS
    print(f"Starting mock draft: {LEAGUE_SIZE} teams, {NUM_ROUNDS} rounds ({total_picks} picks)\n")

    # Run the draft
    for i in range(total_picks):
        if i % 5 == 2:
            engine.config.verbosity = Verbosity.DEBUG
        engine.make_pick(state)
        engine.config.verbosity = Verbosity.PICKS

    # Summaries
    _print_roster_tallies(state)
    if PRINT_FULL_ROSTERS:
        _print_full_rosters(state, pool)

    print("\nDone.")


if __name__ == "__main__":
    main()
