# live_draft.py
"""
Interactive live draft assistant using DraftEngine.

What this does
--------------
- Loads a projections CSV into a PlayerPool
- Builds a league + engine config (same knobs you used for mock)
- Runs an interactive snake draft where you:
    * Type the real-life player picked for each team
    * When it's YOUR pick, it shows full recommendations (TRACE)
      and lets you choose your player (or auto-pick best)
- Prints a running "board" of picks and supports simple commands

How to use
---------
1) Set CSV_PATH or pass --csv on the CLI (defaults provided).
2) Run:  python live_draft.py --my-team 0  (or whichever team index is yours)
3) Follow the prompts. Type a player's name to register the pick.
   Commands: help, board, roster [team], mine, auto, undo, skip, quit

Notes
-----
- Player lookup is by name (case-insensitive); if ambiguous, you’ll be asked to disambiguate.
- For your pick, we show full TRACE recommendations *before* you draft.
- For others' picks, we simply record who they took. You can also type 'auto' to let the bot pick.
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, List, Optional, Tuple

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
    ValueModelParams,
)
from src.fantasy.draft_engine import DraftEngine


# ==========================
# ===== CONFIG DEFAULTS ====
# ==========================

DEFAULT_CSV = "data/projections_2025_wk0.csv"
DEFAULT_LEAGUE_SIZE = 5
DEFAULT_NUM_ROUNDS = 16

DEFAULT_ROSTER_RULES = RosterRules(
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
    bench=7,
    caps={"QB": 3, "TE": 3, "K": 2, "DST": 2},
)

DEFAULT_ENGINE_PARAMS = EngineParams(
    start_share_priors=StartShareCurves(),
    value_model=ValueModelParams(use_vor=False),
)

# Verbosity: we’ll always print at TRACE when showing recommendations
RUN_VERBOSITY = Verbosity.PICKS  # general chatter


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
# ===== HELPER UTILS  ======
# ==========================

def _team_on_clock_str(state: DraftState, my_team: int) -> str:
    t = state.team_on_the_clock()
    me = " (YOU)" if t == my_team else ""
    return f"Pick {state.pick_number:3d} — Team {t}{me}"


def _find_matches(pool: PlayerPool, query: str) -> List[Player]:
    q = query.strip().lower()
    # First try exact name match
    exact = [p for p in pool.by_uid.values() if p.name.lower() == q]
    if exact:
        return exact
    # Else substring
    return [p for p in pool.by_uid.values() if q in p.name.lower()]


def _disambiguate(matches: List[Player]) -> Optional[Player]:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    print("Ambiguous — choose one:")
    for i, p in enumerate(matches, 1):
        tag = f"{p.name} ({p.position.value}, {p.team or 'FA'})"
        print(f"  {i:2d}) {tag}")
    while True:
        s = input("Enter number (or blank to cancel): ").strip()
        if s == "":
            return None
        if s.isdigit():
            k = int(s)
            if 1 <= k <= len(matches):
                return matches[k - 1]
        print("Invalid selection.")


def _apply_pick(state: DraftState, pool: PlayerPool, team_idx: int, player: Player) -> None:
    state.drafted_uids.add(player.uid)
    state.rosters[team_idx].add(player.position, player.uid)
    state.advance_one_pick()


def _undo_last_pick(state: DraftState) -> bool:
    # Assumes DraftState remembers pick order? If not, we keep our own stack in main.
    return False  # we’ll manage undo using our own stack in the script


def _print_recommendations(engine: DraftEngine, state: DraftState) -> None:
    """
    Print full TRACE-style recommendations for the team currently on the clock,
    without mutating state.
    """
    team_idx = state.team_on_the_clock()
    # Temporarily raise verbosity to TRACE for printing
    prev_v = engine.config.verbosity
    engine.config.verbosity = getattr(Verbosity, "TRACE", 3)

    best_row, breakdown, top5, rows, pos_cond_debug = engine._recommend_for_team(state, team_idx)

    # Mirror the printing logic from DraftEngine.make_pick (but do NOT pick)
    print(f"\n{_team_on_clock_str(state, team_idx)} — RECOMMENDATIONS (TRACE):")
    if not rows:
        print("  (no feasible candidates)")
        engine.config.verbosity = prev_v
        return

    # Top candidates table
    print("    Top candidates:")
    for r in rows[:5]:
        print(
            f"      - {r.name:20s} {r.position.value:>3s}  "
            f"Δ_now={r.value_now:7.2f}  Δ_next={r.exp_next:7.2f}  "
            f"U={r.utility:7.2f}  ADP={r.adp if r.adp is not None else 'NA'}"
        )
    shown_positions = {r.position for r in rows[:5]}
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        if pos not in shown_positions:
            best_pos = max((r for r in rows if r.position == pos),
                           key=lambda r: r.utility, default=None)
            if best_pos:
                print(
                    f"      - {best_pos.name:20s} {best_pos.position.value:>3s}  "
                    f"Δ_now={best_pos.value_now:7.2f}  Δ_next={best_pos.exp_next:7.2f}  "
                    f"U={best_pos.utility:7.2f}  ADP={best_pos.adp if best_pos.adp is not None else 'NA'}"
                )

    if breakdown is not None:
        print("    Next-turn breakdown by position:")
        for p in sorted(breakdown.exp_by_pos.keys(), key=lambda x: x.value):
            val = breakdown.exp_by_pos[p]
            rnk = breakdown.rank_by_pos[p]
            print(f"      * {p.value:>3s}: exp={val:7.2f}, r_p(a)={rnk:.2f}")

    if pos_cond_debug:
        print("    Next-turn breakdowns conditioned on picking the top player at each position:")
        for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
            entry = pos_cond_debug.get(pos)
            if not entry:
                continue
            nm = entry['name']
            print(f"    If pick {nm:20s} {pos.value:>3s}:")
            exp_map = entry.get('exp_by_pos', {}) or {}
            rnk_map = entry.get('rank_by_pos', {}) or {}
            for p in (Position.DST, Position.K, Position.QB, Position.RB, Position.TE, Position.WR):
                val = float(exp_map.get(p, 0.0))
                rnk = rnk_map.get(p, None)
                if rnk is None:
                    print(f"      * {p.value:>3s}: exp={val:7.2f}, r_p(a)=   -  ")
                else:
                    print(f"      * {p.value:>3s}: exp={val:7.2f}, r_p(a)={float(rnk):.2f}")

    # Restore verbosity
    engine.config.verbosity = prev_v


def _print_roster(state: DraftState, pool: PlayerPool, team_idx: int) -> None:
    r = state.rosters[team_idx]
    print(f"\nTeam {team_idx} roster:")
    by_pos: Dict[Position, List[str]] = {p: [] for p in Position}
    for uid in r.players:
        pl = pool.by_uid[uid]
        by_pos[pl.position].append(f"{pl.name} ({pl.team or 'FA'})")
    for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
        if by_pos[p]:
            print(f"  {p.value}: " + ", ".join(by_pos[p]))


# ==========================
# ======= MAIN LOOP  =======
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Interactive live draft assistant.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to projections CSV")
    parser.add_argument("--league-size", type=int, default=DEFAULT_LEAGUE_SIZE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--my-team", type=int, required=True, help="Your team index (0-based)")
    parser.add_argument("--trace", action="store_true", help="Force TRACE prints even when not your pick")
    args = parser.parse_args()

    # Load pool
    pool = load_player_pool_from_csv(args.csv)

    # League + config + state + engine
    league = LeagueConfig(league_size=args.league_size, rules=DEFAULT_ROSTER_RULES)
    cfg = DraftEngineConfig(league=league, engine=DEFAULT_ENGINE_PARAMS, verbosity=RUN_VERBOSITY)
    state = DraftState(league=league)
    engine = DraftEngine(config=cfg, pool=pool)

    total_picks = args.league_size * args.rounds
    print(f"Live draft ready: {args.league_size} teams, {args.rounds} rounds ({total_picks} picks)")
    print(f"Your team index: {args.my_team}\nType 'help' for commands.\n")

    # Track a simple pick history stack for 'undo'
    history: List[Tuple[int, str]] = []  # (team_idx, uid)

    while state.pick_number <= total_picks:
        t_on_clock = state.team_on_the_clock()
        pick_hdr = _team_on_clock_str(state, args.my_team)
        print(f"\n=== {pick_hdr} ===")

        # Optional: always show recommendations (TRACE) even when not your pick
        if args.trace:
            _print_recommendations(engine, state)

        # If it's your pick, auto-show recommendations
        if t_on_clock == args.my_team and not args.trace:
            _print_recommendations(engine, state)

        # Prompt
        who = "your" if t_on_clock == args.my_team else f"Team {t_on_clock}'s"
        s = input(f"Enter {who} pick (name), or command: ").strip()

        # Commands
        if s.lower() in {"q", "quit", "exit"}:
            print("Exiting without completing the draft.")
            break

        if s.lower() in {"h", "help"}:
            print(
                "\nCommands:\n"
                "  help                Show this help\n"
                "  board               Show last 12 picks\n"
                "  roster [team]       Show roster for team (default: yours)\n"
                "  mine                Show TRACE recommendations for your next pick (no state change)\n"
                "  auto                Auto-pick (engine) for the team on the clock\n"
                "  undo                Undo last pick you entered (one step)\n"
                "  skip                Advance pick with no selection (debug only)\n"
                "  quit                Exit immediately\n"
                "Or type a player name to register the pick for the team on the clock.\n"
            )
            continue

        if s.lower() == "board":
            if not history:
                print("(no picks yet)")
            else:
                print("\nLast picks:")
                for team_idx, uid in history[-12:]:
                    p = pool.by_uid[uid]
                    print(f"  Team {team_idx}: {p.name} ({p.position.value}, {p.team or 'FA'})")
            continue

        if s.lower().startswith("roster"):
            parts = s.split()
            team = args.my_team
            if len(parts) > 1 and parts[1].isdigit():
                team = int(parts[1])
            _print_roster(state, pool, team)
            continue

        if s.lower() == "mine":
            _print_recommendations(engine, state)
            continue

        if s.lower() == "auto":
            # Let engine pick for the team on the clock
            prev = engine.config.verbosity
            engine.config.verbosity = Verbosity.PICKS
            row = engine.make_pick(state)
            engine.config.verbosity = prev
            if row is not None:
                history.append((t_on_clock, row.uid))
            continue

        if s.lower() == "undo":
            if not history:
                print("Nothing to undo.")
                continue
            # Pop last pick
            last_team, last_uid = history.pop()
            print("Undoing last pick…")

            # Rebuild state from scratch using the remaining history
            state = DraftState(league=league)  # fresh rosters with proper ctor args
            for team_idx, uid in history:
                pl = pool.by_uid[uid]
                state.drafted_uids.add(uid)
                state.rosters[team_idx].add(pl.position, uid)
                state.advance_one_pick()
            continue

        if s.lower() == "skip":
            print("Skipping pick (no player) — debug only.")
            state.advance_one_pick()
            continue

        # Otherwise: treat input as a player name
        matches = _find_matches(pool, s)
        # Filter out already drafted
        matches = [m for m in matches if m.uid not in state.drafted_uids]

        if not matches:
            print("No available player matched that input.")
            continue

        player = _disambiguate(matches)
        if player is None:
            print("Cancelled.")
            continue

        # Apply pick to the team on the clock
        _apply_pick(state, pool, t_on_clock, player)
        history.append((t_on_clock, player.uid))

    if state.pick_number > total_picks:
        print("\nDraft complete. Good luck!")
    else:
        print("\nDraft ended early.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
        sys.exit(0)
