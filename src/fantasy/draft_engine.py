from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import copy
import numpy as np
import pandas as pd
import copy
import math


# -----------------------------
# Config + Roster
# -----------------------------

@dataclass
class DraftConfig:
    league_size: int                 # e.g., 6, 10, 12
    starters: Dict[str, int]         # {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1,"DST":1,"K":1}
    bench: int                       # e.g., 7
    flex_positions: Tuple[str, ...] = ("RB", "WR", "TE")
    risk_lambda: float = 0.0         # risk penalty on "uncertainty" if points_vor not present
    max_per_pos_softcap: Dict[str, int] = field(
        default_factory=lambda: {"QB":2, "RB":7, "WR":8, "TE":2, "DST":1, "K":1}
    )

    def total_roster_slots(self) -> int:
        return sum(self.starters.values()) + self.bench


@dataclass
class Roster:
    starters: Dict[str, int]
    flex_positions: Tuple[str, ...]
    bench_slots: int
    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for p in ["QB", "RB", "WR", "TE", "DST", "K"]:
            self.counts.setdefault(p, 0)

    def current_starter_need(self, pos: str) -> int:
        return max(0, self.starters.get(pos, 0) - self.counts.get(pos, 0))

    def total_flex_need(self) -> int:
        return max(0, self.starters.get("FLEX", 0))

    def add(self, pos: str):
        self.counts[pos] = self.counts.get(pos, 0) + 1

    def need_factor(self, pos: str) -> float:
        w = 1.0 + 0.75 * self.current_starter_need(pos)
        if pos in self.flex_positions and self.total_flex_need() > 0:
            w += 0.5
        if pos in ("K", "DST") and sum(self.counts.values()) < 8:
            w *= 0.6
        return max(0.15, w)


# -----------------------------
# Draft State (snake-aware)
# -----------------------------

@dataclass
class DraftState:
    cfg: DraftConfig
    rosters: List[Roster] = field(default_factory=list)
    drafted_ids: Set[str] = field(default_factory=set)
    pick_number: int = 1  # global pick number (1-indexed)

    def __post_init__(self):
        if not self.rosters:
            self.rosters = [
                Roster(
                    starters=copy.deepcopy(self.cfg.starters),
                    flex_positions=self.cfg.flex_positions,
                    bench_slots=self.cfg.bench,
                )
                for _ in range(self.cfg.league_size)
            ]

    def team_on_the_clock(self) -> int:
        N = self.cfg.league_size
        r = (self.pick_number - 1) // N  # 0-based round
        i = (self.pick_number - 1) % N   # 0-based index within round
        return i if r % 2 == 0 else (N - 1 - i)

    def next_pick_distance_to_team(self, team_idx: int) -> int:
        """How many picks until 'team_idx' next selects (excluding current pick)."""
        N = self.cfg.league_size
        dist, pn = 0, self.pick_number + 1
        while True:
            r = (pn - 1) // N
            i = (pn - 1) % N
            team = i if r % 2 == 0 else (N - 1 - i)
            if team == team_idx:
                return dist
            pn += 1
            dist += 1

    def advance_one_pick(self):
        self.pick_number += 1


# -----------------------------
# Draft Assistant
# -----------------------------
class DraftAssistant:
    """
    Position-driven, snake-aware draft assistant:
    - Recomputes VOR to your league (starters + FLEX).
    - Models opponents at the POSITION level (no ADP): each upcoming team
      has a probability over {QB,RB,WR,TE,K,DST} based on roster needs and
      the top shaped value available at each position for that team.
    - Sums those probabilities across upcoming picks to get expected counts
      taken per position; uses them to compute expected replacement at your
      next pick; chooses the candidate maximizing value_now + expected_best_next.
    - Filters out IDP positions (DL/LB/DB) and enforces simple hard caps.
    """

    # Allowed fantasy positions (exclude IDP)
    ALLOWED_POS = {"QB", "RB", "WR", "TE", "K", "DST"}

    # Simple hard caps to prevent nonsense like 6 kickers / 3 DST.
    # You can tune these or make them configurable later.
    HARD_CAP = {"K": 1, "DST": 1}

    # Softmax temperature for converting position scores -> probabilities.
    # Lower = more peaky (teams behave more deterministically).
    POS_SOFTMAX_K = 0.05

    def __init__(self, cfg: DraftConfig, projections: pd.DataFrame, recompute_vor: bool = True):
        self.cfg = cfg
        self.df = self._prepare_df(projections, recompute_vor=recompute_vor)

    # ---------- Prep / VOR (league-aware) ----------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # Normalize column names
        if "pos" in d.columns and "position" not in d.columns:
            d.rename(columns={"pos": "position"}, inplace=True)

        # Filter out non-fantasy positions (IDP etc.)
        if "position" in d.columns:
            d = d[d["position"].isin(DraftAssistant.ALLOWED_POS)].copy()

        # Stable id if missing
        if "id" not in d.columns:
            d["id"] = (
                d["player"].astype(str).str.strip()
                + "|"
                + d["team"].astype(str).str.strip()
                + "|"
                + d["position"].astype(str).str.strip()
            )

        # Ensure points exist for VOR recompute
        if "points" not in d.columns:
            raise ValueError("Expected a 'points' column in projections for VOR recomputation.")
        d["points"] = d["points"].astype(float)

        # If ADP exists, we keep it only for display; logic below doesn't use it.
        if "adp" in d.columns:
            d["adp"] = d["adp"].astype(float)

        return d

    def _compute_replacement_points(self, d: pd.DataFrame) -> dict:
        """
        Compute replacement points per position for THIS league by filling
        dedicated starters and allocating FLEX greedily across RB/WR/TE.
        """
        N = self.cfg.league_size
        starters = self.cfg.starters
        flex_elig = set(self.cfg.flex_positions)

        # Build per-position sorted points (desc)
        points_by_pos = {}
        for p in d["position"].unique():
            pts = (
                d.loc[d["position"] == p, "points"]
                .astype(float)
                .fillna(0.0)
                .sort_values(ascending=False)
                .to_list()
            )
            points_by_pos[p] = pts

        # How many starters used per position (dedicated first)
        used = {p: 0 for p in points_by_pos.keys()}
        for p in points_by_pos.keys():
            need = N * starters.get(p, 0)
            used[p] = min(need, len(points_by_pos[p]))

        # Allocate FLEX across RB/WR/TE greedily
        total_flex = N * starters.get("FLEX", 0)

        def next_best():
            best_pos, best_pts = None, -1.0
            for p in flex_elig:
                lst = points_by_pos.get(p, [])
                idx = used.get(p, 0)
                if idx < len(lst):
                    pts = lst[idx]
                    if pts > best_pts:
                        best_pts, best_pos = pts, p
            return best_pos, best_pts

        for _ in range(total_flex):
            p, pts = next_best()
            if p is None:
                break
            used[p] = used.get(p, 0) + 1

        # Replacement = next available after starters+flex
        replacement_points = {}
        for p, lst in points_by_pos.items():
            idx = used.get(p, 0)
            replacement_points[p] = float(lst[idx]) if idx < len(lst) else 0.0

        return replacement_points

    def _prepare_df(self, df: pd.DataFrame, recompute_vor: bool) -> pd.DataFrame:
        d = self._normalize_columns(df)

        if recompute_vor:
            repl = self._compute_replacement_points(d)
            d["vor_league"] = d.apply(
                lambda r: float(r["points"]) - float(repl.get(r["position"], 0.0)), axis=1
            )
            d["base_value"] = d["vor_league"]
        else:
            if "points_vor" in d.columns:
                d["base_value"] = d["points_vor"].astype(float)
            else:
                lam = self.cfg.risk_lambda
                d["base_value"] = d["points"].astype(float) - lam * d.get("uncertainty", 0.0).astype(float)

        d["value"] = d["base_value"]
        return d

    # ---------- Roster shaping (your team & opponents) ----------

    def _shape_value_for_roster(self, row: pd.Series, roster: Roster) -> float:
        """
        Reuse the same shaping for you and opponents:
        - +10% per open dedicated starter at that position
        - +5% if FLEX exists and pos is FLEX-eligible
        - Early de-emphasis of K/DST
        """
        pos = row["position"]
        v = float(row["base_value"])

        need = roster.current_starter_need(pos)
        if need > 0:
            v *= (1.0 + 0.10 * need)

        if pos in self.cfg.flex_positions and self.cfg.starters.get("FLEX", 0) > 0:
            v *= 1.05

        total_taken = sum(roster.counts.values())
        starters_total = sum(self.cfg.starters.values())
        if pos in ("K", "DST") and total_taken < max(6, starters_total - 2):
            v *= 0.65

        return v

    # ---------- Position-level probabilities for an opponent team ----------

    def _pos_probs_for_team(self, pool: pd.DataFrame, roster: Roster) -> dict:
        """
        Return a dict pos->prob for a single upcoming team:
        prob(pos) ∝ exp(k * score_pos), where score_pos = top_shaped_value_at_pos * need_weight,
        with hard caps enforced and normalized over allowed positions.
        """
        k = self.POS_SOFTMAX_K
        scores = {}

        # Enforce simple hard caps (e.g., no more than 1 K or 1 DST drafted per team)
        def at_hard_cap(p: str) -> bool:
            cap = self.HARD_CAP.get(p, None)
            return (cap is not None) and (roster.counts.get(p, 0) >= cap)

        for pos in self.ALLOWED_POS:
            if at_hard_cap(pos):
                continue  # cannot pick more of this position

            # Top shaped value available at this position for THIS roster
            pool_pos = pool[pool["position"] == pos]
            if pool_pos.empty:
                continue

            shaped_vals = pool_pos.apply(lambda r: self._shape_value_for_roster(r, roster), axis=1)
            top_val = float(shaped_vals.max())

            # Need weight: 1 + 0.75 per open dedicated starter at that pos
            need_w = 1.0 + 0.75 * roster.current_starter_need(pos)

            # If FLEX remains and pos is FLEX-eligible, small bump
            if pos in self.cfg.flex_positions and self.cfg.starters.get("FLEX", 0) > 0:
                need_w *= 1.10

            score = top_val * need_w
            if score > 0:
                scores[pos] = score

        if not scores:
            # fallback: uniform over any positions not hard-capped
            elig = [p for p in self.ALLOWED_POS if not at_hard_cap(p)]
            if not elig:
                return {}
            w = 1.0 / len(elig)
            return {p: w for p in elig}

        # softmax to probabilities
        max_s = max(scores.values())
        # stabilize
        weights = {p: math.exp(k * (s - max_s)) for p, s in scores.items()}
        total_w = sum(weights.values())
        return {p: (w / total_w) for p, w in weights.items()}

    # ---------- Expected counts by position until next pick ----------

    def _expected_counts_to_next(self, state: DraftState, pool: pd.DataFrame, my_team_idx: int) -> dict:
        """
        Sum position probabilities across the specific sequence of teams who pick
        before my_team_idx picks again.
        """
        N = self.cfg.league_size
        counts = {p: 0.0 for p in self.ALLOWED_POS}

        pn = state.pick_number + 1
        while True:
            r = (pn - 1) // N
            i = (pn - 1) % N
            team = i if r % 2 == 0 else (N - 1 - i)
            pn += 1
            if team == my_team_idx:
                break  # reached our next pick

            roster = state.rosters[team]
            probs = self._pos_probs_for_team(pool, roster)
            for p, prob in probs.items():
                counts[p] += prob

        return counts  # expected (fractional) number taken per position

    # ---------- Read expected replacement value after removing expected counts ----------

    @staticmethod
    def _expected_top_after_removals(values_desc: list, expected_removed: float) -> float:
        """
        Given a descending list of values and an expected number removed `x`,
        return the expected new top value using linear interpolation between
        floor(x) and ceil(x). Indexing is 0-based (remove x from the front).
        """
        if not values_desc:
            return 0.0
        n = len(values_desc)
        x = max(0.0, min(float(expected_removed), float(n - 1)))
        k = int(math.floor(x))
        frac = x - k
        i0 = min(k, n - 1)
        i1 = min(k + 1, n - 1)
        v0, v1 = values_desc[i0], values_desc[i1]
        return (1.0 - frac) * v0 + frac * v1

    # ---------- Public API: recommendation ----------

    def recommend_for_team(self, state: DraftState, team_idx: int, top_k: int = 10) -> pd.DataFrame:
        """
        Pure position-probability lookahead (no ADP).
        Utility(candidate) = shaped_value_now + expected_best_next_value,
        where the expected next board is computed by removing the expected number
        of players by position across the opponents who pick before us.
        """
        # Filter pool to exclude anything we've already hard-capped for *our* team
        def my_hard_capped(pos: str) -> bool:
            cap = self.HARD_CAP.get(pos, None)
            return (cap is not None) and (state.rosters[team_idx].counts.get(pos, 0) >= cap)

        pool = self.df[
            (~self.df["id"].isin(state.drafted_ids))
            & (self.df["position"].apply(lambda p: not my_hard_capped(p)))
        ].copy()

        my_roster = state.rosters[team_idx]

        # Current shaped value for you (now)
        pool["value_now"] = pool.apply(lambda r: self._shape_value_for_roster(r, my_roster), axis=1)

        # Precompute per-position sorted value lists (desc)
        values_by_pos = {}
        for p in self.ALLOWED_POS:
            vals = pool.loc[pool["position"] == p, "value_now"].astype(float).sort_values(ascending=False).tolist()
            if vals:
                values_by_pos[p] = vals

        # Expected counts taken by others (position-level)
        exp_counts = self._expected_counts_to_next(state, pool, my_team_idx=team_idx)

        rows = []
        # Consider a modest candidate frontier: top N overall by now-value
        candidates = pool.sort_values("value_now", ascending=False).head(40)

        for _, A in candidates.iterrows():
            posA = A["position"]

            # Build a temporary copy of per-position boards if we take A now:
            # - Remove A from its position (approx: remove the top item)
            # - Then remove expected counts exp_counts[p] for opponents (fractional)
            vals_after = {}
            for p, vals in values_by_pos.items():
                if not vals:
                    continue
                list_copy = vals.copy()
                if p == posA:
                    # Remove one item from the top as a proxy for taking A
                    if list_copy:
                        list_copy.pop(0)
                vals_after[p] = list_copy

            # Compute expected top value at your next turn per position
            expected_tops = {}
            for p, lst in vals_after.items():
                if not lst:
                    expected_tops[p] = 0.0
                    continue
                removed = exp_counts.get(p, 0.0)
                expected_tops[p] = self._expected_top_after_removals(lst, removed)

            # Next-pick best we expect among all positions
            exp_best_next = max(expected_tops.values()) if expected_tops else 0.0

            utility = float(A["value_now"]) + float(exp_best_next)

            rows.append({
                "player": A["player"],
                "team": A["team"],
                "position": posA,
                "value": round(float(A["value_now"]), 2),
                "exp_next": round(float(exp_best_next), 2),
                "utility": round(float(utility), 2),
                "adp": A.get("adp", None),
                "points": A.get("points", None),
                "points_vor": A.get("points_vor", None),
                "uncertainty": A.get("uncertainty", None),
                "id": A["id"],
            })

        out = pd.DataFrame(rows).sort_values("utility", ascending=False).head(top_k)
        return out[["player","team","position","value","exp_next","utility","adp","points","points_vor","uncertainty","id"]]

    # ---------- Apply pick ----------

    def apply_pick(self, state: DraftState, player_id: str):
        """
        Apply the current on-the-clock pick (whoever it is) to the shared state.
        Enforces HARD_CAP for that team's roster simply by never recommending
        over-cap positions; if a human forces a pick elsewhere, we still record it.
        """
        state.drafted_ids.add(player_id)
        team = state.team_on_the_clock()
        pos = self.df.loc[self.df["id"] == player_id, "position"].values[0]
        state.rosters[team].add(pos)
        state.advance_one_pick()
