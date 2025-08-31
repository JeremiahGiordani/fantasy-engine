# src/fantasy/draft_engine.py
from __future__ import annotations

"""
DraftEngine: orchestrates candidate evaluation, one-step lookahead, and pick execution.

Responsibilities
---------------
- Build the feasible candidate set for the team on the clock (respecting caps/gates).
- For each candidate a:
    Δ_now(a) = J(R ∪ {a}) - J(R)
    Δ_next^(k)(a) ≈ max_p  m̃_p^(a)(r_p^(a)),  with  r_p^(a) = 1 + 1{p=π(a)} + λ_p
  where λ_p are expected removals by position between now and the team's next pick.
- Choose the candidate with maximal Utility = Δ_now + Δ_next.
- Apply the pick to the DraftState and print diagnostics per verbosity level.

This engine assumes:
- PlayerPool contains all players (by_uid) and can filter availability via state.drafted_uids.
- models.py, marginal_value.py, opponent_behavior.py are present as designed.

Verbosity
---------
- 0 (QUIET): no printing
- 1 (PICKS): one line per pick with key numbers
- 2 (DEBUG): plus top-5 candidate table and next-turn per-position breakdown (values and ranks)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import (
    DraftEngineConfig,
    DraftState,
    PlayerPool,
    Position,
    Verbosity,
    CandidateRow,
    NextTurnBreakdown,
)
from .marginal_value import (
    delta_now_for_candidate,
    top_marginals_by_position_after_pick,
    interpolate_at_rank,
    value_now_for_candidates,
)
from .opponent_behavior import position_probabilities_for_team


# -----------------------------
# Helpers: snake math & gates
# -----------------------------

def team_on_the_clock_at(pick_number: int, league_size: int) -> int:
    """Return team index on the clock at absolute pick_number (1-based)."""
    n = league_size
    r = (pick_number - 1) // n  # 0-based round
    i = (pick_number - 1) % n   # 0-based within round, left-to-right
    return i if (r % 2 == 0) else (n - 1 - i)


def next_pick_index_for_team(current_pick: int, league_size: int, team_idx: int) -> int:
    """
    Return the next absolute pick index (> current_pick) where team_idx will draft.
    Simple iteration is fine (n is small).
    """
    p = current_pick + 1
    while True:
        if team_on_the_clock_at(p, league_size) == team_idx:
            return p
        p += 1


def _core_starters_filled(state: DraftState, team_idx: int) -> bool:
    """Mirror of opponent_behavior._core_starters_filled for our own gating."""
    rules = state.league.rules
    roster = state.rosters[team_idx]
    needed = rules.starters_required_non_kdst()

    have = 0
    have += min(roster.count(Position.QB), rules.starter_slots_for(Position.QB))
    have += min(roster.count(Position.RB), rules.starter_slots_for(Position.RB))
    have += min(roster.count(Position.WR), rules.starter_slots_for(Position.WR))
    have += min(roster.count(Position.TE), rules.starter_slots_for(Position.TE))

    flex_need = int(rules.starters.get("FLEX", 0))
    if flex_need > 0:
        rb_excess = max(0, roster.count(Position.RB) - rules.starter_slots_for(Position.RB))
        wr_excess = max(0, roster.count(Position.WR) - rules.starter_slots_for(Position.WR))
        te_excess = max(0, roster.count(Position.TE) - rules.starter_slots_for(Position.TE))
        have += min(flex_need, rb_excess + wr_excess + te_excess)

    return have >= needed


def _picks_remaining_for_team(state: DraftState, team_idx: int) -> int:
    """
    Remaining total picks this team will have (including the one on the clock)
    based on roster rules: total slots = sum(starters) + bench.
    """
    rules = state.league.rules
    total_slots = int(sum(rules.starters.values()) + rules.bench)
    current = len(state.rosters[team_idx].players)
    return max(0, total_slots - current)


def _required_starter_needs(state: DraftState, team_idx: int) -> tuple[dict[Position, int], int]:
    """
    Return (dedicated_needs, flex_need_remaining) as INTEGERS.
    - dedicated_needs[p]: number of starting slots still to fill at position p (QB,RB,WR,TE,DST,K).
    - flex_need_remaining: number of FLEX starters still to fill after counting current excess
      of FLEX-eligible players above their dedicated starters.
    """
    rules = state.league.rules
    roster = state.rosters[team_idx]

    # Dedicated needs
    need: dict[Position, int] = {p: 0 for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K)}
    for p in need.keys():
        need[p] = max(0, int(rules.starter_slots_for(p)) - roster.count(p))

    # FLEX remaining
    flex_total = int(rules.starters.get("FLEX", 0))
    if flex_total <= 0:
        flex_left = 0
    else:
        rb_excess = max(0, roster.count(Position.RB) - int(rules.starter_slots_for(Position.RB)))
        wr_excess = max(0, roster.count(Position.WR) - int(rules.starter_slots_for(Position.WR)))
        te_excess = max(0, roster.count(Position.TE) - int(rules.starter_slots_for(Position.TE)))
        flex_left = max(0, flex_total - (rb_excess + wr_excess + te_excess))

    return need, flex_left


def _would_pick_be_feasible(
    state: DraftState,
    team_idx: int,
    pick_pos: Position,
) -> bool:
    """
    Feasibility check: if team_idx selects a player at position `pick_pos` now,
    will they still be able to fill all REQUIRED starters with their remaining picks?
    """
    L = _picks_remaining_for_team(state, team_idx)      # includes this pick
    need, flex_left = _required_starter_needs(state, team_idx)

    # Apply the hypothetical pick to needs (greedy: reduce dedicated; else reduce FLEX if eligible)
    if pick_pos in need and need[pick_pos] > 0:
        need[pick_pos] -= 1
    elif pick_pos in state.league.rules.flex_positions and flex_left > 0:
        flex_left -= 1
    # else: purely a bench/duplicate pick, no reduction to required needs

    # Remaining required after taking this pick must fit into remaining picks (L-1)
    req_after = int(sum(need.values()) + flex_left)
    return req_after <= max(0, L - 1)


# -----------------------------
# DraftEngine
# -----------------------------

@dataclass
class DraftEngine:
    config: DraftEngineConfig
    pool: PlayerPool

    # ---------- Public API ----------

    def make_pick(self, state: DraftState) -> Optional[CandidateRow]:
        """
        Execute a pick for the team currently on the clock, mutating `state`.
        Returns the chosen CandidateRow (or None if no feasible candidates).
        """
        team_idx = state.team_on_the_clock()

        best_row, breakdown, top5 = self._recommend_for_team(state, team_idx)
        if best_row is None:
            if self.config.verbosity >= Verbosity.PICKS:
                print(f"{state.pick_number:3d}. Team {team_idx} -> (no feasible candidates)")
            state.advance_one_pick()
            return None

        # Apply pick
        chosen_uid = best_row.uid
        chosen_player = self.pool.by_uid[chosen_uid]
        state.drafted_uids.add(chosen_uid)
        state.rosters[team_idx].add(chosen_player.position, chosen_uid)

        # Print
        if self.config.verbosity >= Verbosity.PICKS:
            print(
                f"{state.pick_number:3d}. Team {team_idx} -> "
                f"{chosen_player.name} ({chosen_player.position.value}, {chosen_player.team or 'FA'})  "
                f"[value={best_row.value_now:.2f}, expNext={best_row.exp_next:.2f}, "
                f"util={best_row.utility:.2f}, ADP={best_row.adp if best_row.adp is not None else 'NA'}]"
            )

        if self.config.verbosity >= Verbosity.DEBUG and top5:
            print("    Top candidates:")
            for r in top5:
                print(
                    f"      - {r.name:20s} {r.position.value:>3s}  "
                    f"Δ_now={r.value_now:7.2f}  Δ_next={r.exp_next:7.2f}  "
                    f"U={r.utility:7.2f}  ADP={r.adp if r.adp is not None else 'NA'}"
                )
            if breakdown is not None:
                print("    Next-turn breakdown by position:")
                for p in sorted(breakdown.exp_by_pos.keys(), key=lambda x: x.value):
                    val = breakdown.exp_by_pos[p]
                    rnk = breakdown.rank_by_pos[p]
                    print(f"      * {p.value:>3s}: exp={val:7.2f}, r_p(a)={rnk:.2f}")

        state.advance_one_pick()
        return best_row

    # ---------- Internals ----------

    def _recommend_for_team(
        self,
        state: DraftState,
        team_idx: int,
    ) -> Tuple[Optional[CandidateRow], Optional[NextTurnBreakdown], List[CandidateRow]]:
        """
        Core evaluation: compute Utility = Δ_now(a) + Δ_next^(k)(a) for feasible candidates,
        return the best, plus optional breakdown and a top-5 list for verbose printing.
        """
        candidates = self._feasible_candidates_for_team(state, team_idx)
        if not candidates:
            return None, None, []

        # Expected removals λ_p between now and our next pick
        lam = self._expected_removals_between_our_picks(state, team_idx)

        # Evaluate candidates
        rows: List[CandidateRow] = []
        per_pos_debug_for_best: Optional[NextTurnBreakdown] = None
        best: Optional[CandidateRow] = None

        # Precompute Δ_now for speed (vectorized-ish)
        delta_now_map = value_now_for_candidates(state, self.pool, self.config, team_idx, [p.uid for p in candidates])

        for pl in candidates:
            dn = delta_now_map.get(pl.uid, 0.0)

            # Build m_p^{(a)} lists after taking a = pl
            m_by_pos, _ = top_marginals_by_position_after_pick(
                state=state,
                pool=self.pool,
                config=self.config,
                team_idx=team_idx,
                a_uid=pl.uid,
                per_pos_k=5,  # enough for interpolation around r in typical cases
            )

            # Compute r_p^(a) and interpolated expected next values by position
            rank_by_pos: Dict[Position, float] = {}
            exp_by_pos: Dict[Position, float] = {}
            for p, vals in m_by_pos.items():
                rp = 1.0 + (1.0 if p == pl.position else 0.0) + lam.get(p, 0.0)
                rank_by_pos[p] = rp
                exp_by_pos[p] = interpolate_at_rank(vals, rp) if vals else 0.0

            dnext = max(exp_by_pos.values()) if exp_by_pos else 0.0
            util = dn + dnext

            row = CandidateRow(
                uid=pl.uid,
                name=pl.name,
                team=pl.team,
                position=pl.position,
                value_now=dn,
                exp_next=dnext,
                utility=util,
                adp=pl.proj.adp,
                mu=pl.proj.mu,
                vor=pl.proj.vor,
            )
            rows.append(row)

            if (best is None) or (row.utility > best.utility):
                best = row
                if self.config.verbosity >= Verbosity.DEBUG:
                    per_pos_debug_for_best = NextTurnBreakdown(exp_by_pos=exp_by_pos, rank_by_pos=rank_by_pos)

        # Sort for top-5 printing
        rows.sort(key=lambda r: r.utility, reverse=True)
        top5 = rows[:5]

        return best, per_pos_debug_for_best, top5


    def _feasible_candidates_for_team(self, state: DraftState, team_idx: int) -> List:
        """
        Build the candidate pool for the team on the clock, with HARD FEASIBILITY:
        - A candidate is kept only if, after taking them, the team can still fill
        all REQUIRED starters with its remaining picks.
        - Also applies position caps and early K/DST gate.
        - Pre-trims by a fast baseline score to limit evaluation cost.
        """
        cfg = self.config
        rules = cfg.league.rules
        cap_for = cfg.cap_for
        gate_kdst = cfg.engine.opponent_model.kdst_gate

        roster = state.rosters[team_idx]
        taken = state.drafted_uids

        # Compute must-pick window flag (if remaining picks == required still to fill)
        need, flex_left = _required_starter_needs(state, team_idx)
        L = _picks_remaining_for_team(state, team_idx)
        req_total = int(sum(need.values()) + flex_left)
        must_pick_window = (L == req_total)

        pool_list = []
        for uid, pl in self.pool.by_uid.items():
            if uid in taken:
                continue

            # Cap per position
            cap = cap_for(pl.position)
            if cap is not None and roster.count(pl.position) >= cap:
                continue

            # Early K/DST gate
            if gate_kdst and pl.position in (Position.K, Position.DST) and not _core_starters_filled(state, team_idx):
                # still allow if we're in a must-pick window that requires K/DST
                if not must_pick_window or need.get(pl.position, 0) <= 0:
                    continue

            # HARD FEASIBILITY: skip if taking this player would make it impossible
            # to finish required slots in remaining picks.
            if not _would_pick_be_feasible(state, team_idx, pl.position):
                continue

            # If we are in a must-pick window, only allow picks that reduce required needs
            # (i.e., dedicated need at that position > 0, or FLEX-eligible when flex_left > 0)
            if must_pick_window:
                reduces_dedicated = (pl.position in need and need[pl.position] > 0)
                reduces_flex = (pl.position in rules.flex_positions and flex_left > 0)
                if not (reduces_dedicated or reduces_flex):
                    continue

            pool_list.append(pl)

        if not pool_list:
            return []

        # Pre-trim by a fast baseline score (μ with tiny ADP nudge)
        def baseline(pl):
            mu = float(pl.proj.mu or 0.0)
            adp = pl.proj.adp
            adp_bonus = (0.001 / max(1.0, adp)) if adp is not None else 0.0
            return mu + adp_bonus

        pool_list.sort(key=baseline, reverse=True)
        return pool_list[: int(cfg.engine.candidate_pool_size)]

    def _expected_removals_between_our_picks(self, state: DraftState, team_idx: int) -> Dict[Position, float]:
        """
        Compute λ_p = expected number of players at position p removed between our
        current pick and our next pick (exclusive of our own picks).
        Sum P_t^{(q)}(p) across each opponent pick q in that span.
        """
        lam: Dict[Position, float] = {p: 0.0 for p in Position}
        n = state.league.league_size
        current_pick = state.pick_number
        next_pick = next_pick_index_for_team(current_pick, n, team_idx)

        # For each pick between (current_pick, next_pick), add that team's position probabilities
        for q in range(current_pick + 1, next_pick):
            opp_team = team_on_the_clock_at(q, n)
            # Skip our own team defensively
            if opp_team == team_idx:
                continue

            probs = position_probabilities_for_team(
                state=state,
                pool=self.pool,
                config=self.config,
                team_idx=opp_team,
                pick_index=q,
            )
            for p, pr in probs.items():
                lam[p] += float(pr)

        return lam
