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
        Build the candidate pool for the team on the clock:
        - Must be available (not drafted).
        - Respect per-position caps if configured.
        - Optionally gate K/DST until core starters filled (reuse opponent kdst_gate flag).
        - Reduce to top N by a simple baseline score before deep evaluation, for speed.
        """
        cfg = self.config
        rules = cfg.league.rules
        cap_for = cfg.cap_for
        gate_kdst = cfg.engine.opponent_model.kdst_gate

        roster = state.rosters[team_idx]
        taken = state.drafted_uids

        # Collect available players, apply caps & gates
        pool_list = []
        for uid, pl in self.pool.by_uid.items():
            if uid in taken:
                continue

            # Cap per position
            cap = cap_for(pl.position)
            if cap is not None and roster.count(pl.position) >= cap:
                continue

            # Gate K/DST if core starters not yet filled
            if gate_kdst and pl.position in (Position.K, Position.DST) and not _core_starters_filled(state, team_idx):
                continue

            pool_list.append(pl)

        if not pool_list:
            return []

        # Pre-trim the list to candidate_pool_size by a fast baseline score (mu or vor)
        def baseline(pl):
            mu = float(pl.proj.mu or 0.0)
            vor = pl.proj.vor
            base = float(vor) if (cfg.engine.value_model.use_vor and vor is not None) else mu
            # Nudge by ADP (earlier ADP slightly favored to break ties, but tiny)
            adp_bonus = 0.0
            if pl.proj.adp is not None:
                adp_bonus = 0.001 / max(1.0, pl.proj.adp)
            return base + adp_bonus

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
