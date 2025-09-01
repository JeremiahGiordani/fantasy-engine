# src/fantasy/draft_engine.py
from __future__ import annotations

"""
DraftEngine: orchestrates candidate evaluation, one-step lookahead, and pick execution.

Responsibilities
---------------
- Build the feasible candidate set for the team on the clock (respecting caps/gates).
- For each candidate a:
    Δ_now(a) = J(R ∪ {a}) - J(R)
    Non-corner: Δ_next^(k)(a) ≈ max_p  m̃_p^(a)(r_p^(a)),  with  r_p^(a) = 1 + 1{p=π(a)} + λ_p
    Corner   : Utility(a1) = Δ_now(a1)
                            + max_p m̃_p^(a1)(1 + 1{p=π(a1)})                    [best immediate 2nd]
                            + max_p m̃_p^(a1)(1 + 1{p=π(a2*)} + λ_p_postpair)     [next-turn after pair]
- Choose the candidate with maximal Utility and apply the pick.

Verbosity
---------
- 0 (QUIET): no printing
- 1 (PICKS): one line per pick with key numbers
- 2 (DEBUG): plus top-5 candidate table and per-position breakdown (values and ranks)
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


def _is_back_to_back(state: DraftState, team_idx: int) -> bool:
    """
    True if this team picks again immediately next (corner of the snake).
    """
    n = state.league.league_size
    cur = state.pick_number
    nxt = next_pick_index_for_team(cur, n, team_idx)
    return (nxt == cur + 1)


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


def _k_needed_from_lambda(lam: Dict[Position, float], primary_bonus: int = 1, margin: int = 2, cap: int = 24) -> int:
    """
    We will evaluate ranks of the form r_p = 1 + (primary?primary_bonus:0) + lam[p].
    Ensure per_pos_k >= ceil(max_p r_p) + margin, but cap to keep runtime sane.
    """
    import math
    max_r = 0.0
    for p, lp in lam.items():
        r = 1.0 + float(lp) + (primary_bonus if p is not None else 0)
        if r > max_r:
            max_r = r
    k = int(math.ceil(max_r) + margin)
    return max(5, min(k, cap))


# --- AGGREGATED EXPECTED REMOVALS (ADP × aggregated roster demand) ---

def _aggregate_expected_removals(
    state: DraftState,
    pool: PlayerPool,
    start_pick: int,   # exclusive
    end_pick: int,     # exclusive
    my_team_idx: int,
) -> Dict[Position, float]:
    """
    Compute λ_p over (start_pick, end_pick) as an aggregate:
      - Demand side: sum required-starter needs of *opponents in span* (dedicated + FLEX share)
      - Supply side: ADP mass among the top (M*2) available players by ADP, per position
      - λ_p = M * [ (Demand_p^α * ADPmass_p^β) / Σ_q (Demand_q^α * ADPmass_q^β) ]
    where M is number of opponent picks in the interval.
    """
    n_teams = state.league.league_size
    M = 0
    teams_in_span: List[int] = []
    for q in range(start_pick + 1, end_pick):
        t = team_on_the_clock_at(q, n_teams)
        if t == my_team_idx:
            continue
        teams_in_span.append(t)
        M += 1
    if M <= 0:
        return {p: 0.0 for p in Position}

    # -------- Demand: aggregate required starters for teams in span --------
    # dedicated needs + equal FLEX share across eligible positions
    demand: Dict[Position, float] = {p: 0.0 for p in Position}
    flex_positions = set(state.league.rules.flex_positions)

    for t in teams_in_span:
        need, flex_left = _required_starter_needs(state, t)
        for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
            demand[p] += float(max(0, need.get(p, 0)))
        if flex_left > 0 and flex_positions:
            share = float(flex_left) / float(len(flex_positions))
            for p in flex_positions:
                demand[p] += share

    # Light gating: if many teams still lack core starters, downweight K/DST demand a bit
    core_unfilled = 0
    for t in teams_in_span:
        if not _core_starters_filled(state, t):
            core_unfilled += 1
    if core_unfilled > 0:
        demand[Position.K] *= 0.35
        demand[Position.DST] *= 0.35

    # Discourage early backup QB/TE demand somewhat (aggregate version)
    # If average QB/TE owned >= 1 across span, shrink backup demand a bit
    # (crude but stabilizing)
    avg_qb_owned = sum(state.rosters[t].count(Position.QB) for t in teams_in_span) / max(1, len(teams_in_span))
    avg_te_owned = sum(state.rosters[t].count(Position.TE) for t in teams_in_span) / max(1, len(teams_in_span))
    if avg_qb_owned >= 1.0:
        demand[Position.QB] *= 0.7
    if avg_te_owned >= 1.0:
        demand[Position.TE] *= 0.8

    # -------- Supply: ADP mass among top available players --------
    taken = state.drafted_uids
    avail = [pl for pl in pool.by_uid.values() if pl.uid not in taken]
    # Sort by ADP (missing ADP treated as very late)
    def _adp_key(pl):
        return float(pl.proj.adp) if pl.proj.adp is not None else 1e9
    avail.sort(key=_adp_key)

    topN = max(15, M * 2)  # look a bit beyond span size
    top_avail = avail[:topN]

    adp_mass: Dict[Position, float] = {p: 0.0 for p in Position}
    # Weighting: closer ADP -> larger mass. Use 1/(1+ADP) as a simple monotone weight.
    for pl in top_avail:
        adp = pl.proj.adp if pl.proj.adp is not None else 1e9
        w = 1.0 / (1.0 + float(adp))
        adp_mass[pl.position] += w

    # If a position has zero ADP mass (e.g., no top players left), give a tiny epsilon
    for p in adp_mass:
        if adp_mass[p] <= 0.0:
            adp_mass[p] = 1e-6

    # -------- Combine demand & supply --------
    alpha = 1.0  # demand exponent
    beta = 1.0   # ADP mass exponent
    raw: Dict[Position, float] = {}
    for p in Position:
        raw[p] = (demand.get(p, 0.0) ** alpha) * (adp_mass.get(p, 0.0) ** beta)

    tot = sum(raw.values())
    if tot <= 0.0:
        # Fallback: split evenly across skill positions
        base = M / 4.0
        return {
            Position.QB: base,
            Position.RB: base,
            Position.WR: base,
            Position.TE: base,
            Position.DST: 0.0,
            Position.K: 0.0,
        }

    lam = {p: (M * raw[p] / tot) for p in Position}
    return lam



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

        best_row, breakdown, top5, rows = self._recommend_for_team(state, team_idx)
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

            shown_positions = {r.position for r in top5}
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

        state.advance_one_pick()
        return best_row

    # ---------- Internals ----------
    def _recommend_for_team(
        self,
        state: DraftState,
        team_idx: int,
    ) -> Tuple[Optional[CandidateRow], Optional[NextTurnBreakdown], List[CandidateRow], List[CandidateRow]]:
        """
        Core evaluation:
        - Non-corner: Utility = Δ_now(a) + (need-weighted expectation over positions)
        - Corner    : Utility = Δ_now(a1) + best_immediate_second(a1) + weighted next-turn after the pair
        where the next-turn component is computed with aggregate λ over the interval after our second pick.
        DEBUG breakdown now shows **RAW** next-turn expectations: evaluated at r_raw = 1 + λ_p
        using marginal lists built from the current state (no hypothetical pick applied).
        Returns: (best_row, debug_breakdown, top5, rows)
        """
        candidates = self._feasible_candidates_for_team(state, team_idx)
        if not candidates:
            return None, None, [], []

        # Precompute Δ_now for speed (vectorized-ish)
        delta_now_map = value_now_for_candidates(
            state, self.pool, self.config, team_idx, [p.uid for p in candidates]
        )

        rows: List[CandidateRow] = []
        per_pos_debug_for_best: Optional[NextTurnBreakdown] = None
        best: Optional[CandidateRow] = None

        # Small local helper: build RAW marginal lists (no pick applied) for top-k per position
        def _raw_marginals_for_current_state(per_pos_k: int) -> Dict[Position, List[float]]:
            taken = state.drafted_uids
            avail_uids = [uid for uid in self.pool.by_uid.keys() if uid not in taken]
            # Compute Δ_now for each available player if we added them now (current roster R)
            dn_map = value_now_for_candidates(state, self.pool, self.config, team_idx, avail_uids)
            by_pos: Dict[Position, List[float]] = {p: [] for p in Position}
            for uid in avail_uids:
                pl = self.pool.by_uid[uid]
                by_pos[pl.position].append(float(dn_map.get(uid, 0.0)))
            # Sort desc and truncate to k
            for p in by_pos:
                by_pos[p].sort(reverse=True)
                if per_pos_k is not None and per_pos_k > 0:
                    by_pos[p] = by_pos[p][:per_pos_k]
            return by_pos

        # ----- Corner (back-to-back) branch: pair-aware scoring with weighted next-turn -----
        if _is_back_to_back(state, team_idx):
            n = state.league.league_size
            cur_pick = state.pick_number
            next_after_pair = next_pick_index_for_team(cur_pick + 1, n, team_idx)

            # Expected removals AFTER the pair (used for both sizing and post-pair ranks)
            lam_post = self._expected_removals_between_picks(
                state, team_idx, start_pick=cur_pick + 1, end_pick=next_after_pair
            )
            # Depth needed so r_post(p) = 1 + 1{p = p2*} + λ_p sits inside lists
            k_needed_corner = _k_needed_from_lambda(lam_post, primary_bonus=1, margin=2, cap=24)

            # --- DEBUG RAW breakdown (corner): build raw lists from current state, no picks applied
            if self.config.verbosity >= Verbosity.DEBUG:
                m_raw = _raw_marginals_for_current_state(per_pos_k=k_needed_corner)
                raw_rank_by_pos: Dict[Position, float] = {}
                raw_exp_by_pos: Dict[Position, float] = {}
                for p, vals in m_raw.items():
                    r_raw = 1.0 + float(lam_post.get(p, 0.0))  # post-pair window, but no +1 for our picks
                    raw_rank_by_pos[p] = r_raw
                    raw_exp_by_pos[p] = interpolate_at_rank(vals, r_raw) if vals else 0.0
                per_pos_debug_for_best = NextTurnBreakdown(
                    exp_by_pos=raw_exp_by_pos, rank_by_pos=raw_rank_by_pos
                )

            # Scoring loop for candidates (uses actual pair-aware weighted expectation)
            for pl in candidates:
                dn = float(delta_now_map.get(pl.uid, 0.0))

                # Per-position marginal lists AFTER taking a1 = pl (for immediate second + post-pair calc)
                m_by_pos, _ = top_marginals_by_position_after_pick(
                    state=state,
                    pool=self.pool,
                    config=self.config,
                    team_idx=team_idx,
                    a_uid=pl.uid,
                    per_pos_k=k_needed_corner,
                )

                # Best immediate second pick (no opponents between our two picks)
                sec_vals: Dict[Position, float] = {}
                for p, vals in m_by_pos.items():
                    rp2 = 1.0 + (1.0 if p == pl.position else 0.0)
                    sec_vals[p] = interpolate_at_rank(vals, rp2) if vals else 0.0

                if sec_vals:
                    p2_star = max(sec_vals.keys(), key=lambda pos: sec_vals[pos])
                    d2_immediate = sec_vals[p2_star]
                else:
                    p2_star = pl.position
                    d2_immediate = 0.0

                # Next-turn *after the pair* using weighted expectation:
                post_exp_by_pos: Dict[Position, float] = {}
                for p, vals in m_by_pos.items():
                    rp_post = 1.0 + (1.0 if p == p2_star else 0.0) + float(lam_post.get(p, 0.0))
                    post_exp_by_pos[p] = interpolate_at_rank(vals, rp_post) if vals else 0.0

                
                # IMPLEMENTATION WILL GO HERE

                # AS YOU CAN SEE, WE'RE DOING A SIMPLE SUM OVER EXPECTATION

                # REPLACE THESE WITH THE ALGORITHM

                dnext_postpair = sum(post_exp_by_pos.values())

                util = dn + d2_immediate + dnext_postpair

                # END IMLEMENTATION

                row = CandidateRow(
                    uid=pl.uid,
                    name=pl.name,
                    team=pl.team,
                    position=pl.position,
                    value_now=dn,
                    exp_next=dnext_postpair,  # after the pair
                    utility=util,
                    adp=pl.proj.adp,
                    mu=pl.proj.mu,
                    vor=pl.proj.vor,
                )
                rows.append(row)

                if (best is None) or (row.utility > best.utility):
                    best = row

            rows.sort(key=lambda r: r.utility, reverse=True)
            top5 = rows[:5]
            return best, per_pos_debug_for_best, top5, rows

        # ----- Non-corner: one-step lookahead with weighted expectation -----
        lam = self._expected_removals_between_our_picks(state, team_idx)

        # --- DEBUG RAW breakdown (non-corner): build raw lists from current state, no pick applied
        k_needed = _k_needed_from_lambda(lam, primary_bonus=0, margin=2, cap=24)  # primary_bonus=0 for raw
        m_raw = None
        if self.config.verbosity >= Verbosity.DEBUG:
            m_raw = _raw_marginals_for_current_state(per_pos_k=k_needed)
            raw_rank_by_pos: Dict[Position, float] = {}
            raw_exp_by_pos: Dict[Position, float] = {}
            for p, vals in m_raw.items():
                r_raw = 1.0 + float(lam.get(p, 0.0))
                raw_rank_by_pos[p] = r_raw
                raw_exp_by_pos[p] = interpolate_at_rank(vals, r_raw) if vals else 0.0
            per_pos_debug_for_best = NextTurnBreakdown(
                exp_by_pos=raw_exp_by_pos, rank_by_pos=raw_rank_by_pos
            )

        # Scoring loop (uses hypothetical a added because Term-2 needs post-a ranks)
        for pl in candidates:
            dn = float(delta_now_map.get(pl.uid, 0.0))

            k_needed_eval = _k_needed_from_lambda(lam, primary_bonus=1, margin=2, cap=24)
            m_by_pos, _ = top_marginals_by_position_after_pick(
                state=state,
                pool=self.pool,
                config=self.config,
                team_idx=team_idx,
                a_uid=pl.uid,
                per_pos_k=k_needed_eval,
            )

            exp_by_pos: Dict[Position, float] = {}
            for p, vals in m_by_pos.items():
                rp = 1.0 + (1.0 if p == pl.position else 0.0) + lam.get(p, 0.0)
                exp_by_pos[p] = interpolate_at_rank(vals, rp) if vals else 0.0
            
            # IMPLEMENTATION WILL GO HERE

            # AS YOU CAN SEE, WE'RE DOING A SIMPLE SUM OVER EXPECTATION

            # REPLACE THESE WITH THE ALGORITHM

            dnext = sum(exp_by_pos.values())

            util = dn + dnext

            # END IMLEMENTATION

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

        rows.sort(key=lambda r: r.utility, reverse=True)
        top5 = rows[:5]
        return best, per_pos_debug_for_best, top5, rows


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
        Aggregate model: compute λ_p over the entire span to our next pick using ADP mass
        and aggregated roster demand across the opponents in that span.
        """
        n = state.league.league_size
        current_pick = state.pick_number
        next_pick = next_pick_index_for_team(current_pick, n, team_idx)
        return _aggregate_expected_removals(
            state=state,
            pool=self.pool,
            start_pick=current_pick,
            end_pick=next_pick,
            my_team_idx=team_idx,
        )

    def _expected_removals_between_picks(
        self,
        state: DraftState,
        team_idx: int,
        start_pick: int,
        end_pick: int,
    ) -> Dict[Position, float]:
        """
        Aggregate model over an arbitrary interval (start, end), exclusive of endpoints.
        """
        return _aggregate_expected_removals(
            state=state,
            pool=self.pool,
            start_pick=start_pick,
            end_pick=end_pick,
            my_team_idx=team_idx,
        )
