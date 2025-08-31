# src/fantasy/opponent_behavior.py
from __future__ import annotations

"""
Opponent position-choice model P_t^{(k)}(p) using a combined VOR+ADP+roster-need approach.

What this module does
---------------------
- Rational pressure (per team, per position) uses the *best available VOR* at that position,
  scaled by how much that team *needs* the position (including FLEX share).
- ADP pressure is computed as *quantile-local mass* around the current pick quantile, so it's
  league-size agnostic and reflects ESPN-like ordering without hard-coded round priors.
- ADP mass is down-weighted for positions the team does not need right now (demand-aware).
- Backup discouragement: strongly discourage QB/TE backups before core starters are filled.
- Hard feasibility + must-pick windows: an opponent cannot choose positions that would make it
  impossible to complete starters on time; when picks-left equals required-slots-left, the
  probability collapses onto the required positions.
- K/DST are gated until core starters are filled (unless must-pick window requires them).

Public entry point:
    position_probabilities_for_team(state, pool, config, team_idx, pick_index=None) -> Dict[Position,float]
"""

from math import exp
from typing import Dict, List, Optional

from .models import (
    DraftEngineConfig,
    DraftState,
    Player,
    PlayerPool,
    Position,
)

# ------------------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------------------

def _softmax(scores: Dict[Position, float], tau: float) -> Dict[Position, float]:
    if not scores:
        return {}
    if tau <= 0:
        mpos = max(scores, key=lambda p: scores[p])
        return {p: (1.0 if p == mpos else 0.0) for p in scores}
    exps = {p: exp(tau * s) for p, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    return {p: exps[p] / Z for p in exps}

def _total_slots_per_team(state: DraftState) -> int:
    rules = state.league.rules
    return int(sum(rules.starters.values()) + rules.bench)

def _total_draft_picks(state: DraftState) -> int:
    return state.league.league_size * _total_slots_per_team(state)

def _pick_quantile(state: DraftState, pick_index: int) -> float:
    """Map absolute 1-based pick_index to [0,1] quantile of the entire draft."""
    T = max(1, _total_draft_picks(state))
    q = (pick_index - 0.5) / float(T)
    return max(0.0, min(1.0, q))

# ------------------------------------------------------------------------------
# Roster need / feasibility accounting
# ------------------------------------------------------------------------------

def _flex_remaining_for_team(state: DraftState, team_idx: int) -> int:
    rules = state.league.rules
    flex_need = int(rules.starters.get("FLEX", 0))
    if flex_need <= 0:
        return 0
    roster = state.rosters[team_idx]
    rb_ex = max(0, roster.count(Position.RB) - rules.starter_slots_for(Position.RB))
    wr_ex = max(0, roster.count(Position.WR) - rules.starter_slots_for(Position.WR))
    te_ex = max(0, roster.count(Position.TE) - rules.starter_slots_for(Position.TE))
    filled = min(flex_need, rb_ex + wr_ex + te_ex)
    return max(0, flex_need - filled)

def _gap_vector_for_team(state: DraftState, team_idx: int) -> Dict[Position, float]:
    rules = state.league.rules
    roster = state.rosters[team_idx]
    gaps: Dict[Position, float] = {p: 0.0 for p in Position}
    for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
        need = rules.starter_slots_for(p)
        have = roster.count(p)
        gaps[p] = float(max(0, need - have))
    # FLEX share across eligible positions
    flex_left = _flex_remaining_for_team(state, team_idx)
    if flex_left > 0 and rules.flex_positions:
        share = float(flex_left) / float(len(rules.flex_positions))
        for p in rules.flex_positions:
            gaps[p] += share
    return gaps

def _core_starters_filled(state: DraftState, team_idx: int) -> bool:
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
        rb_ex = max(0, roster.count(Position.RB) - rules.starter_slots_for(Position.RB))
        wr_ex = max(0, roster.count(Position.WR) - rules.starter_slots_for(Position.WR))
        te_ex = max(0, roster.count(Position.TE) - rules.starter_slots_for(Position.TE))
        have += min(flex_need, rb_ex + wr_ex + te_ex)
    return have >= needed

def _picks_remaining_for_team(state: DraftState, team_idx: int) -> int:
    total_slots = _total_slots_per_team(state)
    current = len(state.rosters[team_idx].players)
    return max(0, total_slots - current)

def _required_starter_needs(state: DraftState, team_idx: int) -> tuple[dict[Position, int], int]:
    rules = state.league.rules
    roster = state.rosters[team_idx]
    need: dict[Position, int] = {p: 0 for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K)}
    for p in need.keys():
        need[p] = max(0, int(rules.starter_slots_for(p)) - roster.count(p))
    flex_total = int(rules.starters.get("FLEX", 0))
    if flex_total <= 0:
        flex_left = 0
    else:
        rb_ex = max(0, roster.count(Position.RB) - int(rules.starter_slots_for(Position.RB)))
        wr_ex = max(0, roster.count(Position.WR) - int(rules.starter_slots_for(Position.WR)))
        te_ex = max(0, roster.count(Position.TE) - int(rules.starter_slots_for(Position.TE)))
        flex_left = max(0, flex_total - (rb_ex + wr_ex + te_ex))
    return need, flex_left

def _position_pick_feasible(state: DraftState, team_idx: int, pos: Position) -> bool:
    """If this team picked position `pos` now, could they still finish all required starters?"""
    L = _picks_remaining_for_team(state, team_idx)  # includes current pick
    need, flex_left = _required_starter_needs(state, team_idx)
    # Apply hypothetical pick
    if pos in need and need[pos] > 0:
        need[pos] -= 1
    elif pos in state.league.rules.flex_positions and flex_left > 0:
        flex_left -= 1
    req_after = int(sum(need.values()) + flex_left)
    return req_after <= max(0, L - 1)

def _apply_kdst_gate(
    probs: Dict[Position, float],
    state: DraftState,
    config: DraftEngineConfig,
    team_idx: int,
) -> Dict[Position, float]:
    out = probs.copy()
    if config.engine.opponent_model.kdst_gate and not _core_starters_filled(state, team_idx):
        out[Position.K] = 0.0
        out[Position.DST] = 0.0
    Z = sum(out.values()) or 1.0
    return {p: out[p] / Z for p in out}

# ------------------------------------------------------------------------------
# ADP (quantile-local, league-size agnostic) + demand scaling
# ------------------------------------------------------------------------------

def _compute_adp_quantiles(state: DraftState, pool: PlayerPool) -> dict[str, float]:
    """Quantile of ADP among *undrafted* players. Lower ADP -> lower quantile."""
    taken = state.drafted_uids
    avail = [pl for pl in pool.by_uid.values() if (pl.uid not in taken and pl.proj.adp is not None)]
    if not avail:
        return {}
    avail.sort(key=lambda pl: float(pl.proj.adp))  # earliest ADP first
    N = len(avail)
    qmap: dict[str, float] = {}
    for rank, pl in enumerate(avail, start=1):
        qmap[pl.uid] = rank / float(N)  # 0..1-ish
    return qmap

def _team_demand_scale(state: DraftState, team_idx: int) -> Dict[Position, float]:
    """
    Scale ADP mass by whether this team needs the position now.
    1.0 if dedicated need > 0; 0.85 if FLEX could use it; else downweight.
    """
    need, flex_left = _required_starter_needs(state, team_idx)
    scale = {p: 1.0 for p in Position}
    for p in Position:
        if need.get(p, 0) > 0:
            scale[p] = 1.0
        elif p in state.league.rules.flex_positions and flex_left > 0:
            scale[p] = 0.85
        else:
            scale[p] = 0.35 if p in (Position.RB, Position.WR, Position.TE) else 0.20
    return scale

def _adp_pressure_by_position_quantile_local(
    state: DraftState,
    pool: PlayerPool,
    pick_index: int,
    q_sigma: float = 0.05,
    q_window: float = 0.10,
    demand_scale: Optional[Dict[Position, float]] = None,
) -> Dict[Position, float]:
    """
    ADP pressure as *quantile-local mass* centered at the current pick quantile.
    """
    q_pick = _pick_quantile(state, pick_index)
    qmap = _compute_adp_quantiles(state, pool)
    pos_mass: Dict[Position, float] = {p: 0.0 for p in Position}
    if not qmap:
        return pos_mass

    def kernel(dq: float) -> float:
        x = dq / max(1e-6, q_sigma)
        return float(exp(-0.5 * x * x))

    taken = state.drafted_uids
    for uid, q_player in qmap.items():
        if uid in taken:
            continue
        pl = pool.by_uid[uid]
        dq = abs(q_player - q_pick)
        if dq <= q_window:
            pos_mass[pl.position] += kernel(dq)

    if demand_scale:
        for p in pos_mass:
            pos_mass[p] *= float(demand_scale.get(p, 1.0))

    return pos_mass

# ------------------------------------------------------------------------------
# Rational pressure (VOR × need) + backup discouragement
# ------------------------------------------------------------------------------

def _best_vor_by_position(state: DraftState, pool: PlayerPool) -> Dict[Position, float]:
    """
    For each position, find the best available 'value' where value = VOR if present,
    else μ as a fallback (so module stays usable if some rows miss VOR).
    """
    taken = state.drafted_uids
    best: Dict[Position, float] = {p: 0.0 for p in Position}
    for pl in pool.by_uid.values():
        if pl.uid in taken:
            continue
        vor = pl.proj.vor
        val = float(vor) if vor is not None else float(pl.proj.mu or 0.0)
        if val > best[pl.position]:
            best[pl.position] = val
    return best

def _need_scaler(gap: float, beta: float) -> float:
    """Gentle linear urgency; gap is fractional (includes FLEX share)."""
    return 1.0 + beta * max(0.0, float(gap))

def _backup_penalty(state: DraftState, team_idx: int, pos: Position) -> float:
    """
    Discourage backups for QB/TE before core starters are filled.
    """
    roster = state.rosters[team_idx]
    have = roster.count(pos)
    if pos in (Position.QB, Position.TE) and have >= 1 and not _core_starters_filled(state, team_idx):
        return 0.1
    return 1.0

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def position_probabilities_for_team(
    state: DraftState,
    pool: PlayerPool,
    config: DraftEngineConfig,
    team_idx: int,
    pick_index: Optional[int] = None,
) -> Dict[Position, float]:
    """
    Compute P_t^{(k)}(p) mixing:
      - Rational: need-weighted best VOR at position p (backup discouraged)
      - ADP: quantile-local mass near current pick, down-weighted if team doesn't need p
      - Feasibility masking and must-pick window collapse
      - K/DST gate until core starters are filled

    Returns probabilities summing to 1.0.
    """
    if pick_index is None:
        pick_index = state.pick_number

    params = config.engine.opponent_model

    # 1) Rational pressure: best VOR at position × need × backup penalty
    best_vor = _best_vor_by_position(state, pool)
    gaps = _gap_vector_for_team(state, team_idx)

    rat_scores: Dict[Position, float] = {}
    for p in Position:
        base = best_vor.get(p, 0.0)
        s = _need_scaler(gaps.get(p, 0.0), params.gap_beta) * base
        s *= _backup_penalty(state, team_idx, p)
        rat_scores[p] = s

    # 2) ADP pressure: quantile-local mass + demand scaling
    demand_scale = _team_demand_scale(state, team_idx)
    adp_scores = _adp_pressure_by_position_quantile_local(
        state=state,
        pool=pool,
        pick_index=pick_index,
        q_sigma=0.05,
        q_window=0.10,
        demand_scale=demand_scale,
    )

    # 3) Mixture
    mixed_scores: Dict[Position, float] = {}
    for p in Position:
        s = (1.0 - params.eta) * rat_scores.get(p, 0.0) + params.eta * adp_scores.get(p, 0.0)
        mixed_scores[p] = s

    # 4) Softmax
    probs = _softmax(mixed_scores, tau=params.tau)

    # 5) HARD FEASIBILITY MASK + MUST-PICK WINDOW
    need, flex_left = _required_starter_needs(state, team_idx)
    L = _picks_remaining_for_team(state, team_idx)
    req_total = int(sum(need.values()) + flex_left)
    must_pick_window = (L == req_total)

    feasible_masked: Dict[Position, float] = {}
    for p, pr in probs.items():
        feasible_masked[p] = (pr if _position_pick_feasible(state, team_idx, p) else 0.0)

    if must_pick_window:
        for p in Position:
            reduces_dedicated = (p in need and need[p] > 0)
            reduces_flex = (p in state.league.rules.flex_positions and flex_left > 0)
            if not (reduces_dedicated or reduces_flex):
                feasible_masked[p] = 0.0

    Z = sum(feasible_masked.values())
    if Z <= 0.0:
        # Failsafe fallback
        fallback: Dict[Position, float] = {p: 0.0 for p in Position}
        if req_total > 0:
            has_ded = any(need[p] > 0 for p in need)
            if has_ded:
                for p in need:
                    if need[p] > 0:
                        fallback[p] = 1.0
            else:
                for p in state.league.rules.flex_positions:
                    fallback[p] = 1.0
        else:
            fallback = probs.copy()
        Z2 = sum(fallback.values()) or 1.0
        probs = {p: fallback[p] / Z2 for p in fallback}
    else:
        probs = {p: feasible_masked[p] / Z for p in feasible_masked}

    # 6) K/DST gate
    probs = _apply_kdst_gate(probs, state, config, team_idx)
    return probs
