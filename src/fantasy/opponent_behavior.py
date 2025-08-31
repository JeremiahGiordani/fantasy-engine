# src/fantasy/opponent_behavior.py
from __future__ import annotations

"""
Opponent position-choice model P_t^{(k)}(p).

Implements the hybrid methodology from Section 2.4:
- Rational pressure: need(gap) × best-available-value-for-them at each position.
- Market pressure: ADP-derived density mass at the (upcoming) pick, aggregated by position.
- Mixture of the two, then softmax with temperature (plus gating/constraints).

This module intentionally models ONLY position probabilities, not specific players.
The DraftEngine will aggregate these per-opponent probabilities into expected
removals by position between our current pick and our next pick.

Public entry point:
    position_probabilities_for_team(state, pool, config, team_idx, pick_index)

Notes:
- 'pick_index' should be the absolute pick number (1-based) at which 'team_idx'
  will make the contemplated pick. The engine will pass the appropriate pick
  number for each opponent between our picks.
- K/DST "gates" are enforced: if a team hasn't filled core starters yet,
  K/DST probability is zeroed (configurable).
"""

from dataclasses import dataclass
from math import exp, sqrt
from typing import Dict, List, Optional, Tuple

from fantasy.models import (
    DraftState,
    DraftEngineConfig,
    Player,
    PlayerPool,
    Position,
)


# -----------------------------
# Small helpers
# -----------------------------

def _player_value_for_team(player: Player, config: DraftEngineConfig) -> float:
    """
    A quick scalar proxy for how valuable 'player' would be to an arbitrary team,
    used inside the rational pressure term.
    We intentionally avoid doing a full Δ_now computation here for speed.
    """
    vm = config.engine.value_model
    mu = float(player.proj.mu or 0.0)
    vor = player.proj.vor
    sigma = float(player.proj.sigma or 0.0)
    base = float(vor) if (vm.use_vor and vor is not None) else mu
    return base - vm.risk_lambda * sigma


def _available_by_position(state: DraftState, pool: PlayerPool) -> Dict[Position, List[Player]]:
    by_pos: Dict[Position, List[Player]] = {p: [] for p in Position}
    taken = state.drafted_uids
    for uid, pl in pool.by_uid.items():
        if uid in taken:
            continue
        by_pos[pl.position].append(pl)
    return by_pos


def _softmax(scores: Dict[Position, float], tau: float) -> Dict[Position, float]:
    if tau <= 0:
        # Pick argmax deterministically
        max_pos = max(scores, key=lambda p: scores[p]) if scores else None
        return {p: (1.0 if p == max_pos else 0.0) for p in scores}
    exps = {p: exp(tau * s) for p, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    return {p: exps[p] / Z for p in scores}


# -----------------------------
# Need (gap) computation
# -----------------------------

def _flex_remaining_for_team(state: DraftState, team_idx: int, include_qb_te=False) -> int:
    """
    Estimate remaining FLEX slots for the team that are not plausibly filled yet.
    We keep this simple: required FLEX slots minus the count of FLEX-eligible players
    currently on roster that exceed their dedicated starters. This is a heuristic.
    """
    rules = state.league.rules
    flex_need = int(rules.starters.get("FLEX", 0))
    if flex_need <= 0:
        return 0

    roster = state.rosters[team_idx]
    rb = roster.count(Position.RB)
    wr = roster.count(Position.WR)
    te = roster.count(Position.TE)

    # Dedicated starter needs per position
    rb_need = rules.starter_slots_for(Position.RB)
    wr_need = rules.starter_slots_for(Position.WR)
    te_need = rules.starter_slots_for(Position.TE)

    # Count "excess" FLEX-eligible above dedicated starters
    rb_excess = max(0, rb - rb_need)
    wr_excess = max(0, wr - wr_need)
    te_excess = max(0, te - te_need)

    flex_already_filled = min(flex_need, rb_excess + wr_excess + te_excess)
    return max(0, flex_need - flex_already_filled)


def _gap_vector_for_team(state: DraftState, team_idx: int) -> Dict[Position, float]:
    """
    Build a simple gap (need) per position based on remaining dedicated starters
    plus a proportional share of remaining FLEX demand for RB/WR/TE.
    """
    rules = state.league.rules
    roster = state.rosters[team_idx]

    gaps: Dict[Position, float] = {p: 0.0 for p in Position}
    # Dedicated starter gaps
    for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
        need = rules.starter_slots_for(p)
        have = roster.count(p)
        gaps[p] = float(max(0, need - have))

    # FLEX share
    flex_left = _flex_remaining_for_team(state, team_idx)
    if flex_left > 0:
        flex_positions = list(rules.flex_positions)
        if flex_positions:
            share = float(flex_left) / float(len(flex_positions))
            for p in flex_positions:
                gaps[p] += share

    return gaps


def _phi_need(gap: float, beta: float) -> float:
    """Concave-ish urgency map for positional gaps."""
    # Simple linear with floor at 0; could be log(1+beta*gap) if you prefer.
    return 1.0 + beta * max(0.0, float(gap))


# -----------------------------
# ADP density per position at a pick
# -----------------------------

def _adp_density_at_pick(adp: float, pick_index: int, sigma: float) -> float:
    """
    Gaussian-like density centered at ADP. Units are "picks".
    This is NOT a probability density function in a formal sense across all picks,
    but a convenient pressure signal near the current pick index.
    """
    if adp is None:
        return 0.0
    x = (float(pick_index) - float(adp)) / max(1e-6, sigma)
    return float(exp(-0.5 * x * x))


def _adp_pressure_by_position(
    state: DraftState,
    pool: PlayerPool,
    pick_index: int,
    sigma: float,
) -> Dict[Position, float]:
    """
    Sum ADP densities of all available players at their positions for a given pick index.
    """
    taken = state.drafted_uids
    pos_mass: Dict[Position, float] = {p: 0.0 for p in Position}
    for uid, pl in pool.by_uid.items():
        if uid in taken:
            continue
        adp = pl.proj.adp
        if adp is None:
            continue
        d = _adp_density_at_pick(adp, pick_index, sigma)
        pos_mass[pl.position] += d
    return pos_mass


# -----------------------------
# Gating / constraints
# -----------------------------

def _core_starters_filled(state: DraftState, team_idx: int) -> bool:
    """
    Returns True if the team has filled all non-K/DST starters (QB,RB,WR,TE,FLEX).
    """
    rules = state.league.rules
    roster = state.rosters[team_idx]
    needed = rules.starters_required_non_kdst()

    have = 0
    have += min(roster.count(Position.QB), rules.starter_slots_for(Position.QB))
    have += min(roster.count(Position.RB), rules.starter_slots_for(Position.RB))
    have += min(roster.count(Position.WR), rules.starter_slots_for(Position.WR))
    have += min(roster.count(Position.TE), rules.starter_slots_for(Position.TE))

    # FLEX: count any excess RB/WR/TE beyond dedicated starters (up to FLEX)
    flex_need = int(rules.starters.get("FLEX", 0))
    if flex_need > 0:
        rb_excess = max(0, roster.count(Position.RB) - rules.starter_slots_for(Position.RB))
        wr_excess = max(0, roster.count(Position.WR) - rules.starter_slots_for(Position.WR))
        te_excess = max(0, roster.count(Position.TE) - rules.starter_slots_for(Position.TE))
        have += min(flex_need, rb_excess + wr_excess + te_excess)

    return have >= needed


def _apply_gates(
    probs: Dict[Position, float],
    state: DraftState,
    config: DraftEngineConfig,
    team_idx: int,
) -> Dict[Position, float]:
    """
    Enforce gates/caps: zero out K/DST if core starters not filled (if enabled).
    Then renormalize.
    """
    out = probs.copy()
    if config.engine.opponent_model.kdst_gate and not _core_starters_filled(state, team_idx):
        out[Position.K] = 0.0
        out[Position.DST] = 0.0

    total = sum(out.values()) or 1.0
    return {p: out[p] / total for p in out}


# -----------------------------
# Public API
# -----------------------------

def position_probabilities_for_team(
    state: DraftState,
    pool: PlayerPool,
    config: DraftEngineConfig,
    team_idx: int,
    pick_index: Optional[int] = None,
) -> Dict[Position, float]:
    """
    Compute P_t^{(k)}(p) for team 'team_idx' at the contemplated pick index.

    Args:
        state: DraftState at the time of evaluating this opponent.
        pool: PlayerPool (available players known via state.drafted_uids).
        config: DraftEngineConfig.
        team_idx: the opponent team index.
        pick_index: absolute pick number (1-based). If None, we default to the
                    current state's pick_number, but typically the engine will
                    pass the specific upcoming pick index for this team.

    Returns:
        Dict[Position, probability] summing to 1.0 (after gating/renorm).
    """
    if pick_index is None:
        pick_index = state.pick_number

    params = config.engine.opponent_model
    rules = state.league.rules

    # 1) Rational pressure: need(gap) × best value available at each position.
    gaps = _gap_vector_for_team(state, team_idx)
    avail = _available_by_position(state, pool)

    rat_scores: Dict[Position, float] = {}
    for p, players in avail.items():
        if not players:
            rat_scores[p] = 0.0
            continue
        best_val = max((_player_value_for_team(pl, config) for pl in players), default=0.0)
        rat_scores[p] = _phi_need(gaps.get(p, 0.0), params.gap_beta) * max(0.0, best_val)

    # 2) ADP pressure: sum densities at this pick, per position.
    # sigma in "picks": scale by teams-per-round (league size).
    n = state.league.league_size
    sigma_picks = max(1.0, params.adp_sigma * float(n))
    adp_scores = _adp_pressure_by_position(state, pool, pick_index, sigma=sigma_picks)

    # 3) Mixture
    mixed_scores: Dict[Position, float] = {}
    for p in Position:
        mixed_scores[p] = (1.0 - params.eta) * rat_scores.get(p, 0.0) + params.eta * adp_scores.get(p, 0.0)

    # 4) Softmax to probabilities
    probs = _softmax(mixed_scores, tau=params.tau)

    # 5) Gates / constraints (e.g., no K/DST until core filled)
    probs = _apply_gates(probs, state, config, team_idx)

    return probs
