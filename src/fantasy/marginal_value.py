# src/fantasy/marginal_value.py
from __future__ import annotations

"""
Utilities for computing lineup value J(R), marginal value Δ_now(a),
and rank-based marginal functions m_p^{(a)}(r) used in the one-step lookahead.

This module follows the math in the paper sections:
- 2.2 Estimating Marginal Value of a Player
- 2.3 Estimating Expected Value at the Next Turn

Key ideas implemented:
- Expected lineup value J(R) is computed from player projections weighted by
  start-share probabilities S_i(R) that depend on roster composition and slots.
- Start-share S_i(R) is built via:
  (1) deterministic assignment to dedicated starters and FLEX using projections,
  (2) bench start priors by depth (from config),
  (3) optional distribution-overlap adjustment (pairwise Pr[X_i > X_j]) to make
      bench shares responsive to floor/ceiling/uncertainty.

- Δ_now(a) = J(R ∪ {a}) - J(R).
- m_p^{(a)}(r): given that we selected 'a' already, the marginal lineup gain of
  adding the r-th best candidate at position p (ranked by Δ_now relative to R∪{a}).

NOTE: For computational practicality in live-draft settings, we use Normal-based
pairwise win probabilities. Truncation by [floor, ceiling] is acknowledged but
approximated by a moment-matched Normal (sigma fallback if missing).
"""

from dataclasses import dataclass
from math import erf, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from fantasy.models import (
    Player,
    PlayerPool,
    DraftState,
    DraftEngineConfig,
    Position,
)


# -----------------------------
# Distribution Helpers
# -----------------------------

@dataclass(frozen=True)
class FittedDist:
    """Simple container for an approximate (truncated) Normal for weekly outcomes X_i."""
    mu: float
    sigma: float
    floor: Optional[float]
    ceiling: Optional[float]


def _phi(z: float) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * z * z) / sqrt(2.0 * np.pi)


def _Phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def fit_weekly_distribution(player: Player, default_weekly_sigma: float = 8.0) -> FittedDist:
    """
    Fit an approximate weekly outcome distribution X_i for a player.
    We treat 'proj.mu' as season expectation; for lineup-comparison purposes,
    we operate in 'value units' consistently (seasonal or per-week) since all terms
    are in the same currency. If you prefer per-week, scale mu and sigma beforehand.

    Here, we return a Normal(mu, sigma^2) with:
      - mu := player.proj.mu
      - sigma := player.proj.sigma if provided, else a default
      - floor/ceiling carried through for potential future refinement
    """
    mu = float(player.proj.mu if player.proj.mu is not None else 0.0)
    sigma = float(player.proj.sigma if player.proj.sigma is not None else default_weekly_sigma)
    floor = player.proj.floor
    ceiling = player.proj.ceiling
    # We do not apply hard truncation in the pairwise analytic formula; we keep
    # floor/ceiling for potential Monte Carlo upgrades.
    return FittedDist(mu=mu, sigma=max(1e-6, sigma), floor=floor, ceiling=ceiling)


def prob_X_greater_Y(di: FittedDist, dj: FittedDist) -> float:
    """
    Approximate Pr[X_i > X_j] assuming independence and Normal marginals.
    For independent Normals, X_i - X_j ~ Normal(mu_i - mu_j, sigma_i^2 + sigma_j^2).
    Then Pr[X_i > X_j] = Phi( (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2) ).

    Truncation ignored in closed-form for speed; acceptable in-rank approximations.
    """
    denom = sqrt(di.sigma * di.sigma + dj.sigma * dj.sigma)
    if denom <= 1e-9:
        return 0.5 if abs(di.mu - dj.mu) < 1e-9 else (1.0 if di.mu > dj.mu else 0.0)
    z = (di.mu - dj.mu) / denom
    return float(_Phi(z))


# -----------------------------
# Start-Share Computation
# -----------------------------

def _value_scalar_for_player(player: Player, config: DraftEngineConfig) -> float:
    """
    Return a scalar 'value' for ordering within-roster competitions.
    You may use VOR if available/desired, otherwise mu with risk penalty.
    """
    vm = config.engine.value_model
    mu = float(player.proj.mu or 0.0)
    vor = player.proj.vor
    sigma = float(player.proj.sigma or 0.0)
    base = float(vor) if (vm.use_vor and vor is not None) else mu
    return base - vm.risk_lambda * sigma


def _assign_starters_and_flex(
    roster_players: List[Player],
    config: DraftEngineConfig,
) -> Tuple[Dict[str, float], Dict[Position, List[Player]], List[Player]]:
    """
    Assign dedicated starters and FLEX deterministically by value scalar.
    Returns:
      S_init: uid -> initial start share (1 for starters/FLEX, 0 otherwise)
      by_pos_remaining: remaining bench by position (after removing starters)
      flex_pool_remaining: remaining FLEX-eligible after removing starters and FLEX
    """
    rules = config.league.rules
    # Split by position
    by_pos: Dict[Position, List[Player]] = {p: [] for p in Position}
    for pl in roster_players:
        by_pos[pl.position].append(pl)

    # Sort within each position by value descending
    for p in by_pos:
        by_pos[p].sort(key=lambda x: _value_scalar_for_player(x, config), reverse=True)

    S_init: Dict[str, float] = {}

    # Dedicated starters
    taken_uids: set[str] = set()
    for p in (Position.QB, Position.RB, Position.WR, Position.TE, Position.DST, Position.K):
        need = rules.starter_slots_for(p)
        if need <= 0:
            continue
        plist = by_pos.get(p, [])
        for idx, pl in enumerate(plist):
            if idx < need:
                S_init[pl.uid] = 1.0
                taken_uids.add(pl.uid)

    # FLEX
    flex_need = int(rules.starters.get("FLEX", 0))
    if flex_need > 0:
        flex_elig = []
        for p in rules.flex_positions:
            for pl in by_pos.get(p, []):
                if pl.uid not in taken_uids:
                    flex_elig.append(pl)
        flex_elig.sort(key=lambda x: _value_scalar_for_player(x, config), reverse=True)
        for idx, pl in enumerate(flex_elig):
            if idx < flex_need:
                S_init[pl.uid] = 1.0
                taken_uids.add(pl.uid)

    # Remaining pools
    by_pos_remaining: Dict[Position, List[Player]] = {p: [] for p in Position}
    flex_pool_remaining: List[Player] = []
    for p in Position:
        for pl in by_pos.get(p, []):
            if pl.uid not in taken_uids:
                by_pos_remaining[p].append(pl)
                if p in rules.flex_positions:
                    flex_pool_remaining.append(pl)

    return S_init, by_pos_remaining, flex_pool_remaining


def _bench_start_shares_with_overlap(
    by_pos_remaining: Dict[Position, List[Player]],
    flex_pool_remaining: List[Player],
    config: DraftEngineConfig,
) -> Dict[str, float]:
    """
    Compute bench start shares for remaining players using:
      - position-specific priors by depth rank,
      - optional distribution-overlap adjustment via average pairwise win prob
        within-position and vs FLEX pool.

    Returns uid -> bench start share (additive; starters already fixed at 1.0).
    """
    curves = config.engine.start_share_priors
    shares: Dict[str, float] = {}

    # Pre-fit distributions for speed
    dist_cache: Dict[str, FittedDist] = {}
    def dist_of(pl: Player) -> FittedDist:
        d = dist_cache.get(pl.uid)
        if d is None:
            d = fit_weekly_distribution(pl)
            dist_cache[pl.uid] = d
        return d

    # Within-position depth priors
    for pos, plist in by_pos_remaining.items():
        curve = curves.curve_for(pos)
        for depth_idx, pl in enumerate(plist):
            prior = curve[depth_idx] if depth_idx < len(curve) else curve[-1] if curve else 0.0
            shares[pl.uid] = shares.get(pl.uid, 0.0) + float(prior)

    # Overlap adjustment (lightweight): boost players who often beat their same-position peers
    for pos, plist in by_pos_remaining.items():
        if len(plist) <= 1:
            continue
        for pl in plist:
            di = dist_of(pl)
            # average win prob vs other same-pos bench players
            wins = 0.0
            cnt = 0
            for other in plist:
                if other.uid == pl.uid:
                    continue
                wins += prob_X_greater_Y(di, dist_of(other))
                cnt += 1
            avg_win = (wins / cnt) if cnt > 0 else 0.5
            # small boost scaled around 0.5 baseline
            shares[pl.uid] = shares.get(pl.uid, 0.0) + 0.05 * (avg_win - 0.5)

    # FLEX overlap: if player is FLEX-eligible, small boost proportional to chance
    # of beating others in the FLEX pool (beyond same-position).
    if flex_pool_remaining:
        # Precompute distributions for FLEX pool
        flex_dists = {pl.uid: dist_of(pl) for pl in flex_pool_remaining}
        for pl in flex_pool_remaining:
            di = flex_dists[pl.uid]
            wins = 0.0
            cnt = 0
            for other_uid, dj in flex_dists.items():
                if other_uid == pl.uid:
                    continue
                wins += prob_X_greater_Y(di, dj)
                cnt += 1
            avg_win_flex = (wins / cnt) if cnt > 0 else 0.5
            shares[pl.uid] = shares.get(pl.uid, 0.0) + 0.05 * (avg_win_flex - 0.5)

    # Prevent > 1.0 contributions from bench; cap gently
    for uid in list(shares.keys()):
        shares[uid] = max(0.0, min(0.99, shares[uid]))

    return shares


def compute_start_shares_for_roster(
    roster_players: List[Player],
    config: DraftEngineConfig,
) -> Dict[str, float]:
    """
    Compute S_i(R) for all players on a given team roster (by uid).
    - Deterministic starters and FLEX get S=1.0.
    - Remaining players get bench start shares from priors with small overlap boosts.
    """
    S_init, by_pos_remaining, flex_pool_remaining = _assign_starters_and_flex(roster_players, config)
    bench_shares = _bench_start_shares_with_overlap(by_pos_remaining, flex_pool_remaining, config)

    S: Dict[str, float] = {}
    for pl in roster_players:
        base = S_init.get(pl.uid, 0.0)
        bench = bench_shares.get(pl.uid, 0.0)
        S[pl.uid] = float(max(0.0, min(1.0, base + bench)))
    return S


# -----------------------------
# J(R) and Δ_now(a)
# -----------------------------

def compute_lineup_value_J(
    roster_players: List[Player],
    config: DraftEngineConfig,
) -> float:
    """
    J(R) = E[ sum_i S_i(R) * μ_i ]  (or VOR-adjusted if desired)
    Uses start shares computed from roster composition.
    """
    S = compute_start_shares_for_roster(roster_players, config)
    vm = config.engine.value_model
    total = 0.0
    for pl in roster_players:
        mu = float(pl.proj.mu or 0.0)
        vor = pl.proj.vor
        val = float(vor) if (vm.use_vor and vor is not None) else mu
        total += S[pl.uid] * val
    return float(total)


def delta_now_for_candidate(
    state: DraftState,
    pool: PlayerPool,
    config: DraftEngineConfig,
    team_idx: int,
    candidate_uid: str,
) -> float:
    """
    Δ_now(a) for the specified team: J(R ∪ {a}) - J(R)
    """
    roster = state.rosters[team_idx]
    roster_uids = list(roster.players)
    roster_players = [pool.by_uid[uid] for uid in roster_uids]
    J_before = compute_lineup_value_J(roster_players, config)

    a_player = pool.by_uid[candidate_uid]
    roster_players_after = roster_players + [a_player]
    J_after = compute_lineup_value_J(roster_players_after, config)

    return float(J_after - J_before)


# -----------------------------
# Rank-based marginal functions m_p^{(a)}(r)
# -----------------------------

def _eligible_candidates_by_position_after_a(
    state: DraftState,
    pool: PlayerPool,
    team_idx: int,
    a_uid: str,
) -> Dict[Position, List[Player]]:
    """
    Return remaining available candidates by position after taking 'a'.
    Does NOT enforce per-position caps for the target team; that logic should
    be handled by the calling engine when building feasible sets.
    """
    taken = set(state.drafted_uids)
    taken.add(a_uid)
    by_pos: Dict[Position, List[Player]] = {p: [] for p in Position}
    for uid, pl in pool.by_uid.items():
        if uid in taken:
            continue
        by_pos[pl.position].append(pl)
    return by_pos


def marginal_gain_if_add(
    base_roster_players: List[Player],
    add_player: Player,
    config: DraftEngineConfig,
) -> float:
    """
    Helper: J(R_base ∪ {add}) - J(R_base)
    """
    J_before = compute_lineup_value_J(base_roster_players, config)
    J_after = compute_lineup_value_J(base_roster_players + [add_player], config)
    return float(J_after - J_before)


def top_marginals_by_position_after_pick(
    state: DraftState,
    pool: PlayerPool,
    config: DraftEngineConfig,
    team_idx: int,
    a_uid: str,
    per_pos_k: int = 5,
) -> Tuple[Dict[Position, List[float]], List[Player]]:
    """
    Construct m_p^{(a)}(r) for r=1..per_pos_k by actually evaluating the top
    candidates at each position after taking 'a'.

    Returns:
      - dict: Position -> list of marginal gains [m(r=1), m(2), ...]
      - base_roster_after_a: the roster player list used for these computations
    """
    roster = state.rosters[team_idx]
    roster_players = [pool.by_uid[uid] for uid in roster.players]
    a_player = pool.by_uid[a_uid]
    base_after_a = roster_players + [a_player]

    # Build candidate lists by position (after removing 'a' from pool)
    by_pos = _eligible_candidates_by_position_after_a(state, pool, team_idx, a_uid)

    # For each position, score candidates by marginal gain and keep top per_pos_k
    result: Dict[Position, List[float]] = {}
    for pos, plist in by_pos.items():
        if not plist:
            result[pos] = []
            continue

        # Compute marginal gains
        gains: List[Tuple[float, Player]] = []
        for pl in plist:
            g = marginal_gain_if_add(base_after_a, pl, config)
            gains.append((g, pl))

        gains.sort(key=lambda x: x[0], reverse=True)
        top_vals = [g for (g, _) in gains[:per_pos_k]]
        result[pos] = top_vals

    return result, base_after_a


def interpolate_at_rank(values_desc: List[float], r_float: float) -> float:
    """
    Linear interpolation between neighboring integer ranks for fractional expected removals.
    values_desc: list of m(r) for r=1..K (descending order of value).
    r_float: effective rank (e.g., 1.8) -> interpolate between r=1 and r=2 values, etc.

    If r exceeds available list length, we clamp to the last available value (or 0).
    """
    if not values_desc:
        return 0.0
    if r_float <= 1.0:
        return float(values_desc[0])

    r = float(r_float)
    i0 = int(np.floor(r) - 1)  # zero-based index for floor rank
    i1 = i0 + 1

    if i0 >= len(values_desc):
        return float(values_desc[-1])
    if i1 >= len(values_desc):
        return float(values_desc[-1])

    frac = r - np.floor(r)
    v0 = float(values_desc[i0])
    v1 = float(values_desc[i1])
    return float((1.0 - frac) * v0 + frac * v1)


# -----------------------------
# Convenience: value-now for pool listing
# -----------------------------

def value_now_for_candidates(
    state: DraftState,
    pool: PlayerPool,
    config: DraftEngineConfig,
    team_idx: int,
    candidate_uids: List[str],
) -> Dict[str, float]:
    """
    Compute Δ_now(a) for a batch of candidate uids (used for building
    the top-k list printed at verbosity=2).
    """
    out: Dict[str, float] = {}
    for uid in candidate_uids:
        out[uid] = delta_now_for_candidate(state, pool, config, team_idx, uid)
    return out
