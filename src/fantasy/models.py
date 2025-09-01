# src/fantasy/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple, Iterable


# -----------------------------
# Core enums & type aliases
# -----------------------------

class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"
    # If your CSV contains IDP or other positions, filter them out upstream.


class Verbosity(IntEnum):
    QUIET = 0
    PICKS = 1
    DEBUG = 2
    TRACE = 3


class DistributionFamily(str, Enum):
    """
    Which parametric family to use for weekly outcome distributions X_i.
    We fit each player's distribution from (mu, floor, ceiling, sigma/uncertainty).
    """
    TRUNC_NORMAL = "trunc_normal"   # truncated normal on [floor, ceiling]
    LOGNORMAL_TRUNC = "lognormal_trunc"  # lognormal, truncated to [floor, ceiling]
    SCALED_BETA = "scaled_beta"     # beta mapped to [floor, ceiling]


# -----------------------------
# Player projections & roster
# -----------------------------

@dataclass(frozen=True)
class Projection:
    """
    Projection container for a single player.
    - mu: projected season points (or per-week mean if you choose to normalize).
    - floor, ceiling: conservative/optimistic seasonal projections (same units as mu).
    - sigma: uncertainty metric compatible with chosen DistributionFamily (e.g., stddev).
    - vor: optional Value Over Replacement provided by source (if present).
    - adp: Average Draft Position (lower = earlier).
    - source: text label for provenance ("ESPN", "FantasyPros", etc).
    - bye_week: optional, if you want to consider bye alignment in advanced logic.
    """
    mu: float
    floor: Optional[float] = None
    ceiling: Optional[float] = None
    sigma: Optional[float] = None
    vor: Optional[float] = None
    adp: Optional[float] = None
    source: Optional[str] = None
    bye_week: Optional[int] = None


@dataclass(frozen=True)
class Player:
    """
    Immutable player record.

    'uid' should be a stable unique id (e.g., "Name|Team|Pos" or a numeric id).
    """
    uid: str
    name: str
    position: Position
    team: Optional[str]
    proj: Projection


# -----------------------------
# League & roster configuration
# -----------------------------

@dataclass(frozen=True)
class RosterRules:
    """
    Roster configuration shared by all teams in the league.

    starters:
        e.g., {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
        Use string keys; engine will map to Position where applicable.

    flex_positions:
        Set of positions eligible for FLEX (e.g., {RB, WR, TE}).

    bench:
        Number of bench slots.

    caps:
        Hard caps per position (e.g., {"QB": 2, "TE": 2, "K": 1, "DST": 1}).
        These prevent nonsense rosters (hoarding low-yield slots).
    """
    starters: Dict[str, int]
    flex_positions: Set[Position]
    bench: int
    caps: Dict[str, int] = field(default_factory=dict)

    def starters_required_non_kdst(self) -> int:
        """Count of non-K/DST starter slots (QB,RB,WR,TE,FLEX)."""
        nk = 0
        for k, v in self.starters.items():
            if k in {"QB", "RB", "WR", "TE", "FLEX"}:
                nk += int(v)
        return nk

    def starter_slots_for(self, pos: Position) -> int:
        return int(self.starters.get(pos.value, 0))


@dataclass(frozen=True)
class LeagueConfig:
    """
    League-wide parameters.
    """
    league_size: int
    rules: RosterRules


# -----------------------------
# Start-share priors (bench curves)
# -----------------------------

@dataclass(frozen=True)
class StartShareCurves:
    """
    Priors for bench start-share as a function of depth rank *within a position*,
    after filling dedicated starters and FLEX. These are *per-team* priors used
    to approximate S_i when we don't run full weekly sims.

    Semantics:
        - For RB/WR: higher bench utilization, e.g., [0.25, 0.12, 0.06, 0.03]
        - For TE/QB: modest bench utilization, e.g., [0.10, 0.04, 0.02]
        - For K/DST: near-zero, e.g., [0.02, 0.0]

    The engine can combine these priors with distribution-overlap logic to
    refine S_i, or use them as fallback when overlaps are inconclusive.
    """
    rb: List[float] = field(default_factory=lambda: [0.25, 0.12, 0.06, 0.03])
    wr: List[float] = field(default_factory=lambda: [0.25, 0.12, 0.06, 0.03])
    te: List[float] = field(default_factory=lambda: [0.10, 0.04, 0.02])
    qb: List[float] = field(default_factory=lambda: [0.10, 0.04, 0.02])
    k:  List[float] = field(default_factory=lambda: [0.02, 0.00])
    dst:List[float] = field(default_factory=lambda: [0.02, 0.00])

    def curve_for(self, pos: Position) -> List[float]:
        return {
            Position.RB: self.rb,
            Position.WR: self.wr,
            Position.TE: self.te,
            Position.QB: self.qb,
            Position.K:  self.k,
            Position.DST: self.dst,
        }[pos]


# -----------------------------
# Engine hyperparameters
# -----------------------------

@dataclass(frozen=True)
class ValueModelParams:
    """
    Controls how we convert projections into comparable 'value' numbers.
    - use_vor: whether to use provided VOR when present.
    - risk_lambda: shrinkage coefficient on uncertainty (mu - lambda * sigma) when needed.
    - distribution_family: how we shape the weekly outcome distributions X_i.
    """
    use_vor: bool = True
    risk_lambda: float = 0.0
    distribution_family: DistributionFamily = DistributionFamily.TRUNC_NORMAL


@dataclass(frozen=True)
class OpponentBehaviorParams:
    """
    Parameters for the opponent position-choice model P_t^{(k)}(p).
    - eta: mixture weight between rational pressure and ADP pressure.
    - tau: softmax temperature (lower -> more deterministic).
    - gap_beta: controls curvature for the need function phi(gap) ~ 1 + beta * gap.
    - adp_sigma: width (in picks) for converting ADP to a pick-time density.
    - kdst_gate: enforce that teams cannot pick K/DST until all non-K/DST starters are filled.
    """
    eta: float = 0.35
    tau: float = 0.08
    gap_beta: float = 0.7
    adp_sigma: float = 0.8  # roughly 0.8 * teams-per-round
    kdst_gate: bool = True


@dataclass(frozen=True)
class EngineParams:
    """
    Top-level hyperparameters for the engine.
    - start_share_priors: priors for bench start shares.
    - value_model: projection-to-value conversion params.
    - opponent_model: opponent behavior model params.
    - verbosity: 0,1,2 controls printing detail.
    - candidate_pool_size: how many top candidates to evaluate deeply per pick.
    - next_step_positions: positions to consider for next-pick expected value (usually all).
    """
    start_share_priors: StartShareCurves = field(default_factory=StartShareCurves)
    value_model: ValueModelParams = field(default_factory=ValueModelParams)
    opponent_model: OpponentBehaviorParams = field(default_factory=OpponentBehaviorParams)
    verbosity: Verbosity = Verbosity.PICKS
    candidate_pool_size: int = 40
    next_step_positions: Set[Position] = field(default_factory=lambda: {
        Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST
    })


# -----------------------------
# Team & draft state
# -----------------------------

@dataclass
class Roster:
    """
    Mutable per-team roster during the draft.
    We store player uids and keep counts per position.
    """
    team_id: int
    name: str
    counts: Dict[Position, int] = field(default_factory=lambda: {
        Position.QB: 0, Position.RB: 0, Position.WR: 0,
        Position.TE: 0, Position.K: 0, Position.DST: 0
    })
    players: List[str] = field(default_factory=list)

    def add(self, pos: Position, uid: str) -> None:
        self.counts[pos] = self.counts.get(pos, 0) + 1
        self.players.append(uid)

    def total(self) -> int:
        return len(self.players)

    def count(self, pos: Position) -> int:
        return int(self.counts.get(pos, 0))


@dataclass
class DraftState:
    """
    Mutable draft state:
      - pick_number starts at 1
      - drafted_uids: set of all taken players
      - rosters: list of per-team rosters (index = team_id)
    """
    league: LeagueConfig
    pick_number: int = 1
    drafted_uids: Set[str] = field(default_factory=set)
    rosters: List[Roster] = field(default_factory=list)

    def __post_init__(self):
        if not self.rosters:
            self.rosters = [Roster(team_id=i, name=f"Team {i}") for i in range(self.league.league_size)]

    def round_index(self) -> int:
        """0-based round index."""
        n = self.league.league_size
        return (self.pick_number - 1) // n

    def index_in_round(self) -> int:
        """0-based index within the current round (left-to-right)."""
        n = self.league.league_size
        return (self.pick_number - 1) % n

    def team_on_the_clock(self) -> int:
        """
        Compute current team index under snake order.
        Even rounds (0,2,4,...) go L->R; odd rounds go R->L.
        """
        n = self.league.league_size
        r = self.round_index()
        i = self.index_in_round()
        return i if (r % 2 == 0) else (n - 1 - i)

    def advance_one_pick(self) -> None:
        self.pick_number += 1

    def is_available(self, uid: str) -> bool:
        return uid not in self.drafted_uids


# -----------------------------
# Board / pool containers
# -----------------------------

@dataclass
class PlayerPool:
    """
    The available player universe (immutable metadata).
    'by_uid' lets you resolve details quickly.
    'uids_by_pos' accelerates position filtering.
    """
    by_uid: Dict[str, Player]
    uids_by_pos: Dict[Position, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.uids_by_pos:
            up: Dict[Position, List[str]] = {p: [] for p in Position}
            for uid, p in self.by_uid.items():
                up[p.position].append(uid)
            self.uids_by_pos = {k: v for k, v in up.items()}

    def iter_available(self, state: DraftState) -> Iterable[Player]:
        for uid, p in self.by_uid.items():
            if uid not in state.drafted_uids:
                yield p

    def available_by_position(self, state: DraftState, pos: Position) -> List[Player]:
        out: List[Player] = []
        for uid in self.uids_by_pos.get(pos, []):
            if uid not in state.drafted_uids:
                out.append(self.by_uid[uid])
        return out


# -----------------------------
# Verbose/debugging payloads
# -----------------------------

@dataclass
class CandidateRow:
    """
    A row for reporting/printing top candidates at a pick.
    All numbers are expected to be 'in lineup value' units (same currency as J).
    """
    uid: str
    name: str
    team: Optional[str]
    position: Position
    value_now: float                       # Δ_now(a)
    exp_next: float                        # Δ_next^(k)(a)
    utility: float                         # value_now + exp_next
    adp: Optional[float] = None
    mu: Optional[float] = None
    vor: Optional[float] = None


@dataclass
class NextTurnBreakdown:
    """
    For verbose=2: show expected next-turn contributions per position and ranks.
    - exp_by_pos: expected marginal lineup gain by position (tilde m_p^(a)(r_p)).
    - rank_by_pos: effective 'r_p^(a)' used for interpolation per position.
    """
    exp_by_pos: Dict[Position, float]
    rank_by_pos: Dict[Position, float]


# -----------------------------
# Public engine config wrapper
# -----------------------------

@dataclass(frozen=False)
class DraftEngineConfig:
    """
    Aggregates everything needed to instantiate a DraftEngine.

    - league: league size & roster rules
    - engine: hyperparameters for value modeling, start-share priors, opponent behavior
    - verbosity: printing level (0/1/2)
    """
    league: LeagueConfig
    engine: EngineParams = field(default_factory=EngineParams)
    verbosity: Verbosity = Verbosity.PICKS

    def cap_for(self, pos: Position) -> Optional[int]:
        """Convenience: per-position cap if specified."""
        c = self.league.rules.caps.get(pos.value, None)
        return int(c) if c is not None else None
