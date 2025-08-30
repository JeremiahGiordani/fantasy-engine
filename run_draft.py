import pandas as pd
from fantasy.draft_engine import DraftConfig, DraftState, DraftAssistant

# --------------------------
# Load projections
# --------------------------
CSV_PATH = "projections_2025_wk0.csv"  # put this next to the script or adjust path
df = pd.read_csv(CSV_PATH)

# --------------------------
# League setup (6 teams, 3 rounds)
# --------------------------
LEAGUE_SIZE = 10
ROUNDS = 16

cfg = DraftConfig(
    league_size=LEAGUE_SIZE,
    starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1},
    bench=7,
    risk_lambda=0.0,
)

state = DraftState(cfg=cfg)

# Create 6 assistants (one per team) using the same projections/config
assistants = [DraftAssistant(cfg, df) for _ in range(LEAGUE_SIZE)]

# --------------------------
# Run a 3-round snake draft
# Each team simply picks its top recommendation when on the clock.
# --------------------------
total_picks = LEAGUE_SIZE * ROUNDS
pick_log = []

for _ in range(total_picks):
    team_idx = state.team_on_the_clock()
    brain = assistants[team_idx]

    recs = brain.recommend_for_team(state, team_idx, top_k=1)
    if recs.empty:
        raise RuntimeError("No candidates available — ran out of players?")

    row = recs.iloc[0]
    brain.apply_pick(state, row["id"])

    pick_log.append({
        "Pick": len(pick_log) + 1,
        "Team": team_idx,
        "Player": row["player"],
        "Pos": row["position"],
        "TeamNFL": row["team"],
        "ADP": row.get("adp", None),
        "ValueNow": round(float(row["value"]), 2),
        "ExpNext": round(float(row["exp_next"]), 2),
        "Utility": round(float(row["utility"]), 2),
    })

# --------------------------
# Print results
# --------------------------
print("\n=== Draft Picks (6 teams, 3 rounds) ===")
for p in pick_log:
    print(f"{p['Pick']:>2}. Team {p['Team']} -> {p['Player']} ({p['Pos']}, {p['TeamNFL']})  "
          f"[value={p['ValueNow']}, expNext={p['ExpNext']}, util={p['Utility']}, ADP={p['ADP']}]")

print("\n=== Roster tallies by team ===")
for t, r in enumerate(state.rosters):
    print(f"Team {t}: {r.counts}")
