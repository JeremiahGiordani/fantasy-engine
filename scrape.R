library(ffanalytics)
library(dplyr)
library(readr)

# Sources
sources <- c("CBS", "ESPN", "FantasyPros", "NumberFire", "FantasySharks", "Yahoo")
# Positions
positions <- c("QB", "RB", "WR", "TE", "K", "DST")

# Scrape season projections (week = 0 means full season)
proj <- scrape_data(src = sources, pos = positions, season = 2025, week = 0)

# Combine into one dataframe
df <- bind_rows(proj)

# Save to CSV in your project folder
write_csv(df, "ffa_season_projections_2025.csv")

print("Saved ffa_season_projections_2025.csv")
