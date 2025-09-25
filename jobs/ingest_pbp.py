# jobs/ingest_pbp.py
import wandb, nflreadpy as nfl, pandas as pd

SEASON = 2025

run = wandb.init(project="nfl-pbp", job_type="ingest")

# Pull current season pbp
df = nfl.load_pbp().to_pandas()
df = df.drop_duplicates(subset=["game_id","play_id"])

# Latest game date in this pull
latest_date = pd.to_datetime(df["game_date"]).max()

# Try to fetch last artifact to compare
try:
    prev = run.use_artifact(f"raw_pbp-{SEASON}:latest")
    prev_meta = prev.metadata
    prev_date = pd.to_datetime(prev_meta.get("latest_date"))
except Exception:
    prev_date = None

# If no new games, skip
if prev_date and latest_date <= prev_date:
    print(f"No new games since {prev_date.date()}, skipping log.")
else:
    art = wandb.Artifact(f"raw_pbp-{SEASON}", type="dataset",
                         metadata={"season": SEASON, "rows": len(df),
                                   "latest_date": str(latest_date.date())})
    df.to_parquet("pbp.parquet", index=False)
    art.add_file("pbp.parquet")
    run.log_artifact(art, aliases=["latest"])
    print(f"Logged new artifact for data through {latest_date.date()}")

run.finish()
