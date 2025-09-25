# jobs/fe_fetch.py
import wandb, pandas as pd

def fetch_offense_team_week(season, alias="latest"):
    run = wandb.init(project="nfl-pbp", job_type="feature_fetch")
    path = run.use_artifact(f"feat_offense_team_week:{season}:{alias}").download()
    df = pd.read_parquet(f"{path}/feat_offense_team_week.parquet")
    run.finish()
    return df
