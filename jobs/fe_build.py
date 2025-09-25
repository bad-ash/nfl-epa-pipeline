# jobs/fe_build.py
import wandb, pandas as pd, numpy as np, datetime as dt

SEASON = 2025

def ol_dl_indices(pbp):
    # toy example: season-to-date pressure rates per team-week (no leakage)
    pbp["week"] = pbp["week"].astype(int)
    season = pbp['season'].iloc[0]
    rows = []
    for (team), g in pbp.groupby(["posteam"]):
        for wk in sorted(g["week"].unique()):
            past = g[g["week"] < wk]
            if past.empty: continue
            dropbacks = (past["pass"]==1).sum()
            pressures = ((past["qb_hit"]==1) | (past["sack"]==1)).sum()
            rows.append({
                "team_id": team, "season": season, "week": wk,
                "off_pass_pro_idx": (pressures / dropbacks) if dropbacks else np.nan,
                "asof_ts": dt.datetime(season, 1, 1) + pd.to_timedelta(int(wk)*7, unit="D")
            })
    return pd.DataFrame(rows)

run = wandb.init(project="nfl-pbp", job_type="feature_build")
raw = run.use_artifact(f"raw_pbp-{SEASON}:latest").download()
pbp = pd.read_parquet(f"{raw}/pbp.parquet")

feat_team_week = ol_dl_indices(pbp).dropna()

art = wandb.Artifact(f"feat_offense_team_week-{SEASON}", type="feature_table",
                     metadata={"keys":["team_id","season","week"], "rows": len(feat_team_week)})
feat_team_week.to_parquet("feat_offense_team_week.parquet", index=False)
art.add_file("feat_offense_team_week.parquet")
run.log_artifact(art, aliases=["latest"])
run.finish()
