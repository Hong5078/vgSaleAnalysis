"""Google Trends pilot: can pre-release search interest be measured for mid-range Steam games?

30 games released 2019 to June 2024, five from each owner range between 20,000 and 2 million. For each game:
1. look up a Google Knowledge Graph entity of type video game (as in Schaer, Kourentzes & Fildes, EJOR 2026);
2. request weekly search interest (worldwide) from 52 weeks before to 12 weeks after release, for the entity if
   one exists, otherwise for the title within the Games category;
3. request the same with the paper's neutral reference keyword "marker", to see whether the game's series
   survives the shared scaling that makes games comparable.
Raw series and a per-game summary are written to data/google_trends_pilot/ (not tracked in git).
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "google_trends_pilot"))
os.makedirs(OUT_DIR, exist_ok=True)
ANCHOR = "marker"
PAUSE = 20  # seconds between requests; Google Trends refuses rapid requests


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sample_games():
    data = pd.read_csv(os.path.join(DATA_DIR, "steam_model_table.csv.gz"), parse_dates=["release_dt"])
    data["owners"] = (data["owners_millions"] * 1e6).round()
    ok_title = data["name"].fillna("").map(lambda s: s.isascii() and len(s.split()) >= 2 and len(s) <= 40)
    pool = data[(data["release_dt"] >= "2019-01-01") & ok_title
                & data["owners"].isin([35_000, 75_000, 150_000, 350_000, 750_000, 1_500_000])]
    return (pool.groupby("owners", group_keys=False).apply(lambda g: g.sample(5, random_state=42))
            [["appid", "name", "release_dt", "estimated_owners", "owners"]].reset_index(drop=True))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def call(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            result = fn(*args, **kwargs)
            time.sleep(PAUSE)
            return result
        except Exception as e:  # 429 Too Many Requests and timeouts
            wait = 60 * 2 ** attempt
            log(f"  request failed ({type(e).__name__}: {str(e)[:80]}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("gave up after repeated failures")


def interest(trends, keywords, timeframe, cat):
    call(trends.build_payload, keywords, cat=cat, timeframe=timeframe, geo="")
    df = call(trends.interest_over_time)
    return df.drop(columns="isPartial", errors="ignore")


def main():
    games = sample_games()
    games.to_csv(os.path.join(OUT_DIR, "pilot_games.csv"), index=False)
    trends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    done = set()
    summary_path = os.path.join(OUT_DIR, "pilot_summary.csv")
    if os.path.exists(summary_path):
        done = set(pd.read_csv(summary_path)["appid"])
    for i, g in games.iterrows():
        if g["appid"] in done:
            continue
        log(f"[{i + 1}/{len(games)}] {g['name']} ({g['estimated_owners']}, released {g['release_dt'].date()})")
        suggestions = call(trends.suggestions, g["name"])
        entity = next((s for s in suggestions if "game" in s.get("type", "").lower()
                       and norm(s["title"]) == norm(g["name"])), None)
        keyword, cat = (entity["mid"], 0) if entity else (g["name"], 8)  # 8 = Games category
        release = g["release_dt"]
        timeframe = f"{(release - pd.Timedelta(weeks=52)).date()} {(release + pd.Timedelta(weeks=12)).date()}"
        alone = interest(trends, [keyword], timeframe, cat)
        paired = interest(trends, [keyword, ANCHOR], timeframe, cat)
        weeks = ((alone.index - release).days // 7) if len(alone) else np.array([])
        series = alone[keyword] if len(alone) else pd.Series(dtype=float)
        pre = series[(weeks >= -26) & (weeks <= -1)] if len(alone) else series
        post = series[(weeks >= 0) & (weeks <= 4)] if len(alone) else series
        paired_pre = paired[keyword][(weeks >= -26) & (weeks <= -1)] if len(paired) == len(alone) and len(paired) else pd.Series(dtype=float)
        for name, frame in [("alone", alone), ("paired", paired)]:
            frame.assign(week_from_release=weeks if len(frame) == len(weeks) else np.nan) \
                .to_csv(os.path.join(OUT_DIR, f"{g['appid']}_{name}.csv"))
        row = {"appid": g["appid"], "name": g["name"], "release_date": release.date(), "owners": g["estimated_owners"],
               "entity_found": entity is not None, "entity_type": entity["type"] if entity else "",
               "keyword": keyword, "weeks_returned": len(alone),
               "pre_release_weeks_nonzero": int((pre > 0).sum()), "pre_release_sum": float(pre.sum()),
               "first_4_weeks_after_max": float(post.max()) if len(post) else np.nan,
               "peak_week_from_release": int(weeks[int(np.argmax(series.to_numpy()))]) if len(series) and series.max() > 0 else np.nan,
               "with_anchor_pre_release_weeks_nonzero": int((paired_pre > 0).sum()),
               "with_anchor_max": float(paired[keyword].max()) if len(paired) else np.nan}
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)
        log(f"  entity={row['entity_found']} nonzero pre-release weeks={row['pre_release_weeks_nonzero']}/26 "
            f"with anchor={row['with_anchor_pre_release_weeks_nonzero']}/26")
    log("finished")


if __name__ == "__main__":
    sys.exit(main())
