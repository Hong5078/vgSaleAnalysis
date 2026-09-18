# Old analysis (Weeks 1–3) — VGChartz datasets

Frozen snapshot of all work done on the Kaggle VGChartz-based datasets. Kept for
reproducibility and as the justification for the dataset switch
(see [`../DATASET_SWITCH_PROPOSAL.md`](../DATASET_SWITCH_PROPOSAL.md)).

| Path | What it is |
|---|---|
| `vgsales.py`, `vgsales.ipynb`, `vgsales.csv` | Week 1 baseline code found on GitHub (sainathdevulapalli/Video-Game-Sales, CC0) |
| `paper_replication/` | Re-implementation of the parent paper (LR + Decision Tree on `video_game_sales_and_ratings.csv`) |
| `regional_trend_analysis/` | Week 2 Direction 1 — regional sales from Genre + Platform |
| `score_prediction_analysis/` | Week 2 Direction 2 — predicting Critic/User score |
| `week3/01–03` | Phase 1 — leakage-corrected replication (regional sales removed) |
| `week3/04–05` | Phase 2 — pre-release hit classification (≥1M units) |
| `week3/outputs/` | Saved metrics tables and figures |
| `NOTES.md`, `WEEK2_DIRECTIONS_CHECKLIST.md`, `README_original.md` | Working notes |

Run notebooks from their own folder (e.g. `cd week3`) — paths are relative.
