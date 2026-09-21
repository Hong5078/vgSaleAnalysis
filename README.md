# vgSaleAnalysis — Video Game Sales Prediction (DS 340W)

Parent paper: Gray, D.; Shalan, A. M.; Kadlec, C. *Sales Predictions for Video Games Using
Predictive Analytics of Market Data.* IEEE CICN 2024. DOI: 10.1109/CICN63059.2024.10847429

## Repository layout

| Folder | Status | Contents |
|---|---|---|
| [`old_analysis/`](old_analysis/) | Frozen | Weeks 1–3 work on the Kaggle VGChartz datasets: baseline code, parent-paper replication, leakage-corrected replication, and the pre-release hit classifier |
| [`new_analysis/`](new_analysis/) | Active | Pre-release sales prediction on the Steam Games Dataset: Linear Regression, Decision Tree and Random Forest with 5-fold cross-validation and a chronological hold-out. Three notebooks following the pipeline Preprocessing → Transformation → Analysis → Modeling → Validation → Decision making (see [`new_analysis/README.md`](new_analysis/README.md)) |
| [`DATASET_SWITCH_PROPOSAL.md`](DATASET_SWITCH_PROPOSAL.md) | — | Short report: what the old analysis found and why the project is moving to a new dataset |

All code in `old_analysis/` still runs from its own folder — every data path is relative and
the folder structure inside it is unchanged.
