# Interpretation and Error Analysis of the Best Model

**Model analysed (unchanged):** `HistGradientBoostingRegressor(random_state=42)`, defaults; target
log1p(owners); the 42 release-time features; harness split (80/20, seed 42) and 5-fold KFold (seed 42).
Reproduce with `python interpret_best_model.py`, which confirms CV R²_log 0.4367 and test R²_log 0.4255
before the analysis.

All findings are **predictive associations**, not causal effects. Residual = predicted − actual on the
natural-log scale (positive = over-prediction); a residual of ±1.0 is a factor of about 2.7 in owners.

## 1. Which features contribute most?

Permutation importance = drop in R² (log1p owners) when a feature is shuffled. It is measured on the
five cross-validation validation folds (5 folds × 5 repeats); the test set was used only to check the
ranking, which agrees closely (Spearman ρ = 0.97).

| Rank | Feature | Importance (mean ± SD) | Rank | Feature | Importance (mean ± SD) |
|---|---|---|---|---|---|
| 1 | release_year | 0.304 ± 0.010 | 9 | n_audio_languages | 0.009 ± 0.002 |
| 2 | n_languages | 0.127 ± 0.007 | 10 | required_age | 0.008 ± 0.003 |
| 3 | price | 0.080 ± 0.005 | 11 | n_screenshots | 0.007 ± 0.002 |
| 4 | genre_Free_To_Play | 0.036 ± 0.002 | 12 | genre_RPG | 0.006 ± 0.001 |
| 5 | cat_Steam_Cloud | 0.021 ± 0.002 | 13 | genre_Indie | 0.006 ± 0.002 |
| 6 | genre_Casual | 0.012 ± 0.002 | 14 | cat_VR_Only | 0.005 ± 0.001 |
| 7 | description_length | 0.010 ± 0.002 | 15 | mac | 0.004 ± 0.001 |
| 8 | cat_Online_Co_op | 0.010 ± 0.002 | | | |

**Feature groups shuffled jointly:**

| Group | Importance |
|---|---|
| Release year | 0.307 |
| Localisation | 0.155 |
| Price / business model | 0.127 |
| Game genres | 0.033 |
| Play mode | 0.023 |
| Steam platform features | 0.022 |
| Store-page presentation | 0.019 |
| Controller & VR support | 0.014 |
| Age rating & content descriptors | 0.008 |
| Mac/Linux support | 0.006 |

Reviews/ratings and developer/publisher are **not model inputs**: reviews are post-release and were
excluded to avoid leakage, and developer/publisher were never encoded.

**Reading:**
- **Release year** is the strongest predictor, but mainly as a proxy for time on sale and market
  crowding. Older games have had longer to accumulate owners and faced fewer competitors.
- **Within a single release year** the model still explains 24–38% of the variance in log owners (0.32
  on average, `within_year_r2_test.csv`). So localisation, price/business model, genre and play-mode
  features carry substantial signal among games released at the same time.
- Importances are not additive, because correlated features share credit.

## 2. Which games are easiest to predict?

- **Games in the lowest owner range** (<20k owners, 70% of games): MAE_log 0.44.
- **Recent releases** (2023–2024: MAE_log 0.48, against about 1.0 for releases up to 2016).
- **Low-priced paid games** ($0.01–5: 0.52).
- **Niche single-player categories**: VR-only (0.43), tracked-controller (0.47), casual (0.54).

These groups are "easy" largely because their outcomes cluster near the floor of the owner distribution.

## 3. Which games are hardest to predict?

- **Successful games.** MAE_log rises steadily with actual popularity: 0.84 for 20k–100k, 1.67 for
  100k–1M, 2.11 for 1M–10M, 3.37 for 10M+.
- **Online multiplayer / MMO titles:** MAE_log 0.95, against 0.61 for single-player-only games.
- **Free-to-play games:** 0.94, against 0.63 for paid games.
- **Games priced $20–40:** 1.07.
- **Releases up to 2016:** about 1.0.
- **Hardest categories:** online co-op (1.05), massively multiplayer (1.02), level-editor support (0.97).

Online and local multiplayer games cause 98% of the test set's original-scale squared error;
single-player-only games cause 2%.

**Largest individual errors (test set):**
- **Under-predicted hits with a small-game profile:** Lethal Company (10–20M owners, predicted 14k:
  one language, $9.99), Undertale, Buckshot Roulette, Sinking Simulator.
- **Franchise and free-to-play titles:** The Sims 4, Palworld, Brawlhalla, Counter-Strike 2.
- **Label problems** (see Section 4).

## 4. Does the model systematically under-predict blockbusters?

**Yes — conditional on actual popularity.** Mean log bias by actual owner tier:

| Owner tier | <20k | 20k–100k | 100k–1M | 1M–10M | 10M+ |
|---|---|---|---|---|---|
| Mean log bias | +0.44 | −0.63 | −1.53 | −2.06 | −3.37 |
| In owners | ×1.5 | ÷1.9 | ÷4.6 | ÷7.8 | ÷29 |

- **The pattern replicates** on the out-of-fold predictions for 61k training games; each tier's bias is
  within 0.14 of the test value.
- **Conditional on the prediction, residuals average about zero** in every prediction bin (Figure c).
  This is regression toward the mean from incomplete information, not a miscalibrated model.
- **All 20 test games with 10M+ owners are under-predicted.** They account for 96.6% of the test set's
  original-scale squared error.

**Do blockbusters share observable characteristics?** Yes. Across all 86 games with 10M+ owners
(out-of-fold and test predictions), compared with games under 20k owners:

| Characteristic | 10M+ games | Games under 20k |
|---|---|---|
| Free-to-play | 30% | 4% |
| Massively multiplayer | 22% | 1.5% |
| Online multiplayer | 74% | 8% |
| Priced $20+ | 36% | 4% |
| Median supported languages | 13.5 | 1 |
| Full controller support | 47% | 20% |

The model uses these characteristics: it ranks blockbusters near the top (ROC-AUC 0.979; median
prediction percentile 99.3%). Exceptions such as Lethal Company have an ordinary small-game profile.

**Why the current features appear insufficient for magnitude.** Many far smaller games share the same
profile:

| Games selected (highest predicted) | Share with 10M+ owners | Share below 1M | Median actual owners |
|---|---|---|---|
| Top 0.1% (77 games) | 23% | 25% | 3.5M |
| Top 1% (768 games) | 7% | 43% | 1.5M |

The features locate a high-potential group but cannot separate blockbusters from similar-looking games
that sell 10–100× less. The analysis does not show that blockbuster success is unpredictable in
principle; it shows that the current 42 features do not contain what distinguishes them.

## 5. What information may be missing?

Evidence from the diagnostic-only variables and the largest errors:

- **Post-launch reception (unavailable before release, by design).** Games rated ≥80% positive are
  under-predicted (bias −0.18 to −0.21). Games rated <70% are over-predicted (+0.12 to +0.13). Games
  with 10k+ reviews are under-predicted by ÷6.7. Part of the unexplained variance is quality and word
  of mouth revealed after launch.
- **Studio and publisher standing.** Games from external publishers are slightly under-predicted
  (−0.08), and self-published games over-predicted (+0.05). Publishers with 20+ prior releases −0.06,
  first releases +0.06. The effects are modest but systematic.
- **Franchise / IP strength.** Many of the largest under-predictions are established franchises
  (The Sims 4, Call of Duty, Hogwarts Legacy, Ultimate Custom Night / Five Nights at Freddy's).
- **Pre-release demand** (wishlists, followers, media attention) and **marketing budget**, which are not
  in the dataset.
- **Label noise.**
  - 3 test games (0.02%) have more reviews than the upper bound of their owner range, which is
    impossible because a review requires ownership: Rainbow Six Siege X listed twice, and Portal 2.
    They produce 3 of the 20 largest over-predictions.
  - Excluding them changes test R²_log only from 0.426 to 0.430.
  - Softer anomalies exist, such as duplicate *Shadow of the Tomb Raider: Definitive Edition*
    listings each labeled 0–20k owners with about 15k reviews.

## 6. Most justified feature-engineering ideas

1. **Developer/publisher track record, computed only from earlier releases:** number of prior
   releases, and prior hits known at the time.
   - The diagnostic shows systematic bias by publisher type and experience.
   - Blockbusters come from publishers with more prior releases (median 3 vs 0 for games under 20k).
2. **Franchise indicators:** sequel markers and title stems shared with earlier releases. These target
   the franchise titles among the largest under-predictions.
3. **Pre-release attention proxies:** store-page lead time estimated from AppID order, or external hype
   data. This is aimed at the demand information the model visibly lacks.

   In the earlier engineered draft (`notebooks/drafts/v2_engineered/`), lead time was among the most
   informative indicators.

**Also worth doing:**
- **A label-consistency screen**, fixed in advance and applied to training labels only: flag games
  whose review count exceeds their owner range.
- **Validate each idea with the standard cross-validation protocol, not the test set.** Several patterns
  above were observed on test predictions. They replicate out-of-fold, but test-set reuse for feature
  selection must be avoided.

## Files

- **Tables** (`tables/`):
  - Importance: `permutation_importance_features.csv`, `permutation_importance_groups.csv`
  - Test predictions: `test_predictions.csv.gz` (per game)
  - Largest errors: `test_largest_*.csv` (4 lists)
  - Error patterns: `error_by_group_test.csv`, `pattern_replication_test_vs_oof.csv`, `within_year_r2_test.csv`
  - Blockbusters: `blockbuster_profile_by_tier.csv`, `blockbuster_ranking.csv`, `blockbusters_all_86.csv`
  - Labels: `label_consistency_sensitivity.csv`, `test_inconsistent_labels.csv`
- **Figures** (`figures/`):
  - a: permutation importance
  - b: predicted vs actual
  - c: residual vs predicted
  - d: error by owner range
  - e: error by year and game type
  - f: blockbusters
