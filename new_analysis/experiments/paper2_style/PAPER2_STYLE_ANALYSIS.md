# Applying the Second Parent Paper's Approach to the Steam Dataset

Second parent paper: Bondarenko, Cabrera & Fitzek, *Predicting Early Sales of PC Games Using Steam
Wishlists and User Ratings*, IEEE SmartNets 2026 (DOI 10.1109/SMARTNETS69662.2026.11604668).

Three devices are borrowed from that paper, as planned: **separate models for paid and free-to-play
games**, **price bands** instead of one continuous price, and a **log-linear outcome so every effect
is reported as a percentage change**. Its descriptive-table format (n, mean, median per band) and its
coefficient tables (p-values, 95% confidence intervals, multiplicative effect plot) are reproduced.

Reproduce with `python paper2_style_analysis.py`. Data: Steam Games Dataset, pinned to version 49.

## How this differs from the paper

| | Paper | This analysis |
|---|---|---|
| Outcome | 30-day wishlist-to-review conversion | Lifetime estimated owners (SteamSpy range midpoints); Steam wishlists are not in this dataset |
| Main predictor | 30-day user rating | **Not used.** It only exists after release, so it appears here only as a labelled post-release diagnostic |
| Sample | Top 300 paid / 150 free-to-play per month by 30-day reviews, July–November 2025 (1,202 + 343 games) | All 76,727 labeled games released up to June 2024 (68,380 paid, 8,347 free), not selected on the outcome |
| Controls | None needed (5-month window) | Release-year dummies, because the data span 1997–2024 |
| Validation | In-sample R² only | In-sample R² for comparability **and** 5-fold cross-validation using this project's fixed protocol |

Because the outcome differs, the paper's coefficients and these are **not** directly comparable; the
comparison below is about direction and approach, not magnitudes.

## 1. Descriptive tables

**Business model.** Free-to-play games reach a wider audience on average, though the median game in
both groups sits in the lowest owner range.

| Segment | n | Mean owners | Median owners | Share ≥100k owners |
|---|---|---|---|---|
| Paid | 68,380 | 105,837 | 10,000 | 10.0% |
| Free-to-play | 8,347 | 287,642 | 10,000 | 16.6% |

**Price bands (paid games).** Higher price bands are associated with far higher sales, up to $60.

| Price band | n | Mean owners | Median owners | Share ≥100k owners |
|---|---|---|---|---|
| $0.99–9.99 | 51,711 | 49,242 | 10,000 | 6.0% |
| $10–19.99 | 12,489 | 176,082 | 10,000 | 18.1% |
| $20–29.99 | 2,539 | 407,576 | 35,000 | 32.7% |
| $30–39.99 | 831 | 773,724 | 35,000 | 41.2% |
| $40–59.99 | 572 | 1,120,350 | 75,000 | 43.0% |
| $60–79.99 | 53 | 1,800,660 | 10,000 | 35.8% |
| $80+ | 185 | 419,541 | 10,000 | 1.6% |

The $80+ band breaks the pattern. It contains 185 games, mostly niche or unusually priced listings,
and is kept separate rather than merged into the top band.

**This is the opposite sign to the paper's price finding, and both can be true.** The paper finds
that a $10 higher price *reduces* wishlist-to-player conversion by about 15%: among people who
already wishlisted a game, a higher price puts more of them off. Here, a higher price band is
associated with *more* lifetime owners, because price also signals a larger production with a bigger
audience. The two outcomes are different quantities.

## 2. Percentage effects from separate models

OLS on log(owners), fitted separately for paid and free-to-play games, controlling for release year.
Specification B adds store-page features to the paper-style specification. Full table with
coefficients, p-values, classical and robust (HC1) significance and 95% confidence intervals:
`tables/t6_ols_percentage_effects.csv`. Figure: `figures/fig1_effects_paid_vs_free.png`.

**Paid games (n = 68,380; in-sample R² = 0.316).** Effects relative to the $0.99–9.99 band:

| Variable | Effect on expected owners | 95% CI | p |
|---|---|---|---|
| Price $10–19.99 | +38% | +36% to +41% | <0.001 |
| Price $20–29.99 | +142% | +132% to +152% | <0.001 |
| Price $30–39.99 | +212% | +191% to +234% | <0.001 |
| Price $40–59.99 | +257% | +228% to +288% | <0.001 |
| Price $60–79.99 | +137% | +81% to +210% | <0.001 |
| Price $80+ | −49% | −56% to −40% | <0.001 |
| Online co-op | +69% | +62% to +76% | <0.001 |
| Full controller support | +21% | +18% to +23% | <0.001 |
| RPG | +16% | +14% to +19% | <0.001 |
| Online PvP | +15% | +11% to +20% | <0.001 |
| Mac support | +13% | +11% to +16% | <0.001 |
| Each supported language | +1.8% | +1.7% to +2.0% | <0.001 |
| Casual | −12% | −14% to −11% | <0.001 |
| Sports | −13% | −17% to −10% | <0.001 |
| VR-only | −27% | −29% to −24% | <0.001 |

With price as a single continuous variable instead of bands (specification C), each **+$10 is
associated with +9.8% owners** (p < 0.001), against the paper's −15% per $10 on conversion.

**Free-to-play games (n = 8,347; in-sample R² = 0.288).** Price does not apply, as in the paper:

| Variable | Effect on expected owners | 95% CI | p |
|---|---|---|---|
| Massively multiplayer | +34% | +19% to +50% | <0.001 |
| Online co-op | +32% | +19% to +48% | <0.001 |
| Online PvP | +32% | +19% to +47% | <0.001 |
| Mac support | +18% | +9% to +27% | <0.001 |
| Simulation | +15% | +6% to +23% | <0.001 |
| Each supported language | +7.2% | +6.5% to +8.0% | <0.001 |
| Single-player only | −10% | −19% to −1% | 0.026 |
| Indie | −19% | −24% to −14% | <0.001 |
| VR-only | −29% | −36% to −22% | <0.001 |

**What separating the segments reveals.** The same feature carries a different weight in each group:
- **Localisation matters four times more for free-to-play games** (+7.2% vs +1.8% per language).
- **Multiplayer features matter more for free-to-play games**, where massively multiplayer is the
  strongest positive effect; for paid games, online co-op is the strongest.
- **Being an Indie title is a clear negative for free games** (−19%) but nearly neutral for paid ones (−4%).

The paper's in-sample R² values were 0.106 (paid) and 0.164 (free-to-play); ours are higher (0.316
and 0.288), but the outcome, sample and controls all differ, so this is not evidence of a better model.

## 3. Does the paper's approach improve prediction?

The paper never tests predictions on held-out data. Here both devices are evaluated with this
project's protocol (training portion, 5-fold cross-validation, log1p owners). Table:
`tables/t7_pooled_vs_separate_cv.csv`.

| Model | Strategy | Price | R² all games | R² paid | R² free | MAE free |
|---|---|---|---|---|---|---|
| Linear Regression | pooled | continuous | 0.291 | 0.286 | 0.283 | 0.943 |
| Linear Regression | pooled | **bands** | 0.309 | 0.307 | 0.286 | 0.947 |
| Linear Regression | **separate** | continuous | 0.297 | 0.286 | 0.314 | 0.921 |
| Linear Regression | **separate** | **bands** | **0.315** | 0.308 | 0.314 | 0.921 |
| HistGradientBoosting | pooled | continuous | **0.437** | 0.438 | 0.400 | 0.850 |
| HistGradientBoosting | pooled | bands | 0.429 | 0.430 | 0.389 | 0.852 |
| HistGradientBoosting | separate | continuous | 0.433 | 0.438 | 0.376 | 0.862 |
| HistGradientBoosting | separate | bands | 0.428 | 0.431 | 0.376 | 0.862 |

- **For Linear Regression, both devices help.** Price bands add +0.018 R², separate models add +0.005,
  and together +0.023 (0.291 → 0.315). For free-to-play games alone, separate models add +0.031.
- **For gradient boosting, neither helps** (−0.003 for separation, −0.008 for bands). Free-to-play
  games are predicted *worse* by a separate model (0.400 → 0.376), because that model is trained on
  8,347 games instead of all 61,381.
- **Why:** price bands let a linear model represent a non-linear price effect, and separate models
  let every coefficient differ by segment, which is a manual interaction. A gradient-boosted model
  already represents both automatically, and it benefits from pooling the data.

The best model overall is still the pooled HistGradientBoosting from E02 (R² 0.437).

## 4. Rating: post-release diagnostic only

The paper's central variable cannot be used for pre-release prediction, so it is reported only
descriptively, using the paper's rating categories.

| Steam rating category | Paid: n | Paid: mean owners | Paid: share ≥100k | Free: n | Free: mean owners |
|---|---|---|---|---|---|
| Negative (0–39) | 2,139 | 35,636 | 5.8% | 186 | 67,043 |
| Mixed (40–69) | 10,640 | 82,964 | 11.3% | 2,022 | 361,449 |
| Mostly Positive (70–79) | 7,686 | 135,096 | 16.5% | 1,342 | 268,875 |
| Very Positive (80–94) | 16,849 | 207,847 | 19.8% | 2,623 | 461,071 |
| Overwhelmingly Positive (95–100) | 5,282 | 256,180 | 12.8% | 458 | 138,177 |

The paper's monotone rating pattern only partly reproduces here. Mean owners rise with rating for
paid games, but the share reaching 100k owners *falls* in the top category, and the free-to-play
pattern is not monotone at all. A likely reason is that the paper selected the top games by review
count, whereas "Overwhelmingly Positive" across all games includes many small titles with few, very
positive reviews. This is an association measured after release and cannot be used to forecast.

## 5. Conclusions

1. **Separate paid and free-to-play models are worth reporting for interpretation.** The same store-page
   feature has a very different association with sales in each segment, most clearly localisation
   (+7.2% vs +1.8% per language).
2. **Price bands describe the price–sales relationship better than a single slope.** The association
   rises steeply to the $40–59.99 band and reverses above $80.
3. **Neither device improves the best predictive model.** They help Linear Regression (+0.023 R²) but
   slightly hurt gradient boosting, which already captures non-linearity and interactions.
4. **The price result points in the opposite direction to the paper's**, because the outcomes differ:
   conversion of existing wishlisters versus total owners.
5. **All effects are predictive associations, not causal effects.** Price, localisation and multiplayer
   support are chosen together with a game's budget and ambition, none of which is observed here.

## Files

- `tables/t1_business_model.csv`, `t2_price_band_paid.csv`, `t5_genre.csv` — descriptive tables
- `tables/t3_rating_category_paid.csv`, `t4_rating_category_free.csv` — post-release rating diagnostic
- `tables/t6_ols_percentage_effects.csv` — all coefficients, percentage effects, p-values (classical and HC1), 95% CIs
- `tables/t7_pooled_vs_separate_cv.csv` — cross-validated comparison
- `figures/fig1_effects_paid_vs_free.png`, `fig2_price_bands_and_rating.png`, `fig3_pooled_vs_separate.png`
