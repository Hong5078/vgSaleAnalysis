# Final Research Implementation

[`final_research.ipynb`](final_research.ipynb) runs the whole project in ten steps, from the parent paper's result
to the final model and an honest check of how accurate it is. The notebook is saved with its outputs, so it can be
read on GitHub without running it.

**Research question.** How well can the number of owners of a Steam game be predicted from information that exists
on its release day, and which information helps the most?

## Steps

| Step | Content | Key result |
|---|---|---|
| 1 | Re-run parent paper 1 (Gray et al., IEEE CICN 2024) | R² = 1.000 comes from regional sales that add up to the target; the row index (sorted by sales) also leaks. Without both: R² 0.10 / −0.10 |
| 2 | Steam data and the release-day rule | 76,727 games, 42 store-page variables, nothing recorded after launch |
| 3 | Game-history variables (idea from Schaer et al., EJOR 2026) | Owners of the series / developer / publisher predecessor; 52% of games have no earlier game |
| 4 | Training, test and validation sets (70/20/10) | Same files as [`../novel_approach/`](../novel_approach/) |
| 5 | Baseline: parent paper 1's method on Steam | R² below zero |
| 6 | One change at a time (Bondarenko et al., IEEE SmartNets 2026; Schaer et al.; my own) | Log target, Gradient Boosting, studio variables, history variables |
| 7 | Comparison and results by kind of game | Final model one year ahead: 0.59 with a series predecessor, 0.10 with no earlier game |
| 8 | Unseen validation set, scored once | Final model R² 0.506 (test 0.495) |
| 9 | Is the model accurate enough to use? | Good at ranking games (top 10% hold ~6× more 100k+ games), but a single forecast is not more accurate than guessing 10,000 owners; new studios' hits are underestimated about 10× |
| 10 | Conclusions, limitations, future work | |

## Results (R² on log owners)

| Model | Random folds | Test | One year ahead | Validation |
|---|---|---|---|---|
| Baseline: Decision Tree, raw owners (parent paper 1) | −0.193 | −0.215 | −0.875 | −0.219 |
| Linear Regression with parent paper 2's changes | 0.313 | 0.311 | 0.089 | 0.314 |
| Gradient Boosting | 0.435 | 0.431 | 0.244 | 0.440 |
| + studio variables | 0.457 | 0.455 | 0.257 | – |
| **+ history variables (final)** | **0.499** | **0.495** | **0.312** | **0.506** |

**Main finding.** Release-day information can rank Steam games by their likely audience, but it cannot forecast
the owners of a single new game accurately. Meaningful forecasts exist only for games with a history to learn from,
mainly sequels in a series.

## Files

| File | Contents |
|---|---|
| `final_research.ipynb` | The notebook, saved with outputs |
| `data/game_info.csv.gz` | Name, first developer, first publisher, release date and owners of the 76,727 games (used to build the history variables in Step 3) |
| `data/train.csv.gz`, `data/test.csv.gz`, `data/validation.csv.gz` | The 70% / 20% / 10% split with all 59 variables, made in [`../novel_approach/`](../novel_approach/) |
| `data/feature_groups.json` | Which variables belong to each group |
| `data/video_game_sales_and_ratings.csv.gz` | Parent paper 1's dataset ([Kaggle](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)) |

The Steam data come from the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
(Kaggle, version 49, MIT license), cleaned in
[`new_analysis/notebooks/01`](../new_analysis/notebooks/01_preprocessing_and_transformation.ipynb). Kaggle no
longer offers version 49, so the needed data are saved here and the notebook runs from a fresh clone.

## How to run

The **first cell of the notebook is a setup cell**. It checks the Python version, installs the exact library
versions listed in [`requirements.txt`](requirements.txt) (numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2,
scipy 1.15.3, jinja2 3.1.6), imports everything, and checks that the `data/` folder is in place. So on any computer
you only need Python 3.10–3.13 and Jupyter, then **Run All**. The full run takes 1–2 minutes.

### On a Mac that has never run this code

1. **Install Python 3.12** from [python.org/downloads](https://www.python.org/downloads/macos/) (the macOS
   installer). The Python that comes with macOS (3.9) is too old for scikit-learn 1.7.
2. **Get the code.** Either click *Code → Download ZIP* on the GitHub page and unzip it, or in Terminal:
   ```
   git clone https://github.com/Hong5078/vgSaleAnalysis.git
   ```
   (macOS may ask to install the command line developer tools the first time `git` is used; click *Install*.)
3. **Create an environment and install Jupyter** (Terminal):
   ```
   cd vgSaleAnalysis/final_research
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install notebook
   ```
4. **Open the notebook** from this folder:
   ```
   jupyter notebook final_research.ipynb
   ```
   Then choose *Run → Run All Cells*. The setup cell installs the libraries into `.venv` the first time.

To install the libraries yourself instead, run `pip install -r requirements.txt` and set `INSTALL_LIBRARIES = False`
in the setup cell. In VS Code, open the `final_research` folder and pick the `.venv` Python as the kernel.

Tested on Windows with Python 3.10 and with a brand-new, empty Python 3.12 environment, where the setup cell installed
every library and all results came out identical.
