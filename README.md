# Residential prices: own Decision Trees and Pegasos SVM

AI Academy / National AI Center, MLE-AI-201 Final Project V3.
Start with `START_HERE_AZ.md` and `FINAL_Project.ipynb`.

This final package implements the required models in NumPy, runs validation
selection, separate grouped CV and a held-out test, and includes the report,
slides, trained own models and executed notebook. Human names, actual
contributions and the GitHub URL still require the team's information.

## Final test results

All 8,724 residential test rows are retained. Positive class: price > 207,000 AZN.

| Own model | Task | Main test results |
|---|---|---|
| Spatial, routed decision trees | Price regression | RMSE 667,506.79 AZN; MAE 56,567.58 AZN; R² 0.1037 |
| Entropy tree | Premium classification | F1 0.9218; ROC AUC 0.9670 |
| Linear Pegasos | Premium classification | F1 0.9037; ROC AUC 0.9671 |
| RFF + own Pegasos | Premium classification | F1 0.9088; ROC AUC 0.9716 |

The price model's median absolute percentage error is **8.74%**; **77.38%** of
predictions are within 20% of the asking price. R² remains weak because large
errors dominate squared loss. The highest 1% of squared errors contribute
98.84% of the total. These statistics describe different aspects of quality;
F1 must not be described as price-prediction accuracy.

## Run

Use Python 3.12 (verified: 3.12.14). The ZIP includes the supplied
`data/house_sale.csv`; Git intentionally excludes raw data. A Git clone needs
the same course CSV placed in that folder. The data fingerprint is documented
in `data/README.md`.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_all
```

The last command rebuilds all final searches, CV, models, metrics, figures and
report tables from the CSV. The original execution took about 9 minutes over
the three phases on the execution host; your computer can differ. The largest
RFF training matrix alone is about 671 MB; allow several GB of working memory.
Seeds are fixed at 42 and BLAS is limited to two threads.

Open `FINAL_Project.ipynb` in your existing Jupyter installation. Its default
mode reads the supplied results and trained models. It does not repeat the
long experiment unless `RUN_FULL_EXPERIMENT=True` is selected.

If a LaTeX installation provides `pdflatex`, rebuild the PDF too:

```bash
python -m src.run_all --compile-report
```

The finished `report/report.pdf` is already included. Report-only rebuild:
`python -m src.build_report --compile`. Editable slides are supplied as PPTX;
their PDF is also included. Presentation source data are in
`presentation/deck_data.json`; format editing is possible in PowerPoint or
LibreOffice. The optional JS authoring source requires the Artifact Tool runtime.

For a different CSV path:
`python -m src.run_all --data /absolute/path/house_sale.csv`.
Separate stages are `--stage search`, `--stage cv` and `--stage test`.
Search freezes configurations before test; do not retune them after seeing test
results. Repeating the exact experiment is for reproducibility, not test tuning.

## Inference demo

```bash
python -m src.predict examples/listing.json
```

Or call `predict_listing` from `src.predict` with cleaned, physically meaningful
features. Category must be one of `yeni tikili`, `köhnə tikili`,
`həyət evi/bağ evi`. The JSON example is illustrative, not a claimed actual sale.
The price and tier models are independent, so their outputs need not always
agree perfectly near the premium threshold.

## Experimental protocol

- 100,775 raw rows; 64,454 latest listings; 58,452 residential rows.
- Original group memberships remain fixed after residential filtering.
- Train / validation / test: 40,932 / 8,796 / 8,724.
- Premium threshold: residential **training** median, 207,000 AZN (strict >).
- Final classifier input: 141 train-fitted encoded columns.
- Raw text, IDs, owner/shop/address and target-derived fields never enter X.
- Imputation, category vocabularies, scaling and spatial centres fit only on
  training data, separately inside every fold.
- Seven own regression candidates; twelve own classifier-tree candidates;
  eighteen SVM search/diagnostic trials, following short pilot runs.
- Lowest validation RMSE selects regression; validation positive-class F1
  selects classification. Baselines do not replace required own algorithms.
- Final fits use original training only. Five-fold GroupKFold is a separate
  fixed-configuration stability check, not nested CV or a test confidence interval.
- No evaluation rows are removed according to target value or prediction error.

The 600 million AZN training price causes an extreme CV RMSE fold. The maximum
test price is 60 million AZN. Both remain unverified. These historical advertised
prices do not establish current transaction-value accuracy or future performance.

## File map

| Path | Purpose |
|---|---|
| `src/decision_tree.py` | Own Gini/entropy/MSE tree, exact threshold search |
| `src/svm.py` | Own mini-batch Pegasos and RFF; NumPy only |
| `src/data_prep.py` | Cleaning, group split, encoding and scaling |
| `src/residential_models.py` | Train-only geography and category routing |
| `src/final_models.py` | Own classifier pipelines and matched sklearn baselines |
| `src/final_experiment.py` | Search, freeze, CV, fixed test and timings |
| `src/final_figures.py`, `src/final_eda.py` | Reproducible figures and error analysis |
| `outputs/frozen_config.json` | Validation-selected settings and fingerprints |
| `outputs/final_metrics.csv` | All own/baseline validation and test metrics |
| `outputs/final_cv_folds.csv` | Every CV fold; no hidden outlier fold |
| `outputs/svm_margin_violators.csv` | Training positions with signed margin ≤ 1 |
| `models/final_models.pkl` | Saved own models and fitted preprocessors |
| `report/report.tex`, `report/report.pdf` | IEEE-style source and report |
| `presentation/ML_Final_Slides.pptx`, `.pdf` | Editable deck and submission PDF |
| `DEFENSE_GUIDE_AZ.md` | Azerbaijani explanations and demonstration plan |
| `development_history/` | Earlier exploratory work, not final headline scores |

## Submission metadata

The technical work is complete; do not invent author names or contribution
percentages. Fill `submission_metadata.json` and `contribution_report.md` with
real team information, add the actual GitHub URL, then rebuild the report and
add the names to the slide cover. The course asks for the repository URL tagged
`v1.0-final`, the report PDF and the slides PDF. No external repository has been
published by this package.

To publish after creating/choosing the team's repository, use your own Git
identity, commit the reviewed project, create the required tag, add its remote
and push. `.gitignore` excludes the raw CSV, arrays and caches. Do not upload
the raw CSV through GitHub's browser file uploader.

See `AI_DISCLOSURE.md`: substantial ChatGPT/Codex assistance is explicitly disclosed.

## Verification record

All 41 unit tests passed. All 12 final notebook code cells were executed in order
in Python with their outputs and figures captured; launching the Jupyter UI/kernel
on the execution host was not tested. A fresh-output run reproduced search and CV;
after a session interruption, its final stage resumed with unchanged settings.
Five numerical result tables and 36 prediction arrays match the original run
within 1e-10, excluding timings and freeze timestamps. The details are in
`outputs/reproducibility_verification.json`.

For the remaining repository submission steps, see `PUBLISH_GITHUB_AZ.md`.
