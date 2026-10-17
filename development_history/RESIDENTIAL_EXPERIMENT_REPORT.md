# Residential regression experiments — Stage 2.1



## Scope and protocol

Training: 40,932; validation: 8,796. Only new/old apartments and detached/garden houses are included. Offices, commercial properties, land and garages are excluded from this explicitly labelled subtask.

Original split membership and property groups are preserved. The original full-dataset task and its earlier results remain available. No held-out test predictions were computed. No validation rows were removed based on price, prediction error or audit flags.

All reported validation model comparisons use the same residential rows. A reduction from the former full-dataset RMSE is not itself evidence of improved training. The full-scope model is re-evaluated on the same residential rows as a control.

## Training audit and source review

The training-only category-specific log(price/area) rule flags 90 observations outside Q1 − 3 IQR / Q3 + 3 IQR. A separate area-per-room review flag identifies 10 rows. These are review flags, not confirmed errors.

A 200 m² apartment has a recorded price of 600,000,000 AZN, and other records show 73, 400 and 700 AZN. Unknown true prices are not invented or replaced. The 600-million record has no usable description to resolve its true price.

Review of training descriptions identifies the 400 and 700 AZN records as rentals. Two house descriptions explicitly state floor areas of 650 m² and 300 m² where the table says 9 m² and 11 m²; the latter also states 11 sot of land instead of the table’s 310 sot. The source-review experiment excludes the two rentals from training and uses those described areas. It is source-supported curation, not independently verified ground truth.

residential_source_review.json records hashed listing keys, original row references, actions and short source excerpts. Lookup IDs are never predictors. Corrections operate on a copy of training data only; derived area features are recomputed. Raw CSV and all evaluation rows remain unchanged.

Sensitivity models for broad audit filtering and source-reviewed training are presented separately and are not used to select the main models. The remaining ambiguous 14.5 m² record, price 73 and price 1,600 are left for review.

## Spatial features and separate models

Extra features are great-circle distances in kilometres to a UNESCO Walled City reference point and to 12 geographic centres estimated solely from training coordinates. NumPy k-means uses a seeded training sample of at most 10,000 rows. The original latitude/longitude and categorical location features remain available.

These distances are not walking/driving distances. Learned centres are not metro stations or official districts. Verified station coordinates were not supplied, so a metro-distance feature is not fabricated.

The segmented model fits independent preprocessing and trees for apartments versus detached/garden houses. Routing at prediction uses the known property category, never the actual price.

## Models and selection

All candidates learn natural log price and predict exp(output), without a smearing correction. This is a target transformation, not an assertion of unbiased conditional mean price. It reduces the effect of extreme targets on fitting but can underpredict expensive properties.

Own trees retain the NumPy-only implementation. Random Forest (200 trees, leaf minimum 2, max_features=0.8, unrestricted depth) and histogram gradient boosting are sklearn benchmarks. Boosting uses learning_rate=0.07, leaf minimum 20, 300 rounds/31 leaves/L2=1 or 500 rounds/63 leaves/L2=3. Internal early stopping is disabled, so it cannot create a hidden random validation split across property groups. Seed remains 42.

Selected own model: own_spatial_segmented. Selected benchmark: rf_spatial. Selection uses validation RMSE. MAE and median percentage error are reported independently; they need not choose the same winner.

| model | sensitivity | rmse | mae | r2 | median_error_pct | within_20_pct |
| --- | --- | --- | --- | --- | --- | --- |
| own_all_categories | False | 177,350.3115 | 51,852.8846 | 0.6200 | 9.7482 | 0.7581 |
| own_residential | False | 175,594.0440 | 51,158.6370 | 0.6275 | 9.2580 | 0.7616 |
| own_segmented | False | 176,808.6158 | 49,355.6668 | 0.6223 | 9.0608 | 0.7665 |
| own_spatial | False | 171,215.9688 | 49,308.1528 | 0.6458 | 8.8372 | 0.7619 |
| own_spatial_segmented | False | 169,874.6401 | 49,242.0841 | 0.6514 | 8.7966 | 0.7685 |
| own_spatial_depth12_leaf10 | False | 175,863.9524 | 51,927.7713 | 0.6263 | 10.6239 | 0.7509 |
| own_spatial_depth20_leaf5 | False | 172,573.2972 | 49,232.8323 | 0.6402 | 8.4261 | 0.7643 |
| own_spatial_depth20_leaf20 | False | 175,986.0966 | 50,407.2096 | 0.6258 | 9.5227 | 0.7566 |
| own_audit_sensitivity | True | 172,601.8629 | 50,460.1913 | 0.6401 | 9.3404 | 0.7658 |
| rf_base | False | 159,114.8199 | 39,419.7349 | 0.6941 | 6.1806 | 0.8377 |
| rf_spatial | False | 158,215.2047 | 39,554.4230 | 0.6976 | 6.2094 | 0.8364 |
| hgb_base | False | 167,978.1839 | 45,803.8405 | 0.6591 | 9.4385 | 0.8026 |
| hgb_spatial | False | 166,437.0482 | 44,768.5823 | 0.6653 | 9.2460 | 0.8056 |
| hgb_spatial_500 | False | 159,346.9025 | 41,843.0963 | 0.6932 | 8.1034 | 0.8295 |
| hgb_audit_sensitivity | True | 165,324.6958 | 44,337.5987 | 0.6698 | 8.9535 | 0.8098 |
| own_source_review_sensitivity | True | 171,006.8116 | 49,168.8075 | 0.6467 | 8.7945 | 0.7664 |
| rf_source_review_sensitivity | True | 157,752.0524 | 39,685.3290 | 0.6993 | 6.2795 | 0.8323 |

within_20_pct is a proportion from 0 to 1, not a percentage in this table.

Relative to the original residential tree, the selected benchmark lowers RMSE by 9.90% and MAE by 22.68%. The selected own tree lowers RMSE by 3.26%.

Separate groups alone do not improve every metric. Spatial features and additional trees also do not guarantee improvement. The table retains unsuccessful experiments instead of only presenting the winner.

## Group cross-validation and uncertainty

Five-fold GroupKFold with shuffle=True and seed 42 operates only on the original residential training partition. Every fold refits all preprocessing, geographic centres and models. The fixed selected specifications are checked alongside the original residential tree.

This is a stability diagnostic, not nested model selection and not the final held-out test. The same external validation data has been used repeatedly during development, so its selected scores can be optimistic. Fold standard deviations are descriptive variation, not confidence intervals.

| model | rmse_mean | rmse_std | mae_mean | median_error_pct_mean | median_error_pct_std |
| --- | --- | --- | --- | --- | --- |
| own_residential | 1,430,935.3512 | 2,905,190.4984 | 62,875.2857 | 9.6513 | 0.1019 |
| own_spatial_segmented | 1,431,886.0582 | 2,904,140.9453 | 62,476.0983 | 9.2217 | 0.0544 |
| rf_spatial | 1,414,418.1588 | 2,913,930.9443 | 51,848.3249 | 6.8071 | 0.0506 |

All evaluation rows are retained. The fold containing the 600-million price has a very large RMSE and dominates the fold average. Reporting only the better folds or silently dropping that record would hide the data-quality uncertainty.

| fold | model | n_eval | max_eval_price | rmse | mae | median_error_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | own_residential | 8196 | 5,099,000.0000 | 120,583.0057 | 46,526.3143 | 9.7571 |
| 1 | own_spatial_segmented | 8196 | 5,099,000.0000 | 126,998.0480 | 46,434.4039 | 9.1924 |
| 1 | rf_spatial | 8196 | 5,099,000.0000 | 103,785.4477 | 35,943.6812 | 6.8831 |
| 2 | own_residential | 8190 | 600,000,000.0000 | 6,627,879.7933 | 122,332.4389 | 9.6654 |
| 2 | own_spatial_segmented | 8190 | 600,000,000.0000 | 6,626,946.7316 | 121,573.9788 | 9.2406 |
| 2 | rf_spatial | 8190 | 600,000,000.0000 | 6,627,005.5653 | 111,391.3610 | 6.8316 |
| 3 | own_residential | 8178 | 5,100,000.0000 | 141,307.4423 | 49,647.7753 | 9.6304 |
| 3 | own_spatial_segmented | 8178 | 5,100,000.0000 | 148,269.0011 | 49,662.8907 | 9.1517 |
| 3 | rf_spatial | 8178 | 5,100,000.0000 | 119,214.6228 | 38,170.4689 | 6.7682 |
| 4 | own_residential | 8172 | 5,950,000.0000 | 135,684.6660 | 47,467.6491 | 9.4904 |
| 4 | own_spatial_segmented | 8172 | 5,950,000.0000 | 126,201.2756 | 46,555.1365 | 9.2268 |
| 4 | rf_spatial | 8172 | 5,950,000.0000 | 114,253.2128 | 36,988.7734 | 6.7908 |
| 5 | own_residential | 8196 | 5,100,000.0000 | 129,221.8486 | 48,402.2507 | 9.7130 |
| 5 | own_spatial_segmented | 8196 | 5,100,000.0000 | 131,015.2345 | 48,154.0816 | 9.2971 |
| 5 | rf_spatial | 8196 | 5,100,000.0000 | 107,831.9454 | 36,747.3397 | 6.7615 |

## Category-level validation results

| model | category | n | rmse | mae | median_error_pct |
| --- | --- | --- | --- | --- | --- |
| own_residential | həyət evi/bağ evi | 1478 | 364,198.9153 | 117,426.6637 | 20.7751 |
| own_residential | köhnə tikili | 1846 | 82,620.7150 | 24,318.1389 | 7.3070 |
| own_residential | yeni tikili | 5472 | 106,928.4059 | 42,314.2320 | 8.4700 |
| own_spatial_segmented | həyət evi/bağ evi | 1478 | 350,099.3316 | 117,578.2478 | 20.8346 |
| own_spatial_segmented | köhnə tikili | 1846 | 62,997.5645 | 23,335.6376 | 7.5328 |
| own_spatial_segmented | yeni tikili | 5472 | 109,278.6365 | 39,523.9647 | 7.6300 |
| rf_spatial | həyət evi/bağ evi | 1478 | 336,021.4033 | 98,696.6620 | 16.6167 |
| rf_spatial | köhnə tikili | 1846 | 62,471.6266 | 18,949.2425 | 5.5180 |
| rf_spatial | yeni tikili | 5472 | 91,782.2058 | 30,531.2019 | 5.2023 |

## Interpretation and remaining work

A median percentage error near 6% does not mean 94% accuracy and does not bound error on each property. High RMSE, uncertain labels and a historical listing dataset limit use for real valuation. These are listing prices, not verified completed-sale prices.

The benchmark is an improvement reference; it does not replace the assignment’s own Decision Tree or Pegasos SVM requirements. The original classification threshold is unchanged. Pegasos/nonlinear SVM, final frozen test evaluation, IEEE PDF, slides, contribution records and final repository tag remain separate tasks.

33 automated tests cover preparation, own tree, metrics, spatial leakage, routing and source-review invariants. Timings are single-run and hardware-dependent; all models use the same data scope, but timing includes preprocessing.

## Sources and reproducibility

- UNESCO reference coordinates: https://whc.unesco.org/en/list/958/maps/

- Random Forest: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestRegressor.html

- Histogram boosting: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

- GroupKFold: https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html

Run `python -m src.improve_residential` from the project root. `--skip-cv` repeats the comparison without fold evaluation. Raw listing text, owner details and row-level validation predictions are excluded from the update ZIP.

AI assisted with implementation, data review and explanations. Team members should review the code, reproduce experiments, understand the methodology and disclose assistance. No team contributions are invented.