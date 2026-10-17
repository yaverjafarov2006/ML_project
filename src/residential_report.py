"""Aggregate plots and a reproducible experiment report (no personal data)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_prep import ROOT

LABELS = {
    'own_residential': 'Original residential tree',
    'own_spatial_segmented': 'Own tree: spatial + separate groups',
    'rf_base': 'Random Forest: base features',
    'rf_spatial': 'Random Forest: spatial features',
    'hgb_spatial_500': 'Histogram boosting: spatial, 500 rounds',
}


def markdown_table(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f'{value:,.4f}' if isinstance(value, (float, np.floating)) else str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(output_dir=None):
    out = Path(output_dir or ROOT / 'outputs')
    summary = json.loads((out / 'residential_experiment_summary.json').read_text(encoding='utf-8'))
    table = pd.read_csv(out / 'residential_validation_comparison.csv')
    category = pd.read_csv(out / 'residential_error_by_category.csv')
    cv_path = out / 'residential_cv_folds.csv'
    cv = pd.read_csv(cv_path) if summary['cv'] and cv_path.exists() else pd.DataFrame()
    figdir = ROOT / 'report' / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    view = table.set_index('model').loc[list(LABELS)].copy()
    labels = [LABELS[name] for name in view.index]
    colors = ['#9aa9b5', '#2878a5', '#18a3a5', '#095c6e', '#9a6b35']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, metric, title, multiplier in zip(axes.flat,
        ['rmse', 'mae', 'median_error_pct', 'within_20_pct'],
        ['RMSE (AZN): lower is better', 'MAE (AZN): lower is better',
         'Median absolute percentage error', 'Predictions within 20% of actual price'],
        [1, 1, 1, 100]):
        values = view[metric].to_numpy() * multiplier
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(0, values.max() * 1.22)
        ax.grid(axis='x', alpha=.15)
        for i, value in enumerate(values):
            label = f'{value:,.0f}' if metric in ['rmse', 'mae'] else f'{value:.2f}%'
            ax.text(value + values.max() * .015, i, label, va='center', fontsize=9)
    fig.suptitle(f"Same {summary['validation_rows']:,} residential validation properties; no test evaluation", fontsize=15)
    fig.savefig(figdir / '06_residential_comparison.png', dpi=150)
    plt.close(fig)

    cache = out / 'residential_predictions_local.npz'
    if cache.exists():
        predictions = np.load(cache)
        y = predictions['y']; pred = predictions[summary['best_benchmark']]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        ax = axes[0, 0]
        ax.hexbin(y, pred, xscale='log', yscale='log', mincnt=1, gridsize=45, cmap='viridis')
        limits = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax.plot(limits, limits, 'r--', linewidth=1)
        ax.set(title='Selected benchmark: actual vs predicted', xlabel='Actual price (AZN, log scale)', ylabel='Predicted price (AZN, log scale)')
        ax = axes[0, 1]
        for name in predictions.files:
            if name == 'y': continue
            error = np.sort(100 * np.abs(y - predictions[name]) / y)
            ax.plot(error, 100 * np.arange(1, len(error) + 1) / len(error), label=LABELS.get(name, name))
        ax.set(xlim=(0, 60), ylim=(0, 100), xlabel='Absolute percentage error threshold', ylabel='Properties within threshold (%)', title='Validation error distribution')
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        ax = axes[1, 0]
        pivot = category.pivot(index='category', columns='model', values='median_error_pct')
        pivot.rename(columns=LABELS).plot.bar(ax=ax)
        ax.set(title='Median percentage error by property category', ylabel='Error (%)', xlabel='')
        ax.tick_params(axis='x', rotation=12); ax.legend(fontsize=7)
        ax = axes[1, 1]
        if len(cv):
            for name, part in cv.groupby('model', sort=False):
                ax.plot(part.fold, part.median_error_pct, 'o-', label=LABELS.get(name, name))
            ax.set(title='Original training partition: group CV', xlabel='Fold', ylabel='Median absolute percentage error')
            ax.set_xticks([1, 2, 3, 4, 5]); ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.savefig(figdir / '07_residential_diagnostics.png', dpi=150)
        plt.close(fig)

    baseline = table.set_index('model').loc['own_residential']
    winner = table.set_index('model').loc[summary['best_benchmark']]
    own = table.set_index('model').loc[summary['best_own_tree']]
    audit = summary['train_audit']
    lines = [
        '# Residential regression experiments — Stage 2.1', '',
        '## Scope and protocol',
        f"Training: {summary['train_rows']:,}; validation: {summary['validation_rows']:,}. Only new/old apartments and detached/garden houses are included. Offices, commercial properties, land and garages are excluded from this explicitly labelled subtask.",
        'Original split membership and property groups are preserved. The original full-dataset task and its earlier results remain available. No held-out test predictions were computed. No validation rows were removed based on price, prediction error or audit flags.',
        'All reported validation model comparisons use the same residential rows. A reduction from the former full-dataset RMSE is not itself evidence of improved training. The full-scope model is re-evaluated on the same residential rows as a control.',
        '## Training audit and source review',
        f"The training-only category-specific log(price/area) rule flags {audit['flagged_price_per_m2']} observations outside Q1 − 3 IQR / Q3 + 3 IQR. A separate area-per-room review flag identifies {audit['flagged_area_per_room']} rows. These are review flags, not confirmed errors.",
        'A 200 m² apartment has a recorded price of 600,000,000 AZN, and other records show 73, 400 and 700 AZN. Unknown true prices are not invented or replaced. The 600-million record has no usable description to resolve its true price.',
        'Review of training descriptions identifies the 400 and 700 AZN records as rentals. Two house descriptions explicitly state floor areas of 650 m² and 300 m² where the table says 9 m² and 11 m²; the latter also states 11 sot of land instead of the table’s 310 sot. The source-review experiment excludes the two rentals from training and uses those described areas. It is source-supported curation, not independently verified ground truth.',
        'residential_source_review.json records hashed listing keys, original row references, actions and short source excerpts. Lookup IDs are never predictors. Corrections operate on a copy of training data only; derived area features are recomputed. Raw CSV and all evaluation rows remain unchanged.',
        'Sensitivity models for broad audit filtering and source-reviewed training are presented separately and are not used to select the main models. The remaining ambiguous 14.5 m² record, price 73 and price 1,600 are left for review.',
        '## Spatial features and separate models',
        'Extra features are great-circle distances in kilometres to a UNESCO Walled City reference point and to 12 geographic centres estimated solely from training coordinates. NumPy k-means uses a seeded training sample of at most 10,000 rows. The original latitude/longitude and categorical location features remain available.',
        'These distances are not walking/driving distances. Learned centres are not metro stations or official districts. Verified station coordinates were not supplied, so a metro-distance feature is not fabricated.',
        'The segmented model fits independent preprocessing and trees for apartments versus detached/garden houses. Routing at prediction uses the known property category, never the actual price.',
        '## Models and selection',
        'All candidates learn natural log price and predict exp(output), without a smearing correction. This is a target transformation, not an assertion of unbiased conditional mean price. It reduces the effect of extreme targets on fitting but can underpredict expensive properties.',
        'Own trees retain the NumPy-only implementation. Random Forest (200 trees, leaf minimum 2, max_features=0.8, unrestricted depth) and histogram gradient boosting are sklearn benchmarks. Boosting uses learning_rate=0.07, leaf minimum 20, 300 rounds/31 leaves/L2=1 or 500 rounds/63 leaves/L2=3. Internal early stopping is disabled, so it cannot create a hidden random validation split across property groups. Seed remains 42.',
        f"Selected own model: {summary['best_own_tree']}. Selected benchmark: {summary['best_benchmark']}. Selection uses validation RMSE. MAE and median percentage error are reported independently; they need not choose the same winner.",
        markdown_table(table, ['model', 'sensitivity', 'rmse', 'mae', 'r2', 'median_error_pct', 'within_20_pct']),
        'within_20_pct is a proportion from 0 to 1, not a percentage in this table.',
        f"Relative to the original residential tree, the selected benchmark lowers RMSE by {100*(1-winner.rmse/baseline.rmse):.2f}% and MAE by {100*(1-winner.mae/baseline.mae):.2f}%. The selected own tree lowers RMSE by {100*(1-own.rmse/baseline.rmse):.2f}%.",
        'Separate groups alone do not improve every metric. Spatial features and additional trees also do not guarantee improvement. The table retains unsuccessful experiments instead of only presenting the winner.',
        '## Group cross-validation and uncertainty',
        'Five-fold GroupKFold with shuffle=True and seed 42 operates only on the original residential training partition. Every fold refits all preprocessing, geographic centres and models. The fixed selected specifications are checked alongside the original residential tree.',
        'This is a stability diagnostic, not nested model selection and not the final held-out test. The same external validation data has been used repeatedly during development, so its selected scores can be optimistic. Fold standard deviations are descriptive variation, not confidence intervals.',
    ]
    if len(cv):
        lines += [markdown_table(pd.read_csv(out / 'residential_cv_summary.csv'),
            ['model', 'rmse_mean', 'rmse_std', 'mae_mean', 'median_error_pct_mean', 'median_error_pct_std']),
            'All evaluation rows are retained. The fold containing the 600-million price has a very large RMSE and dominates the fold average. Reporting only the better folds or silently dropping that record would hide the data-quality uncertainty.',
            markdown_table(cv, ['fold', 'model', 'n_eval', 'max_eval_price', 'rmse', 'mae', 'median_error_pct'])]
    lines += ['## Category-level validation results',
        markdown_table(category, ['model', 'category', 'n', 'rmse', 'mae', 'median_error_pct']),
        '## Interpretation and remaining work',
        'A median percentage error near 6% does not mean 94% accuracy and does not bound error on each property. High RMSE, uncertain labels and a historical listing dataset limit use for real valuation. These are listing prices, not verified completed-sale prices.',
        'The benchmark is an improvement reference; it does not replace the assignment’s own Decision Tree or Pegasos SVM requirements. The original classification threshold is unchanged. Pegasos/nonlinear SVM, final frozen test evaluation, IEEE PDF, slides, contribution records and final repository tag remain separate tasks.',
        '33 automated tests cover preparation, own tree, metrics, spatial leakage, routing and source-review invariants. Timings are single-run and hardware-dependent; all models use the same data scope, but timing includes preprocessing.',
        '## Sources and reproducibility',
        '- UNESCO reference coordinates: https://whc.unesco.org/en/list/958/maps/',
        '- Random Forest: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
        '- Histogram boosting: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html',
        '- GroupKFold: https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html',
        'Run `python -m src.improve_residential` from the project root. `--skip-cv` repeats the comparison without fold evaluation. Raw listing text, owner details and row-level validation predictions are excluded from the update ZIP.',
        'AI assisted with implementation, data review and explanations. Team members should review the code, reproduce experiments, understand the methodology and disclose assistance. No team contributions are invented.',
    ]
    (out / 'RESIDENTIAL_EXPERIMENT_REPORT.md').write_text('\n\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    write_report()
