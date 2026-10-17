"""Reproducible residential experiments; held-out test is never predicted.

Run: python -m src.improve_residential [--skip-cv]
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data_prep import ROOT, DATA_PATH, clean, load_raw, make_splits, FeatureEncoder
from .decision_tree import DecisionTree
from .evaluate import regression_metrics
from .residential_models import (SPECS, ModelSpec, ResidentialRegressor,
                                 TrainingPriceAudit, residential_rows, SOURCE_REVIEW)


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    error_pct = 100 * np.abs(y - prediction) / y
    return dict(**regression_metrics(y, prediction),
                median_absolute_error=float(np.median(np.abs(y - prediction))),
                median_error_pct=float(np.median(error_pct)),
                within_10_pct=float(np.mean(error_pct <= 10)),
                within_20_pct=float(np.mean(error_pct <= 20)),
                p90_error_pct=float(np.quantile(error_pct, .9)))


def load_frames(data_path=DATA_PATH):
    split = make_splits(clean(load_raw(data_path)))
    return split.train, residential_rows(split.train), residential_rows(split.val)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_audit(train, out):
    audit = TrainingPriceAudit().fit(train)
    flags = audit.flags(train)
    per_room = train.area_m2 / train.rooms
    physical = per_room.lt(4) | per_room.gt(300)
    rows = train.loc[flags | physical, ['source_row', 'category', 'price', 'area_m2',
                                     'land_area_m2', 'rooms', 'city', 'location']].copy()
    rows['price_per_m2_audit_only'] = rows.price / rows.area_m2
    rows['extreme_price_per_m2'] = flags[flags | physical.to_numpy()]
    rows['area_per_room_review'] = physical.loc[rows.index].to_numpy()
    rows.to_csv(out / 'residential_train_review.csv', index=False)
    (out / 'residential_source_review.json').write_text(json.dumps(SOURCE_REVIEW, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(train_rows=len(train), flagged_price_per_m2=int(flags.sum()),
        flagged_area_per_room=int(physical.sum()), union_review_rows=len(rows),
        category_counts={k: int(v) for k, v in train.category.value_counts().items()},
        price_quantiles={str(k): float(v) for k, v in train.price.quantile([0, .01, .5, .99, 1]).items()},
        missing_area_rows=int(train.area_m2.isna().sum()),
        bounds_price_per_m2={k: list(np.exp(v)) for k, v in audit.bounds_.items()},
        decision='No target or area corrections without source confirmation. Audit-filter experiments are sensitivity analyses only.',
        spatial_reference=dict(name='UNESCO Walled City reference point', latitude=40 + 22 / 60,
            longitude=49 + 49 / 60 + 60 / 3600, source='https://whc.unesco.org/en/list/958/maps/'),
        distance_definition='Great-circle km; not road/walking distance. Learned centres use training coordinates only; they are not metro stations.')
    (out / 'residential_train_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def run_cv(train, specs, out):
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(train, groups=train.group_id))
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        fit_frame, eval_frame = train.iloc[tr_idx], train.iloc[va_idx]
        assert not set(fit_frame.group_id) & set(eval_frame.group_id)
        for spec in specs:
            start = perf_counter()
            model = ResidentialRegressor(spec).fit(fit_frame)
            pred = model.predict(eval_frame)
            row = dict(fold=fold, model=spec.name, n_train=len(fit_frame), n_eval=len(eval_frame),
                       max_eval_price=float(eval_frame.price.max()),
                       seconds=perf_counter() - start, **metrics(eval_frame.price, pred))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'residential_cv_folds.csv', index=False)
            print(f'CV {fold}/5 {spec.name}: RMSE={row["rmse"]:,.0f}, median error={row["median_error_pct"]:.2f}%', flush=True)
    frame = pd.DataFrame(rows)
    summaries = []
    for name, part in frame.groupby('model', sort=False):
        row = {'model': name, 'folds': len(part)}
        for metric in ['rmse', 'mae', 'r2', 'median_error_pct', 'within_20_pct']:
            row[metric + '_mean'] = float(part[metric].mean())
            row[metric + '_std'] = float(part[metric].std(ddof=1))
        summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(out / 'residential_cv_summary.csv', index=False)
    return summaries


def run(data_path=DATA_PATH, output_dir=None, do_cv=True, cached_frames=None):
    out = Path(output_dir or ROOT / 'outputs')
    out.mkdir(parents=True, exist_ok=True)
    all_train, train, val = cached_frames if cached_frames is not None else load_frames(data_path)
    assert not set(train.group_id) & set(val.group_id)
    audit = train_audit(train, out)
    yv = val.price.to_numpy(float)
    rows, predictions = [], {}
    with threadpool_limits(limits=2):
        # Recreate the user's full-scope depth=16/leaf=5 model on the same rows.
        start = perf_counter()
        encoder = FeatureEncoder().fit(all_train)
        old = DecisionTree(task='regression', max_depth=16, min_samples_leaf=5, random_state=42)
        old.fit(encoder.transform(all_train), np.log(all_train.price.to_numpy(float)))
        pred = np.exp(old.predict(encoder.transform(val)))
        predictions['own_all_categories'] = pred
        rows.append(dict(model='own_all_categories', kind='own_tree', spatial=False,
                         segmented=False, sensitivity=False, train_rows=len(all_train),
                         fit_seconds=perf_counter()-start, predict_seconds=None,
                         **metrics(yv, pred)))
        print('Full-scope control:', rows[-1]['rmse'], flush=True)
        for spec in SPECS:
            start = perf_counter()
            model = ResidentialRegressor(spec).fit(train)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = model.predict(val)
            predict_seconds = perf_counter() - start
            predictions[spec.name] = pred
            row = dict(model=spec.name, kind=spec.kind, spatial=spec.spatial,
                segmented=spec.segmented, sensitivity=spec.audit_filter or spec.reviewed_training,
                train_rows=len(train)-model.removed_rows_, fit_seconds=fit_seconds,
                predict_seconds=predict_seconds, **metrics(yv, pred))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'residential_validation_comparison.csv', index=False)
            print(f'{spec.name}: RMSE={row["rmse"]:,.0f}; MAE={row["mae"]:,.0f}; R2={row["r2"]:.4f}; fit={fit_seconds:.1f}s', flush=True)
        table = pd.DataFrame(rows)
        candidates = table[(~table.sensitivity) & table.model.ne('own_all_categories')]
        own = candidates[candidates.kind.eq('own_tree')].sort_values('rmse').iloc[0].model
        benchmark = candidates[~candidates.kind.eq('own_tree')].sort_values('rmse').iloc[0].model
        names = list(dict.fromkeys(['own_residential', own, benchmark]))
        selected_specs = [next(spec for spec in SPECS if spec.name == name) for name in names]
        categories = []
        for name in names:
            for category in sorted(val.category.unique()):
                mask = val.category.eq(category).to_numpy()
                categories.append(dict(model=name, category=category, n=int(mask.sum()),
                                       **metrics(yv[mask], predictions[name][mask])))
        pd.DataFrame(categories).to_csv(out / 'residential_error_by_category.csv', index=False)
        cv = run_cv(train, selected_specs, out) if do_cv else []
    summary = dict(stage='2.1', scope='new/old apartments and detached/garden houses',
        input_sha256=file_digest(data_path),
        excluded_categories=['ofis', 'obyekt', 'torpaq', 'qaraj'],
        train_rows=len(train), validation_rows=len(val), evaluation_split='validation',
        test_evaluated=False, seed=42, target='natural log price; exp inverse without smearing',
        selection_metric='validation RMSE, excluding audit-filter and source-review sensitivity models',
        best_own_tree=own, best_benchmark=benchmark,
        selected_specs={spec.name: spec.to_dict() for spec in selected_specs},
        experiment_specs=[spec.to_dict() for spec in SPECS],
        validation=rows, cv=cv, train_audit=audit,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        limitations=['Repeated validation use makes selection results optimistic.',
            'Five-fold group CV is on the original training partition; hyperparameters were selected on separate validation data. It is a stability diagnostic, not nested CV or a final test.',
            'CV fold standard deviations are not confidence intervals.',
            'Suspected label errors are retained in all evaluation rows.',
            'Spatial proximity, remaining duplicate properties and temporal drift can limit generalization.',
            'The regression scope change does not silently redefine the original classification threshold.'])
    (out / 'residential_experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Private row-level cache for local figures/review, excluded from delivery.
    np.savez_compressed(out / 'residential_predictions_local.npz', y=yv,
                        **{name: predictions[name] for name in names})
    from .residential_report import write_report
    write_report(out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(DATA_PATH))
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args()
    run(args.data, do_cv=not args.skip_cv)
