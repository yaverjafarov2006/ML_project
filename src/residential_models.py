"""Residential regression experiments. Own trees remain NumPy implementations.

sklearn is imported lazily ONLY for benchmark model construction. Spatial
centres, distances and training audit rules are implemented with NumPy/pandas.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .data_prep import FeatureEncoder
from .decision_tree import DecisionTree

RESIDENTIAL_CATEGORIES = ('yeni tikili', 'köhnə tikili', 'həyət evi/bağ evi')
# UNESCO reference point for the Walled City, not a metro entrance or a
# universal city centre. Source: https://whc.unesco.org/en/list/958/maps/
OLD_CITY_COORDINATES = np.array([40 + 22 / 60, 49 + 49 / 60 + 60 / 3600])

# Reviewed on original TRAIN rows only. The descriptions support these
# changes; they do not establish externally verified ground truth.
SOURCE_REVIEW = [
    dict(listing_id='6eda4ed03e9a3460dba08fb547f7e2b1ec72c814e562481756cdd0792277ba97',
         source_row=15608, action='exclude_rental', expected_price=400.,
         evidence='TƏCİLİ KİRAYƏ VERİLİR.'),
    dict(listing_id='2a18086af89b40e2d504755ba6c956e5e69fe650a6bece558571bd361fd081d6',
         source_row=39396, action='exclude_rental', expected_price=700.,
         evidence='2 otaqlı mənzil kirayə verilir'),
    dict(listing_id='605785158c89fde85e5273e02f48ed59b07a4beb9ca8236edf80f21bd2d0568a',
         source_row=15638, action='correct_area', expected_area=9., area_m2=650.,
         evidence='9 sot torpagin icinde ... 650kvadrat sahe'),
    dict(listing_id='ac89b031b4dafdafa5c8f773fe6bac6d8167359e8deb818330ec55ad267f4f5e',
         source_row=90432, action='correct_area', expected_area=11., area_m2=300.,
         land_area_m2=1100., evidence='11 sotun icində 300 kvadrat 2 mərtəbədən ibarətdir'),
]


def reviewed_training_frame(train):
    """Optional source-review sensitivity experiment; original frame unchanged.

    Never used at predict time or to prune validation/CV evaluation rows.
    Identifiers are lookup keys for curation only, never model features.
    """
    result = train.copy()
    for record in SOURCE_REVIEW:
        mask = result.listing_id.eq(record['listing_id'])
        if record['action'] == 'exclude_rental':
            mask &= result.price.eq(record['expected_price'])
            result = result.loc[~mask].copy()
        else:
            mask &= result.area_m2.eq(record['expected_area'])
            result.loc[mask, 'area_m2'] = record['area_m2']
            if 'land_area_m2' in record:
                result.loc[mask, 'land_area_m2'] = record['land_area_m2']
    result['log_area_m2'] = np.log(result.area_m2)
    result['area_per_room'] = result.area_m2 / result.rooms
    return result


def residential_rows(frame):
    return frame.loc[frame.category.isin(RESIDENTIAL_CATEGORIES)].copy()


def haversine_km(points, centres):
    """Pairwise great-circle distances; input order is latitude, longitude."""
    a = np.radians(np.asarray(points, dtype=float))[:, None, :]
    b = np.radians(np.asarray(centres, dtype=float))[None, :, :]
    delta = a - b
    h = np.sin(delta[..., 0] / 2) ** 2
    h += np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(delta[..., 1] / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def geographic_centres(points, n_centres=12, seed=42):
    """Train-only k-means on a local equirectangular projection, NumPy only.

    Centres summarize listing geography, not official districts or stations.
    A seeded 10,000-row training sample bounds runtime. Targets are never read.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return OLD_CITY_COORDINATES[None, :].copy()
    rng = np.random.default_rng(seed)
    if len(points) > 10000:
        points = points[rng.choice(len(points), 10000, replace=False)]
    origin = np.median(points, axis=0)
    scale = np.array([111.195, 111.195 * np.cos(np.radians(origin[0]))])
    xy = (points - origin) * scale
    n_centres = min(n_centres, len(np.unique(xy, axis=0)))
    centres = [xy[rng.integers(len(xy))]]
    for _ in range(1, n_centres):
        distances = np.min(np.sum((xy[:, None] - np.array(centres)) ** 2, axis=2), axis=1)
        centres.append(xy[rng.choice(len(xy), p=distances / distances.sum())])
    centres = np.array(centres)
    for _ in range(25):
        labels = np.argmin(np.sum((xy[:, None] - centres) ** 2, axis=2), axis=1)
        updated = np.array([xy[labels == j].mean(axis=0) if np.any(labels == j)
                            else centres[j] for j in range(n_centres)])
        if np.allclose(updated, centres, atol=1e-4):
            centres = updated
            break
        centres = updated
    return centres / scale + origin


class ResidentialEncoder:
    """Stage 1 features, optionally extended by geographical distances."""
    def __init__(self, spatial=False):
        self.spatial = spatial

    def fit(self, train):
        self.base_ = FeatureEncoder().fit(train)
        self.feature_names_ = list(self.base_.feature_names_)
        if self.spatial:
            self.centres_ = geographic_centres(train[['latitude', 'longitude']].to_numpy(float))
            self.extra_names_ = ['distance_old_city_km'] + [
                f'distance_train_geo_centre_{j:02d}_km' for j in range(len(self.centres_))]
            values = self._extra(train)
            self.extra_medians_ = np.array([
                np.nanmedian(values[:, j]) if np.isfinite(values[:, j]).any() else 0.
                for j in range(values.shape[1])])
            self.feature_names_ += self.extra_names_ + ['geo_distance_missing']
        return self

    def _extra(self, frame):
        points = frame[['latitude', 'longitude']].to_numpy(float)
        return haversine_km(points, np.vstack([OLD_CITY_COORDINATES, self.centres_]))

    def transform(self, frame):
        base = self.base_.transform(frame)
        if not self.spatial:
            return base
        extra = self._extra(frame)
        missing = ~np.isfinite(extra)
        extra = np.where(missing, self.extra_medians_, extra)
        return np.column_stack([base, extra, missing.any(axis=1).astype(float)])


class TrainingPriceAudit:
    """Flag extreme log(price/area) within each training category.

    This is a sensitivity rule, NOT proof of bad labels. Never apply it to
    remove validation/test observations or use price/area as a predictor.
    """
    def fit(self, train):
        self.bounds_ = {}
        for category, group in train.groupby('category', observed=True):
            z = np.log(group.price / group.area_m2)
            z = z[np.isfinite(z)]
            if len(z) < 20:
                continue
            q1, q3 = np.quantile(z, [.25, .75])
            iqr = q3 - q1
            self.bounds_[category] = (float(q1 - 3 * iqr), float(q3 + 3 * iqr))
        return self

    def flags(self, frame):
        z = np.log(frame.price / frame.area_m2).to_numpy(float)
        flags = np.zeros(len(frame), dtype=bool)
        categories = frame.category.to_numpy()
        for category, (low, high) in self.bounds_.items():
            flags |= (categories == category) & np.isfinite(z) & ((z < low) | (z > high))
        return flags


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str = 'own_tree'
    spatial: bool = False
    segmented: bool = False
    audit_filter: bool = False
    reviewed_training: bool = False
    max_depth: int | None = 16
    min_samples_leaf: int = 5
    iterations: int = 300
    max_leaf_nodes: int = 31
    l2: float = 1.

    def to_dict(self):
        return asdict(self)


# Predeclared finite experiment list; no test-based or repeated seed selection.
SPECS = [
    ModelSpec('own_residential'),
    ModelSpec('own_segmented', segmented=True),
    ModelSpec('own_spatial', spatial=True),
    ModelSpec('own_spatial_segmented', spatial=True, segmented=True),
    ModelSpec('own_spatial_depth12_leaf10', spatial=True, max_depth=12, min_samples_leaf=10),
    ModelSpec('own_spatial_depth20_leaf5', spatial=True, max_depth=20),
    ModelSpec('own_spatial_depth20_leaf20', spatial=True, max_depth=20, min_samples_leaf=20),
    ModelSpec('own_audit_sensitivity', audit_filter=True),
    ModelSpec('rf_base', kind='random_forest', max_depth=None, min_samples_leaf=2),
    ModelSpec('rf_spatial', kind='random_forest', spatial=True, max_depth=None, min_samples_leaf=2),
    ModelSpec('hgb_base', kind='hist_boosting', max_depth=None, min_samples_leaf=20),
    ModelSpec('hgb_spatial', kind='hist_boosting', spatial=True, max_depth=None, min_samples_leaf=20),
    ModelSpec('hgb_spatial_500', kind='hist_boosting', spatial=True,
              max_depth=None, min_samples_leaf=20, iterations=500, max_leaf_nodes=63, l2=3.),
    ModelSpec('hgb_audit_sensitivity', kind='hist_boosting', spatial=True,
              max_depth=None, min_samples_leaf=20, audit_filter=True),
    ModelSpec('own_source_review_sensitivity', spatial=True, segmented=True, reviewed_training=True),
    ModelSpec('rf_source_review_sensitivity', kind='random_forest', spatial=True,
              max_depth=None, min_samples_leaf=2, reviewed_training=True),
]


class ResidentialRegressor:
    """Log-price model; optional separate apartments/house models.

    At prediction time routing uses the known property category, never price.
    All preprocessing is fitted inside fit(), including on each CV fold.
    """
    def __init__(self, spec):
        self.spec = spec

    def _new_model(self):
        if self.spec.kind == 'own_tree':
            return DecisionTree(task='regression', max_depth=self.spec.max_depth,
                                min_samples_leaf=self.spec.min_samples_leaf, random_state=42)
        if self.spec.kind == 'sklearn_tree':
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(max_depth=self.spec.max_depth,
                min_samples_leaf=self.spec.min_samples_leaf, random_state=42)
        if self.spec.kind == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, max_depth=self.spec.max_depth,
                                         min_samples_leaf=self.spec.min_samples_leaf,
                                         max_features=.8, n_jobs=2, random_state=42)
        if self.spec.kind == 'hist_boosting':
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_iter=self.spec.iterations,
                learning_rate=.07, max_leaf_nodes=self.spec.max_leaf_nodes,
                max_depth=self.spec.max_depth, min_samples_leaf=self.spec.min_samples_leaf,
                l2_regularization=self.spec.l2,
                early_stopping=False, random_state=42)
        raise ValueError('Unknown benchmark kind')

    def _routes(self, frame):
        if not frame.category.isin(RESIDENTIAL_CATEGORIES).all():
            raise ValueError('This model supports only the declared residential categories')
        if not self.spec.segmented:
            return np.full(len(frame), 'all', dtype=object)
        return np.where(frame.category.eq('həyət evi/bağ evi'), 'house', 'apartment')

    def fit(self, train):
        train = train.copy()
        self.removed_rows_ = 0
        if self.spec.reviewed_training:
            before = len(train)
            train = reviewed_training_frame(train)
            self.removed_rows_ += before - len(train)
        if self.spec.audit_filter:
            self.audit_ = TrainingPriceAudit().fit(train)
            flags = self.audit_.flags(train)
            self.removed_rows_ += int(flags.sum())
            train = train.loc[~flags].copy()
        routes = self._routes(train)
        self.encoders_, self.models_ = {}, {}
        for route in np.unique(routes):
            part = train.loc[routes == route]
            encoder = ResidentialEncoder(self.spec.spatial).fit(part)
            model = self._new_model().fit(encoder.transform(part), np.log(part.price.to_numpy(float)))
            self.encoders_[route], self.models_[route] = encoder, model
        return self

    def predict(self, frame):
        routes = self._routes(frame)
        predictions = np.empty(len(frame), dtype=float)
        for route in np.unique(routes):
            if route not in self.models_:
                raise ValueError(f'No training observations for route: {route}')
            mask = routes == route
            x = self.encoders_[route].transform(frame.loc[mask])
            predictions[mask] = np.exp(self.models_[route].predict(x))
        return predictions
