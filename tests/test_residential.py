"""Leakage, distance, routing and audit invariants for the new experiments."""
import unittest
import numpy as np
import pandas as pd
from test_data_prep import raw_fixture
from src.data_prep import clean
from src.residential_models import (haversine_km, ResidentialEncoder,
    ResidentialRegressor, ModelSpec, TrainingPriceAudit, SOURCE_REVIEW, reviewed_training_frame)


class ResidentialTests(unittest.TestCase):
    def test_distance_units_and_identity(self):
        d = haversine_km([[0, 0]], [[0, 0], [0, 1]])
        self.assertAlmostEqual(d[0, 0], 0.)
        self.assertAlmostEqual(d[0, 1], 111.19508, places=4)

    def test_spatial_features_ignore_target(self):
        frame = clean(raw_fixture())
        a = ResidentialEncoder(spatial=True).fit(frame)
        changed = frame.copy(); changed['price'] *= 1000
        b = ResidentialEncoder(spatial=True).fit(changed)
        np.testing.assert_allclose(a.centres_, b.centres_)
        np.testing.assert_allclose(a.transform(frame), b.transform(changed))
        self.assertFalse(any(name in a.feature_names_ for name in ['price', 'unit_price', 'source_row']))

    def test_heldout_transform_does_not_refit(self):
        frame = clean(raw_fixture())
        encoder = ResidentialEncoder(spatial=True).fit(frame.iloc[:40])
        centres = encoder.centres_.copy()
        medians = encoder.extra_medians_.copy()
        test = frame.iloc[40:].copy()
        test[['latitude', 'longitude']] = [41.9, 45.2]
        encoder.transform(test)
        np.testing.assert_array_equal(centres, encoder.centres_)
        np.testing.assert_array_equal(medians, encoder.extra_medians_)

    def test_missing_coordinates_are_explicit_and_finite(self):
        frame = clean(raw_fixture())
        frame[['latitude', 'longitude']] = np.nan
        encoder = ResidentialEncoder(spatial=True).fit(frame)
        x = encoder.transform(frame)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:, -1], 1)

    def test_segmentation_uses_known_category_not_price(self):
        frame = clean(raw_fixture())
        frame.loc[:39, 'category'] = 'yeni tikili'
        frame.loc[40:, 'category'] = 'həyət evi/bağ evi'
        frame.loc[:39, 'price'] = 100000.
        frame.loc[40:, 'price'] = 300000.
        saved = frame.copy(deep=True)
        model = ResidentialRegressor(ModelSpec('test', segmented=True)).fit(frame)
        pred = model.predict(frame)
        np.testing.assert_allclose(pred, frame.price)
        changed = frame.copy(); changed['price'] = 99999999.
        np.testing.assert_allclose(model.predict(changed), pred)
        pd.testing.assert_frame_equal(saved, frame)

    def test_audit_is_train_fitted_and_retains_normal_rows(self):
        frame = clean(raw_fixture())
        frame['area_m2'] = 100.
        frame['price'] = np.linspace(100000, 300000, len(frame))
        frame.loc[0, 'price'] = 600000000.
        audit = TrainingPriceAudit().fit(frame)
        flags = audit.flags(frame)
        self.assertTrue(flags[0]); self.assertEqual(int(flags.sum()), 1)
        bounds = dict(audit.bounds_)
        altered = frame.copy(); altered['price'] *= 10000
        audit.flags(altered)
        self.assertEqual(audit.bounds_, bounds)

    def test_out_of_scope_categories_rejected(self):
        frame = clean(raw_fixture())
        model = ResidentialRegressor(ModelSpec('test')).fit(frame)
        frame.loc[0, 'category'] = 'qaraj'
        with self.assertRaises(ValueError):
            model.predict(frame)

    def test_source_review_preserves_original_and_updates_derived_features(self):
        frame = clean(raw_fixture()).iloc[:4].copy()
        for i, record in enumerate(SOURCE_REVIEW):
            frame.loc[i, 'listing_id'] = record['listing_id']
            if record['action'] == 'exclude_rental':
                frame.loc[i, 'price'] = record['expected_price']
            else:
                frame.loc[i, 'area_m2'] = record['expected_area']
        original = frame.copy(deep=True)
        revised = reviewed_training_frame(frame)
        self.assertEqual(len(revised), 2)
        np.testing.assert_allclose(revised.area_m2, [650, 300])
        np.testing.assert_allclose(revised.log_area_m2, np.log([650, 300]))
        np.testing.assert_allclose(revised.area_per_room, revised.area_m2 / revised.rooms)
        self.assertEqual(revised.loc[3, 'land_area_m2'], 1100)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == '__main__':
    unittest.main()
