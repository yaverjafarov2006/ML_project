"""Meaningful data checks: units, group isolation, train-only statistics."""
import unittest
import numpy as np
import pandas as pd
from src.data_prep import (NUMERIC_FEATURES,CATEGORICAL_FEATURES,parse_area,clean,
                           make_splits,make_tier_label,FeatureEncoder,standardize)


def raw_fixture():
    rows = []
    for i in range(80):
        rows.append({"estate_rel_url":f"/items/{i}","datetime_scrape_y":"2024-10-05 12:00:00+00",
                     "price":10000+i*1000,"total_price":10000+i*1000,"unit_price":"100 AZN/m²",
                     "currency_x":"AZN","currency_y":"AZN","Sahə":"80 m²","Otaq sayı":2.,
                     "Mərtəbə":"3 / 5","Kateqoriya":"Yeni tikili","Torpaq sahəsi":np.nan,
                     "Təmir":"var","Çıxarış":"var","İpoteka":np.nan,"lat":40.4+i*.0001,
                     "lng":49.8,"city":"Bakı","location":"Example","description":f"Short text {i}"})
    return pd.DataFrame(rows)


class PreparationTests(unittest.TestCase):
    def test_units_and_invalid_values(self):
        a = parse_area(pd.Series(["100 m²","2.5 sot","2,5 sot","1 200 m2",None,"-5 m²","3 acres","0 m²"]))
        np.testing.assert_allclose(a[:4],[100,250,250,1200])
        self.assertTrue(a[4:].isna().all())

    def test_latest_snapshot_and_no_input_mutation(self):
        raw = raw_fixture()
        later = raw.iloc[[0]].copy()
        later.loc[:,"datetime_scrape_y"] = "2024-11-01 12:00:00+00"
        later.loc[:,"price"] = 123456
        combined = pd.concat([raw,later],ignore_index=True)
        saved = combined.copy(deep=True)
        out = clean(combined)
        self.assertEqual(len(out),80)
        self.assertIn(123456,out.price.to_list())
        self.assertNotIn("unit_price",out)
        self.assertNotIn("total_price",out)
        self.assertNotIn("description",out)
        pd.testing.assert_frame_equal(saved,combined)

    def test_cross_listing_group_does_not_use_price(self):
        raw = raw_fixture()
        desc = "The same exact long property description with sufficient details for conservative duplicate grouping."
        raw.loc[0,"description"] = desc
        copy = raw.iloc[[0]].copy()
        copy.loc[:,"estate_rel_url"] = "/items/another"
        copy.loc[:,"price"] = 900000
        out = clean(pd.concat([raw,copy],ignore_index=True))
        self.assertEqual(out.attrs["audit"]["additional_rows_in_shared_property_groups"],1)
        split = make_splits(out)
        sets = [set(x.group_id) for x in [split.train,split.val,split.test]]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_unknown_currency_stops(self):
        raw = raw_fixture(); raw.loc[0,"currency_x"] = "USD"
        with self.assertRaises(ValueError): clean(raw)

    def test_physical_parsing_and_missing_mortgage(self):
        raw = raw_fixture()
        raw.loc[0,["Sahə","Kateqoriya"]] = ["2 sot","Torpaq"]
        raw.loc[0,["lat","lng"]] = [12.,12.]
        out = clean(raw)
        land = out[out.category.eq("torpaq")].iloc[0]
        self.assertEqual(land.area_m2,200)
        self.assertEqual(land.land_area_m2,200)
        self.assertTrue(pd.isna(land.latitude))
        self.assertTrue(out.mortgage.isna().all())
        self.assertAlmostEqual(out.floor_ratio.iloc[0],.6)

    def test_training_only_threshold_and_repeatability(self):
        frame = clean(raw_fixture())
        a,b = make_splits(frame),make_splits(frame)
        self.assertEqual(a.train.source_row.tolist(),b.train.source_row.tolist())
        self.assertEqual(a.test.source_row.tolist(),b.test.source_row.tolist())
        self.assertEqual(a.threshold,float(np.median(a.train.price)))
        changed = frame.copy()
        # Deliberately alter held-out prices; training membership/median stay fixed.
        held = ~changed.group_id.isin(a.train.group_id)
        changed.loc[held,"price"] *= 1.05
        c = make_splits(changed)
        self.assertEqual(a.threshold,c.threshold)
        self.assertEqual(a.train.source_row.tolist(),c.train.source_row.tolist())
        labels,_ = make_tier_label([a.threshold,a.threshold+1],a.threshold)
        np.testing.assert_array_equal(labels,[0,1])

    def test_encoder_fits_training_only_unknowns_and_missing(self):
        frame = clean(raw_fixture())
        train = frame.iloc[:40].copy()
        train["area_m2"] = 80.
        test = frame.iloc[40:].copy()
        test["area_m2"] = np.nan
        test["location"] = "unseen district"
        encoder = FeatureEncoder().fit(train)
        before = encoder.to_dict()
        x = encoder.transform(test)
        self.assertEqual(encoder.to_dict(),before)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x[:,encoder.feature_names_.index("area_m2")],80)
        np.testing.assert_allclose(x[:,encoder.feature_names_.index("location=<UNKNOWN>")],1)
        np.testing.assert_allclose(x[:,encoder.feature_names_.index("mortgage=<MISSING>")],1)
        self.assertFalse(any(c in encoder.feature_names_ for c in ["price","unit_price","total_price","source_row","group_id"]))
        self.assertTrue(np.isfinite(encoder.transform(test,scaled=True)).all())

    def test_standardization_uses_training_statistics(self):
        tr,te = standardize(np.array([[0.,2.],[2.,2.]]),np.array([[101.,2.]]))
        np.testing.assert_allclose(tr,[[-1,0],[1,0]])
        np.testing.assert_allclose(te,[[100,0]])


if __name__ == "__main__":
    unittest.main()
