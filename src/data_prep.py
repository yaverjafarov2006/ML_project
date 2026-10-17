"""Auditable cleaning and train-only preprocessing (NumPy + pandas).

Fit a NEW FeatureEncoder inside every future CV fold. Never fit on the full
dataset before splitting. Model features use an explicit safe allowlist.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "house_sale.csv"
LEAKAGE_COLUMNS = ["unit_price", "total_price"]
NUMERIC_FEATURES = ["area_m2", "land_area_m2", "rooms", "floor", "total_floors",
                    "floor_ratio", "latitude", "longitude", "log_area_m2", "area_per_room"]
CATEGORICAL_FEATURES = ["city", "location", "category", "repair", "bill_of_sale", "mortgage"]
COLUMN_MAP = {"Sahə":"area_m2", "Torpaq sahəsi":"land_area_m2", "Otaq sayı":"rooms",
              "Mərtəbə":"floor + total_floors", "Kateqoriya":"category", "Təmir":"repair",
              "Çıxarış":"bill_of_sale", "İpoteka":"mortgage", "lat":"latitude", "lng":"longitude"}


def load_raw(path=DATA_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Put house_sale.csv in data/ or use --data PATH.")
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def parse_area(values):
    """Explicit units only: m², m2, sot. One sot is 100 square metres."""
    parts = values.astype("string").str.lower().str.strip().str.extract(r"^([\d\s.,]+)\s*(m²|m2|sot)$")
    number = pd.to_numeric(parts[0].str.replace(r"\s+", "", regex=True).str.replace(",", ".", regex=False), errors="coerce").astype(float)
    result = number * np.where(parts[1].eq("sot").fillna(False), 100., 1.)
    return result.where(np.isfinite(result) & result.gt(0))


def _text(values):
    return values.astype("string").str.strip().str.lower().replace("", pd.NA)


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def clean(df):
    """Apply row-level rules, never medians/quantiles. Counts: attrs['audit'].

    Keep latest snapshot per canonical URL. Group possible cross-listing
    copies by exact long description plus physical signature (no price).
    Retain positive price extremes; do not improve the test set by trimming.
    """
    needed = ["estate_rel_url", "datetime_scrape_y", "price", "currency_x", "currency_y",
              "Sahə", "Otaq sayı", "Mərtəbə", "Kateqoriya", "Torpaq sahəsi", "Təmir",
              "Çıxarış", "İpoteka", "lat", "lng", "city", "location", "description"]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    d = df.copy(deep=True)
    d["source_row"] = np.arange(len(d), dtype=int)
    audit = {"raw_rows":len(d), "raw_columns":len(df.columns),
             "exact_duplicate_rows":int(df.duplicated().sum()),
             "raw_missing_counts":{c:int(v) for c,v in df.isna().sum().items()},
             "raw_unique_listing_urls":int(df.estate_rel_url.nunique()),
             "currency_counts":{str(k):int(v) for k,v in df.currency_x.value_counts().items()},
             "target_derived_columns_removed":[c for c in LEAKAGE_COLUMNS if c in df],
             "price_total_price_mismatches":int((df.price != df.total_price).sum()) if "total_price" in df else None}
    for name in ("currency_x", "currency_y"):
        if not d[name].astype("string").str.upper().str.strip().eq("AZN").fillna(False).all():
            raise ValueError("Only verified AZN is supported. Resolve other currencies explicitly; never guess rates.")
    if d.estate_rel_url.isna().any() or d.estate_rel_url.astype(str).str.strip().eq("").any():
        raise ValueError("Canonical listing URLs are required for safe deduplication.")
    d["_scraped"] = pd.to_datetime(d.datetime_scrape_y, format="mixed", utc=True, errors="coerce")
    if d._scraped.isna().any():
        raise ValueError("Unparseable scrape dates: latest-snapshot selection is ambiguous.")
    audit["scrape_start"], audit["scrape_end"] = d._scraped.min().isoformat(), d._scraped.max().isoformat()
    audit["listings_with_price_changes"] = int((d.groupby("estate_rel_url").price.nunique() > 1).sum())
    d = d.sort_values(["_scraped", "source_row"], kind="stable").drop_duplicates("estate_rel_url", keep="last")
    audit["older_listing_snapshots_removed"] = len(df) - len(d)
    d = d.sort_values("estate_rel_url", kind="stable").reset_index(drop=True)
    price = pd.to_numeric(d.price, errors="coerce")
    valid = np.isfinite(price) & price.gt(0)
    audit["invalid_price_rows_removed"] = int((~valid).sum())
    d = d.loc[valid].copy().reset_index(drop=True)
    out = pd.DataFrame({"price":pd.to_numeric(d.price), "source_row":d.source_row})
    out["listing_id"] = d.estate_rel_url.map(_digest)
    out["area_m2"] = parse_area(d["Sahə"])
    out["land_area_m2"] = parse_area(d["Torpaq sahəsi"])
    is_land = d["Kateqoriya"].eq("Torpaq")
    out.loc[is_land,"land_area_m2"] = out.loc[is_land,"area_m2"]
    out["rooms"] = pd.to_numeric(d["Otaq sayı"], errors="coerce")
    out["rooms"] = out.rooms.where(out.rooms.gt(0) & out.rooms.mod(1).eq(0))
    floors = d["Mərtəbə"].astype("string").str.extract(r"^(\d+)\s*/\s*(\d+)$").astype(float)
    valid_floor = floors[0].ge(1) & floors[1].ge(floors[0])
    out["floor"], out["total_floors"] = floors[0].where(valid_floor), floors[1].where(valid_floor)
    out["floor_ratio"] = out.floor / out.total_floors
    out["latitude"] = pd.to_numeric(d.lat, errors="coerce")
    out["longitude"] = pd.to_numeric(d.lng, errors="coerce")
    # Broad, predeclared plausibility box, not an official country boundary.
    valid_geo = out.latitude.between(38,42.5) & out.longitude.between(44,51)
    audit["invalid_coordinate_pairs_set_missing"] = int((~valid_geo).sum())
    out.loc[~valid_geo,["latitude","longitude"]] = np.nan
    out["log_area_m2"] = np.log(out.area_m2)
    out["area_per_room"] = out.area_m2 / out.rooms
    for original,name in [("city","city"),("location","location"),("Kateqoriya","category"),
                          ("Təmir","repair"),("Çıxarış","bill_of_sale"),("İpoteka","mortgage")]:
        out[name] = _text(d[original])
    # Missing mortgage is unknown/not stated, not a made-up negative label.
    desc = _text(d.description).fillna("").str.replace(r"\s+", " ", regex=True)
    signatures = pd.DataFrame({"description":desc,"area":d["Sahə"],"rooms":d["Otaq sayı"],
                               "floor":d["Mərtəbə"],"category":d["Kateqoriya"],
                               "lat":d.lat.round(6),"lng":d.lng.round(6)})
    keys = signatures.fillna("<missing>").astype(str).apply(lambda row: json.dumps(row.tolist(),ensure_ascii=False),axis=1)
    out["group_id"] = out.listing_id.copy()
    usable = desc.str.len().ge(80)
    out.loc[usable,"group_id"] = keys.loc[usable].map(_digest)
    audit["additional_rows_in_shared_property_groups"] = int(len(out)-out.group_id.nunique())
    audit["clean_rows"], audit["split_groups"] = len(out), int(out.group_id.nunique())
    audit["area_sot_values_converted"] = int(d["Sahə"].str.contains("sot",na=False).sum())
    audit["missing_after_cleaning"] = {c:int(out[c].isna().sum()) for c in NUMERIC_FEATURES+CATEGORICAL_FEATURES}
    audit["ignored_raw_columns"] = [c for c in df if c not in needed and c not in LEAKAGE_COLUMNS]
    out.attrs["audit"] = audit
    return out


def make_tier_label(y_price, threshold=None):
    """Without threshold, call ONLY on training prices. Premium is strict >."""
    y = np.asarray(y_price,dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any(y<=0):
        raise ValueError("Prices must be a nonempty finite positive 1D array.")
    threshold = float(np.median(y)) if threshold is None else float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Tier threshold must be finite and positive.")
    return (y>threshold).astype(np.int64), threshold


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    threshold: float


def make_splits(frame,val_size=.15,test_size=.15,seed=SEED):
    """Group-disjoint, approximate 70/15/15 split, training-only median.

    Freeze randomly chosen training groups WITHOUT consulting prices. Then
    derive the actual training median and stratify the remaining groups by
    majority tier into val/test. Class counts can differ because groups stay
    intact. This is NOT a full-data-median stratified split. Never reroll a
    seed to obtain nicer scores. Report exact sizes and class counts.
    """
    if val_size<=0 or test_size<=0 or val_size+test_size>=1:
        raise ValueError("Split proportions must be positive and sum to < 1.")
    groups = np.array(sorted(frame.group_id.unique()))
    if len(groups)<10:
        raise ValueError("At least 10 distinct groups are required.")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_train = max(1,min(len(groups)-4,int(round(len(groups)*(1-val_size-test_size)))))
    in_train = frame.group_id.isin(groups[:n_train])
    train,held = frame.loc[in_train].copy(),frame.loc[~in_train].copy()
    _,threshold = make_tier_label(train.price)
    labels,_ = make_tier_label(held.price,threshold)
    group_labels = (pd.Series(labels,index=held.index).groupby(held.group_id).mean()>=.5).astype(int)
    val_groups = []
    for cls in (0,1):
        members = rng.permutation(group_labels.index[group_labels.eq(cls)].to_numpy())
        if len(members)<2:
            raise ValueError("Too few held-out groups in a tier for this protocol.")
        n_val = max(1,min(len(members)-1,int(round(len(members)*val_size/(val_size+test_size)))))
        val_groups.extend(members[:n_val])
    in_val = held.group_id.isin(val_groups)
    val,test = held.loc[in_val].copy(),held.loc[~in_val].copy()
    sets = [set(z.group_id) for z in (train,val,test)]
    if any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise AssertionError("Property groups overlap across splits.")
    return SplitData(train,val,test,threshold)


class FeatureEncoder:
    """Train medians, missing flags, train-fitted one-hot categorical values.

    Rare, unseen and missing categories get distinct buckets. IDs, text and
    target fields cannot enter X because features use a fixed allowlist.
    """
    def __init__(self,min_category_count=20):
        if min_category_count<1:
            raise ValueError("min_category_count must be >= 1.")
        self.min_category_count = int(min_category_count)
        self.fitted_ = False

    def fit(self,train):
        self.medians_,self.categories_,self.known_categories_ = {},{},{}
        for c in NUMERIC_FEATURES:
            a = train[c].to_numpy(dtype=float,na_value=np.nan).copy()
            a[~np.isfinite(a)] = np.nan
            self.medians_[c] = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.
        for c in CATEGORICAL_FEATURES:
            counts = _text(train[c]).value_counts()
            self.known_categories_[c] = sorted(str(v) for v in counts.index)
            self.categories_[c] = sorted(str(v) for v in counts[counts>=self.min_category_count].index)
        self.feature_names_ = NUMERIC_FEATURES+[c+"__missing" for c in NUMERIC_FEATURES]
        for c in CATEGORICAL_FEATURES:
            self.feature_names_ += [c+"="+v for v in self.categories_[c]+["<RARE>","<UNKNOWN>","<MISSING>"]]
        self.fitted_ = True
        x = self.transform(train)
        self.mean_,self.scale_ = x.mean(axis=0),x.std(axis=0)
        self.scale_[self.scale_<1e-12] = 1.
        return self

    def transform(self,frame,scaled=False):
        if not self.fitted_:
            raise RuntimeError("Fit the encoder on training rows first.")
        numbers = frame[NUMERIC_FEATURES].to_numpy(dtype=float,na_value=np.nan).copy()
        missing = ~np.isfinite(numbers)
        for j,c in enumerate(NUMERIC_FEATURES):
            numbers[missing[:,j],j] = self.medians_[c]
        blocks = [numbers,missing.astype(float)]
        for c in CATEGORICAL_FEATURES:
            known,common = set(self.known_categories_[c]),set(self.categories_[c])
            values = ["<MISSING>" if pd.isna(v) else v if v in common else "<RARE>" if v in known else "<UNKNOWN>" for v in _text(frame[c])]
            codes = {v:i for i,v in enumerate(self.categories_[c]+["<RARE>","<UNKNOWN>","<MISSING>"])}
            block = np.zeros((len(frame),len(codes)),dtype=float)
            block[np.arange(len(frame)),[codes[v] for v in values]] = 1
            blocks.append(block)
        x = np.column_stack(blocks)
        return (x-self.mean_)/self.scale_ if scaled else x

    def to_dict(self):
        if not self.fitted_:
            raise RuntimeError("Encoder has not been fitted.")
        return {"min_category_count":self.min_category_count,"numeric_features":NUMERIC_FEATURES,
                "categorical_features":CATEGORICAL_FEATURES,"medians":self.medians_,
                "categories":self.categories_,"known_categories":self.known_categories_,
                "feature_names":self.feature_names_,"mean":self.mean_.tolist(),"scale":self.scale_.tolist()}


def standardize(X_tr,*others):
    """Standalone train-only standardization, compatible with the starter."""
    train = np.asarray(X_tr,dtype=float)
    mean,scale = train.mean(axis=0),train.std(axis=0)
    scale[scale<1e-12] = 1.
    return tuple((np.asarray(x,dtype=float)-mean)/scale for x in (X_tr,*others))
