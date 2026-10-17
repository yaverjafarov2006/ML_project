"""Run stage 1: python -m src.prepare_data [--data PATH] [--save-arrays]."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data_prep import (DATA_PATH, ROOT, SEED, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                        COLUMN_MAP, load_raw, clean, make_splits, make_tier_label, FeatureEncoder)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def fingerprint(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            sha.update(chunk)
    return sha.hexdigest()


def figures(train,split_summary,directory):
    """Exploratory figures use training rows only; no validation/test prices."""
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.titleweight":"bold","axes.labelcolor":"#334155",
                         "text.color":"#172c46","figure.facecolor":"white"})
    blue,teal = "#2768b6","#168c86"
    fig,ax = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    price = train.price.to_numpy()
    cap = np.quantile(price,.99)
    ax[0,0].hist(price[price<=cap]/1000,bins=55,color=blue)
    ax[0,0].set(title="Price distribution",xlabel="Asking price (thousand AZN)",ylabel="Training listings")
    ax[0,0].text(.98,.96,f"Display up to train P99\n{int((price>cap).sum()):,} higher values retained",ha="right",va="top",transform=ax[0,0].transAxes,fontsize=9)
    ax[0,1].hist(np.log10(price),bins=65,color=teal)
    ax[0,1].set(title="Full price range on a log scale",xlabel="log10(asking price in AZN)",ylabel="Training listings")
    valid = train.area_m2.notna() & train.area_m2.gt(0)
    hb = ax[1,0].hexbin(train.loc[valid,"area_m2"],train.loc[valid,"price"],
                        xscale="log",yscale="log",gridsize=40,mincnt=1,bins="log",cmap="Blues")
    ax[1,0].set(title="Area and price: all property types",xlabel="Area (m², log scale)",ylabel="Price (AZN, log scale)")
    fig.colorbar(hb,ax=ax[1,0],label="Listings per bin")
    counts = train.category.value_counts().sort_values()
    ax[1,1].barh(counts.index,counts.values,color=blue)
    ax[1,1].set(title="Property categories",xlabel="Training listings")
    fig.suptitle(f"Bina.az | Training data overview | n = {len(train):,}",fontsize=16)
    fig.savefig(directory/"01_training_overview.png",dpi=160)
    plt.close(fig)

    fig,ax = plt.subplots(1,3,figsize=(16,5),layout="constrained")
    rooms = train.groupby("rooms").price.agg(["median","size"])
    rooms = rooms[(rooms["size"]>=30)&(rooms.index<=10)]
    ax[0].bar(rooms.index.astype(int).astype(str),rooms["median"]/1000,color=blue)
    ax[0].set(title="Rooms and median asking price",xlabel="Rooms (groups with n ≥ 30)",ylabel="Median price (thousand AZN)")
    locations = train.groupby("location").price.agg(["median","size"]).nlargest(8,"size").sort_values("median")
    ax[1].barh(locations.index,locations["median"]/1000,color=teal)
    ax[1].set(title="Eight most frequent locations",xlabel="Median price (thousand AZN)")
    baku = train[train.latitude.between(40.2,40.65)&train.longitude.between(49.6,50.35)]
    hb = ax[2].hexbin(baku.longitude,baku.latitude,C=np.log10(baku.price),
                        reduce_C_function=np.median,mincnt=10,gridsize=28,cmap="viridis")
    ax[2].set(title="Baku-area spatial price pattern",xlabel="Longitude",ylabel="Latitude")
    fig.colorbar(hb,ax=ax[2],label="Median log10(price), bins n ≥ 10")
    fig.suptitle("Training-only associations; category/area differences can confound these patterns",fontsize=13)
    fig.savefig(directory/"02_location_and_rooms.png",dpi=160)
    plt.close(fig)

    fig,ax = plt.subplots(1,2,figsize=(12,6),layout="constrained")
    missing = train[NUMERIC_FEATURES+CATEGORICAL_FEATURES].isna().mean().sort_values()*100
    ax[0].barh(missing.index,missing.values,color=blue)
    ax[0].set(title="Missing values before train-fitted imputation",xlabel="Missing (%)",xlim=(0,100))
    names = list(split_summary)
    standard = np.array([split_summary[k]["standard"] for k in names])
    premium = np.array([split_summary[k]["premium"] for k in names])
    ax[1].bar(names,standard,label="Standard",color=blue)
    ax[1].bar(names,premium,bottom=standard,label="Premium",color=teal)
    ax[1].set(title="Split sizes and required class-balance check",ylabel="Listings")
    ax[1].legend(frameon=False)
    for i,k in enumerate(names):
        ax[1].text(i,standard[i]+premium[i],f"{split_summary[k]['rows']:,}\n{split_summary[k]['premium_pct']:.1f}% premium",ha="center",va="bottom",fontsize=10)
    ax[1].set_ylim(0,max(standard+premium)*1.18)
    fig.savefig(directory/"03_missingness_and_splits.png",dpi=160)
    plt.close(fig)


def make_report(audit,summary,encoder,train,source_hash):
    splits = "\n".join(f"| {k} | {s['rows']:,} | {s['groups']:,} | {s['standard']:,} | {s['premium']:,} | {s['premium_pct']:.2f}% |" for k,s in summary["splits"].items())
    mapping = "\n".join(f"| `{k}` | `{v}` |" for k,v in COLUMN_MAP.items())
    missing = "\n".join(f"| {c} | {int(train[c].isna().sum()):,} | {train[c].isna().mean()*100:.2f}% |" for c in NUMERIC_FEATURES+CATEGORICAL_FEATURES)
    return f"""# Stage 1: data audit and experimental protocol

Status: data preparation complete. No model has been trained or scored. This is not the final course report.

## Source and provenance
- Source: https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project
- Input: `house_sale.csv`, supplied by the user on 2026-10-15.
- SHA-256: `{source_hash}`.
- Detail scrape timestamps: {audit['scrape_start']} through {audit['scrape_end']}.
- Snapshot concerns 2024 asking prices, not transaction prices or current 2026 valuations.
- Python {platform.python_version()}; NumPy {np.__version__}; pandas {pd.__version__}; matplotlib {matplotlib.__version__}; seed {SEED}.

## Row accounting
| Check | Count |
|---|---:|
| Raw rows | {audit['raw_rows']:,} |
| Raw columns | {audit['raw_columns']} |
| Exact duplicate rows | {audit['exact_duplicate_rows']:,} |
| Older snapshots of a listing removed | {audit['older_listing_snapshots_removed']:,} |
| Invalid/nonpositive target rows removed | {audit['invalid_price_rows_removed']:,} |
| Remaining listings | {audit['clean_rows']:,} |
| Additional rows sharing a conservative property signature | {audit['additional_rows_in_shared_property_groups']:,} |
| Distinct split groups | {audit['split_groups']:,} |
| Listings with price changes between raw snapshots | {audit['listings_with_price_changes']:,} |
| Raw price versus total_price disagreements | {audit['price_total_price_mismatches']:,} |
| Retained Sahə values converted from sot | {audit['area_sot_values_converted']:,} |
| Implausible coordinate pairs set missing | {audit['invalid_coordinate_pairs_set_missing']:,} |

Full-row drop_duplicates() removes nothing because snapshot identifiers vary. The canonical listing URL is the deduplication key. Keep the latest detail scrape; source-row order breaks timestamp ties. The regression target remains `price`, as required; disagreements with `total_price` are recorded, not silently repaired.

Possible cross-listing copies are grouped when their normalized description has at least 80 characters and their area string, rooms, floor string, category, latitude and longitude rounded to six decimals all match. Price is not in the key. These rows are retained but never cross splits. This proxy may miss reworded/rephotographed listings and may group unrelated listings with identical supplied details; it is not a verified property identifier.

## Cleaning decisions
1. Verify both currency columns are AZN. An unknown currency stops the pipeline; no guessed conversion rate.
2. Convert explicit m²/m2/sot units, with 1 sot = 100 m². `area_m2` is listing area, so its interpretation depends on property category. For land, also populate `land_area_m2` from area. Unknown units and nonpositive areas become missing.
3. Parse `Mərtəbə` into current floor and total floors; create floor_ratio. Invalid pairs become missing. Do not assume houses or land have a recorded apartment floor.
4. Use the explicit Təmir and Çıxarış fields. Missing İpoteka is not stated, not a verified no.
5. Coordinates outside the broad, predeclared box latitude [38,42.5], longitude [44,51] become missing. The box is a plausibility screen, not an official boundary.
6. Keep every finite positive target, including extremely low/high values. Training log-price IQR fences flag {summary['train_price_outliers']['flagged_rows']:,} training rows for review; no row is removed by those fences. The plot-only P99 limit does not filter model data. Plan a documented price-vs-log-price modelling comparison and tail-error analysis.

## Features and leakage controls
| Source | Model field |
|---|---|
{mapping}

Additional numerical features: floor_ratio, log_area_m2 and area_per_room. City, location and category remain categorical; no arbitrary ordinal district codes. There are {len(NUMERIC_FEATURES)} numeric and {len(CATEGORICAL_FEATURES)} categorical input fields, producing {len(encoder.feature_names_)} encoded columns including missing flags.

`unit_price` and `total_price` are excluded because they encode the target. `price` is y only, also excluded from X for the tier task. All raw identifiers, URLs, owner/shop details, address, description, attributes, extra_info, views, promotion flags and scrape dates are excluded from model inputs. Description is used only transiently to form duplicate groups. The nearly empty Binanın növü field is excluded; the populated Kateqoriya field supplies the property type. No seller names, addresses or raw descriptions are included in public outputs.

Every numeric feature is filled with its TRAIN median (all-missing fallback 0), accompanied by a missing flag. Categorical vocabularies and the rare-category rule (<{encoder.min_category_count} TRAIN rows) are learned on TRAIN only. Missing, known rare and unseen values have separate one-hot buckets. SVM means/standard deviations also come from TRAIN only; constant columns use scale 1. Trees use unscaled values. Refit this entire encoder inside each future CV training fold.

## Split and label protocol
Training groups are selected first, independently of prices, and then frozen. The actual TRAIN median is **{summary['tier_threshold_azn']:,.0f} AZN**. Standard means price <= threshold; premium means price > threshold. Ties stay standard, so a median split need not be exactly 50/50.

Only then are the remaining groups stratified by their majority tier and divided between validation and test. Groups stay intact. This two-stage protocol is approximately 70/15/15; it avoids using a full-dataset median to define the task. It is not a conventional single-call fully stratified split. Row-level EDA figures use TRAIN only, except the required aggregate split/class-balance counts. No test metric or model selection has occurred.

| Split | Rows | Groups | Standard | Premium | Premium % |
|---|---:|---:|---:|---:|---:|
{splits}

Group overlap across splits: 0. Training median/encoder are never recomputed from validation/test rows. The regression and classification tasks use the same row allocation.

## Training missingness
| Field | Missing | Missing % |
|---|---:|---:|
{missing}

## Exploratory figures
![Training distributions](../report/figures/01_training_overview.png)

![Location and rooms](../report/figures/02_location_and_rooms.png)

![Missingness and split balance](../report/figures/03_missingness_and_splits.png)

## Limits and next stage
- Strong geographic concentration and mixed property types limit generalisation. This is a snapshot benchmark; it is not a temporal or spatial generalisation test.
- Positive price anomalies are unresolved labels. Retaining them is honest, but RMSE may be dominated by a few records. Do not change the held-out set after seeing scores.
- Nonrandom missingness and category/location imbalance should be discussed in the final analysis.
- Implement and test the NumPy decision tree next (Gini, entropy, MSE; stopping rules; predict_proba; impurity-based importance). Then implement linear Pegasos and the required nonlinear extension.
- Tune only on validation data; freeze configurations before final test evaluation. Report CV uncertainty separately from the single held-out test result.
- AI assistance used for this draft pipeline, tests, code explanations and audit. Team members must review it and report their actual work accurately. No team names or contributions have been invented.
"""


def run(data_path=DATA_PATH,output_dir=None,save_arrays=False):
    outdir = Path(output_dir) if output_dir else ROOT/"outputs"
    outdir.mkdir(parents=True,exist_ok=True)
    raw = load_raw(data_path)
    frame = clean(raw)
    audit = frame.attrs["audit"]
    split = make_splits(frame)
    encoder = FeatureEncoder().fit(split.train)
    arrays,counts,manifest = {},{},[]
    for name,part in [("train",split.train),("val",split.val),("test",split.test)]:
        y_tier,_ = make_tier_label(part.price,split.threshold)
        x = encoder.transform(part)
        if not np.isfinite(x).all():
            raise AssertionError(f"Non-finite features in {name}.")
        counts[name] = {"rows":len(part),"groups":int(part.group_id.nunique()),
                       "standard":int((y_tier==0).sum()),"premium":int(y_tier.sum()),
                       "premium_pct":float(y_tier.mean()*100)}
        if save_arrays:
            arrays.update({f"X_{name}":x,f"X_{name}_scaled":encoder.transform(part,scaled=True),
                           f"y_{name}":part.price.to_numpy(),f"y_{name}_tier":y_tier})
        manifest.append(part[["source_row","listing_id","group_id"]].assign(split=name))
    log_price = np.log(split.train.price.to_numpy())
    q1,q3 = np.quantile(log_price,[.25,.75]); iqr = q3-q1
    lo,hi = float(q1-1.5*iqr),float(q3+1.5*iqr)
    source_hash = fingerprint(data_path)
    summary = {"status":"stage_1_only_no_models_trained","seed":SEED,"source_sha256":source_hash,
               "tier_threshold_azn":split.threshold,"encoded_features":len(encoder.feature_names_),
               "splits":counts,"train_price_outliers":{"method":"1.5 IQR fences on TRAIN log(price); flag only",
                   "log_lower":lo,"log_upper":hi,"flagged_rows":int(((log_price<lo)|(log_price>hi)).sum())}}
    write_json(outdir/"data_quality.json",audit)
    write_json(outdir/"stage1_summary.json",summary)
    write_json(outdir/"preprocessor.json",encoder.to_dict())
    pd.concat(manifest).sort_values("source_row").to_csv(outdir/"split_manifest.csv",index=False)
    if save_arrays:
        np.savez_compressed(outdir/"prepared_data.npz",**arrays)
    figures(split.train,counts,ROOT/"report"/"figures")
    (outdir/"DATA_AUDIT.md").write_text(make_report(audit,summary,encoder,split.train,source_hash),encoding="utf-8")
    print(f"Stage 1 complete | {audit['raw_rows']:,} raw -> {len(frame):,} latest listings")
    print(f"Train-only tier threshold: {split.threshold:,.0f} AZN | features: {len(encoder.feature_names_)}")
    for name,values in counts.items():
        print(f"{name:5s}: {values['rows']:6,d} rows | {values['premium_pct']:.2f}% premium")
    print("No models trained or test metrics reported. Read outputs/DATA_AUDIT.md.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--save-arrays",action="store_true",help="Write local NumPy arrays; do not publish these.")
    args = parser.parse_args()
    run(args.data,save_arrays=args.save_arrays)


if __name__ == "__main__":
    main()
