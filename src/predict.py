"""Inference with the saved, validation-selected own models."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .data_prep import ROOT, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .residential_models import RESIDENTIAL_CATEGORIES


def listing_frame(values):
    row={k:values.get(k,np.nan) for k in NUMERIC_FEATURES+CATEGORICAL_FEATURES}
    row['category']=str(row['category']).strip().lower()
    if row['category'] not in RESIDENTIAL_CATEGORIES:
        raise ValueError('Supported categories: '+', '.join(RESIDENTIAL_CATEGORIES))
    for k in NUMERIC_FEATURES:
        try:row[k]=float(row[k])
        except (TypeError,ValueError):row[k]=np.nan
    if np.isfinite(row['area_m2']) and row['area_m2']<=0:raise ValueError('area_m2 must be positive')
    if np.isfinite(row['rooms']) and row['rooms']<=0:raise ValueError('rooms must be positive')
    row['log_area_m2']=np.log(row['area_m2']) if row['area_m2']>0 else np.nan
    row['area_per_room']=row['area_m2']/row['rooms'] if row['rooms']>0 else np.nan
    row['floor_ratio']=row['floor']/row['total_floors'] if row['total_floors']>0 else np.nan
    lat,lon=row['latitude'],row['longitude']
    if not (38<=lat<=42.5 and 44<=lon<=51):row['latitude']=row['longitude']=np.nan
    return pd.DataFrame([row])


def load_models(path=ROOT/'models'/'final_models.pkl'):
    # Only open the supplied project model or a model you trained yourself.
    with Path(path).open('rb') as f:return pickle.load(f)


def predict_listing(values,bundle=None):
    bundle=load_models() if bundle is None else bundle;frame=listing_frame(values)
    model=bundle['models'];kind=bundle['config']['classification_winner']
    label=int(model['own_'+kind].predict(frame)[0])
    return dict(predicted_price_azn=round(float(model['own_tree_regression'].predict(frame)[0]),2),
                premium_class='premium' if label else 'standard',
                premium_threshold_azn=bundle['config']['threshold'],classifier='own_'+kind)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('listing_json');a=p.parse_args()
    print(json.dumps(predict_listing(json.loads(Path(a.listing_json).read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))
