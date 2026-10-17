"""Residential training EDA and descriptive geospatial error analysis."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_prep import ROOT, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .residential_models import geographic_centres, haversine_km


def build(train, test=None, predictions=None):
    out=ROOT/'outputs';figdir=ROOT/'report'/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    axes[0,0].hist(np.log10(train.price),bins=60,color='#2563a6');axes[0,0].set(xlabel='log10 asking price (AZN)',ylabel='Training rows',title='Price distribution: all training rows')
    valid=train.area_m2.gt(0)
    axes[0,1].scatter(train.loc[valid,'area_m2'],train.loc[valid,'price'],s=2,alpha=.1,color='#167d72',rasterized=True)
    axes[0,1].set(xscale='log',yscale='log',xlabel='Area (m²)',ylabel='Price (AZN)',title='Area and price; no price trimming')
    rooms=train.groupby('rooms').price.agg(['median','size']);rooms=rooms.loc[(rooms['size']>=20)&(rooms.index<=10)]
    axes[1,0].bar(rooms.index,rooms['median']/1000,color='#2563a6');axes[1,0].set(xlabel='Rooms (groups with at least 20 rows)',ylabel='Median price (thousand AZN)',title='Room-count association')
    balance=(train.price>train.price.median()).value_counts().sort_index()
    axes[1,1].bar(['Standard','Premium'],balance,color=['#94a6b7','#167d72']);axes[1,1].set(ylabel='Training rows',title='Final task: premium > 207,000 AZN')
    fig.savefig(figdir/'final_eda.png',dpi=170,bbox_inches='tight');fig.savefig(figdir/'final_eda.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,2.5),layout='constrained')
    ax.hist(np.log10(train.price),bins=60,color='#2563a6');ax.set(xlabel='log10 asking price (AZN)',ylabel='Training rows')
    fig.savefig(figdir/'report_eda.pdf',bbox_inches='tight');plt.close(fig)
    train[NUMERIC_FEATURES+CATEGORICAL_FEATURES].isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(out/'final_training_missingness.csv')
    centres=geographic_centres(train[['latitude','longitude']].to_numpy(float))
    distances=haversine_km(train[['latitude','longitude']].to_numpy(float),centres)
    valid=np.isfinite(distances).all(axis=1);labels=np.full(len(train),-1);labels[valid]=np.argmin(distances[valid],axis=1)
    rows=[]
    if test is not None:
        td=haversine_km(test[['latitude','longitude']].to_numpy(float),centres)
        tv=np.isfinite(td).all(axis=1);tl=np.full(len(test),-1);tl[tv]=np.argmin(td[tv],axis=1)
        for k in [-1,*range(len(centres))]:
            tr=labels==k;te=tl==k
            row=dict(cluster=k,train_n=int(tr.sum()),test_n=int(te.sum()),
                train_median_price=float(np.median(train.price.to_numpy()[tr])) if tr.any() else None)
            if te.any():
                y=test.price.to_numpy()[te];p=predictions[te]
                row.update(test_median_ape_pct=float(np.median(np.abs(y-p)/y)*100),test_mae=float(np.mean(np.abs(y-p))))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/'final_geographic_error.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    # Display the Baku area while explicitly reporting that model/metrics keep all locations.
    show=valid&train.latitude.between(40.1,40.7).to_numpy()&train.longitude.between(49.3,50.3).to_numpy()
    ax.scatter(train.longitude.to_numpy()[show],train.latitude.to_numpy()[show],c=labels[show],cmap='tab20',s=2,alpha=.3,rasterized=True)
    ax.scatter(centres[:,1],centres[:,0],c='black',s=20,marker='x')
    ax.set(xlim=(49.3,50.3),ylim=(40.1,40.7),xlabel='Longitude',ylabel='Latitude',title='Train-only geographic clusters: Baku view')
    fig.savefig(figdir/'final_geography.png',dpi=170,bbox_inches='tight');fig.savefig(figdir/'final_geography.pdf',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(centres,columns=['latitude','longitude']).to_csv(out/'final_geographic_centres.csv',index_label='cluster')
