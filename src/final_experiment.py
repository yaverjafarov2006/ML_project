"""Validation selection -> configuration freeze -> grouped CV -> final test.

No model choices below use test outcomes. Existing split membership is retained
after restricting both tasks to residential listings. Runtime excludes I/O
unless stated; baseline pipelines include their own preprocessing.
"""
import argparse
import hashlib
import json
import platform
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .data_prep import ROOT, DATA_PATH, load_raw, clean, make_splits, FeatureEncoder
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures
from .evaluate import regression_metrics, classification_metrics
from .residential_models import ModelSpec, ResidentialRegressor, residential_rows
from .final_models import OwnClassifier, sklearn_classifier, scores


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def reg_metrics(y, p):
    y=np.asarray(y);ape=np.abs(y-p)/y
    return dict(**regression_metrics(y,p),median_ape_pct=float(100*np.median(ape)),
                within_20_pct=float(100*np.mean(ape<=.2)))


def cls_metrics(y, p, s):
    return dict(**classification_metrics(y,p,s),pr_auc_ap=float(average_precision_score(y,s)),
                accuracy=float(np.mean(y==p)))


def fit_time(model, frame, y=None):
    t=perf_counter()
    model.fit(frame) if y is None else model.fit(frame,y)
    return perf_counter()-t


def search(train,val,out):
    enc=FeatureEncoder().fit(train)
    X,V=enc.transform(train),enc.transform(val)
    threshold=float(train.price.median())
    y=(train.price.to_numpy()>threshold).astype(int)
    v=(val.price.to_numpy()>threshold).astype(int)
    rows=[]; reg_specs=[
        ModelSpec('own_plain_d12_l10',max_depth=12,min_samples_leaf=10),
        ModelSpec('own_plain_d16_l5'),
        ModelSpec('own_plain_d20_l5',max_depth=20),
        ModelSpec('own_segmented',segmented=True),
        ModelSpec('own_spatial',spatial=True),
        ModelSpec('own_spatial_segmented',spatial=True,segmented=True),
        ModelSpec('own_spatial_d20_l20',spatial=True,max_depth=20,min_samples_leaf=20)]
    reg_rows=[]
    for spec in reg_specs:
        m=ResidentialRegressor(spec);elapsed=fit_time(m,train)
        row=dict(name=spec.name,config=spec.to_dict(),fit_seconds=elapsed)
        row.update({'val_'+k:v for k,v in reg_metrics(val.price.to_numpy(),m.predict(val)).items()})
        reg_rows.append(row)
        print(f'Regression {spec.name}: validation RMSE={row["val_rmse"]:,.0f}',flush=True)
    pd.DataFrame([{**r,'config':json.dumps(r['config'])} for r in reg_rows]).to_csv(out/'final_regression_search.csv',index=False)
    best_reg=min(reg_rows,key=lambda r:r['val_rmse'])
    for criterion in ['gini','entropy']:
        for depth in [8,12,16]:
            for leaf in [5,10]:
                cfg=dict(kind='tree',criterion=criterion,max_depth=depth,min_samples_leaf=leaf)
                m=DecisionTree(criterion=criterion,max_depth=depth,min_samples_leaf=leaf,random_state=42)
                elapsed=fit_time(m,X,y);p=m.predict(V);s=m.predict_proba(V)[:,1]
                row=dict(config=cfg,fit_seconds=elapsed,train_f1=classification_metrics(y,m.predict(X),m.predict_proba(X)[:,1])['f1'])
                row.update({'val_'+k:v for k,v in cls_metrics(v,p,s).items() if k!='confusion_matrix'})
                rows.append(row)
    print('Tree classification search complete',flush=True)
    XS,VS=enc.transform(train,scaled=True),enc.transform(val,scaled=True)

    def trial(cfg,A,B):
        m=PegasosSVM(lambda_=cfg['lambda_'],n_epochs=cfg['n_epochs'],schedule=cfg.get('schedule','pegasos'))
        elapsed=fit_time(m,A,y)
        row=dict(config=cfg,fit_seconds=elapsed,objective=m.objective_,weight_norm=m.weight_norm_,
            margin_width=m.margin_width_,margin_violator_fraction=float(m.support_mask_.mean()),
            train_f1=classification_metrics(y,m.predict(A),m.decision_function(A))['f1'])
        row.update({'val_'+k:v for k,v in cls_metrics(v,m.predict(B),m.decision_function(B)).items() if k!='confusion_matrix'})
        rows.append(row)
        print(f'SVM {cfg}: validation F1={row["val_f1"]:.4f}',flush=True)
        return row

    for lam in [1e-4,1e-3,1e-2]:
        trial(dict(kind='linear',lambda_=lam,n_epochs=60,schedule='pegasos'),XS,VS)
    for gamma in [.001,.005,.02]:
        mapper=RandomFourierFeatures(gamma,1024);A=mapper.fit_transform(XS);B=mapper.transform(VS)
        for lam in [1e-5,1e-4,1e-3]:
            trial(dict(kind='rff',gamma=gamma,n_components=1024,lambda_=lam,n_epochs=60,schedule='pegasos'),A,B)
    best_rff=max([r for r in rows if r['config']['kind']=='rff'],key=lambda r:r['val_f1'])['config']
    # Bounded follow-up at the best validation bandwidth/regularization only.
    mapper=RandomFourierFeatures(best_rff['gamma'],2048);A=mapper.fit_transform(XS);B=mapper.transform(VS)
    trial({**best_rff,'n_components':2048},A,B)
    trial({**best_rff,'n_components':2048,'n_epochs':120},A,B)
    # Epoch and schedule diagnostics do not change the main Pegasos protocol.
    best_linear=max([r for r in rows if r['config']['kind']=='linear'],key=lambda r:r['val_f1'])['config']
    trial({**best_linear,'n_epochs':20},XS,VS)
    trial({**best_linear,'n_epochs':120},XS,VS)
    for schedule in ['constant','inverse_sqrt']:
        trial({**best_linear,'schedule':schedule},XS,VS)
    pd.DataFrame([{**r,'config':json.dumps(r['config'])} for r in rows]).to_csv(out/'final_classification_search.csv',index=False)
    selected={kind:max([r for r in rows if r['config']['kind']==kind and
               r['config'].get('schedule','pegasos')=='pegasos'],key=lambda r:r['val_f1'])['config']
              for kind in ['tree','linear','rff']}
    winner=max(selected,key=lambda k:max(r['val_f1'] for r in rows if r['config']==selected[k]))
    return dict(regression=best_reg['config'],classification=selected,
                classification_winner=winner,threshold=threshold,
                selection='Validation RMSE for regression; positive-class F1 for classification. No test selection.',
                feature_scope='Residential: new apartments, old apartments, detached/garden houses.',
                final_fit='Original residential training subset only; validation is not refitted.',
                threshold_rule='price > median(original residential training prices)',
                svm_diagnostics_only_schedules=['constant','inverse_sqrt'])


def model_list(config):
    reg=dict(config['regression'])
    return [
        ('own_tree_regression','regression',lambda:ResidentialRegressor(ModelSpec(**reg))),
        ('sklearn_tree_regression','regression',lambda:ResidentialRegressor(ModelSpec(**{**reg,'kind':'sklearn_tree'}))),
        *[(f'{origin}_{kind}','classification',
           (lambda c=c:OwnClassifier(c)) if origin=='own' else (lambda c=c:sklearn_classifier(c)))
          for kind,c in config['classification'].items() for origin in ['own','sklearn']]]


def cv_evaluate(train,config,out):
    rows=[]; threshold=config['threshold']
    cv=GroupKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(it,iv) in enumerate(cv.split(train,groups=train.group_id),1):
        a,b=train.iloc[it],train.iloc[iv]
        ya=(a.price.to_numpy()>threshold).astype(int);yb=(b.price.to_numpy()>threshold).astype(int)
        for name,task,constructor in model_list(config):
            m=constructor();elapsed=fit_time(m,a,ya if task=='classification' else None)
            p=m.predict(b)
            met=reg_metrics(b.price.to_numpy(),p) if task=='regression' else cls_metrics(yb,p,scores(m,b,name.split('_')[-1]))
            rows.append(dict(model=name,task=task,fold=fold,n_train=len(a),n_eval=len(b),
                max_eval_price=float(b.price.max()),fit_seconds=elapsed,**{k:v for k,v in met.items() if k!='confusion_matrix'}))
            print(f'CV fold {fold}/5 {name}',flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/'final_cv_folds.csv',index=False)
    numeric=['rmse','mae','r2','median_ape_pct','within_20_pct','f1','roc_auc','pr_auc_ap','fit_seconds']
    summary=table.groupby(['model','task'])[numeric].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'final_cv_summary.csv',index=False)


def fit_and_test(train,val,test,config,out):
    threshold=config['threshold']; rows=[]; predictions={};models={}
    y=(train.price.to_numpy()>threshold).astype(int)
    hist=[];diagnostics=[];importance=[];weights=[]
    support=[];classification_errors=[]
    for name,task,constructor in model_list(config):
        m=constructor();elapsed=fit_time(m,train,y if task=='classification' else None)
        models[name]=m
        for subset,frame in [('validation',val),('test',test)]:
            started=perf_counter();p=m.predict(frame)
            score=scores(m,frame,name.split('_')[-1]) if task=='classification' else p
            predict_seconds=perf_counter()-started
            truth=frame.price.to_numpy() if task=='regression' else (frame.price.to_numpy()>threshold).astype(int)
            metrics=reg_metrics(truth,p) if task=='regression' else cls_metrics(truth,p,score)
            rows.append(dict(model=name,task=task,subset=subset,n=len(frame),fit_seconds=elapsed,
                             predict_seconds=predict_seconds,**metrics))
            predictions[f'{subset}_{name}_prediction']=p
            predictions[f'{subset}_{name}_score']=score
        if name in ['own_linear','own_rff']:
            hist += [dict(model=name,**h) for h in m.model_.history_]
            diagnostics.append(dict(model=name,objective=m.model_.objective_,weight_norm=m.model_.weight_norm_,
                margin_width=m.model_.margin_width_,margin_violator_fraction=float(m.model_.support_mask_.mean()),
                margin_violator_count=int(m.model_.support_mask_.sum()),n_steps=m.model_.n_steps_))
            train_scores=m.decision_function(train)
            for i in np.flatnonzero(m.model_.support_mask_):
                support.append(dict(model=name,train_position=int(i),signed_margin=float((2*y[i]-1)*train_scores[i])))
        if task=='classification' and name.startswith('own_'):
            p=predictions[f'test_{name}_prediction'];s=predictions[f'test_{name}_score']
            truth=(test.price.to_numpy()>threshold).astype(int)
            bands=np.where(np.abs(test.price.to_numpy()-threshold)<=threshold*.1,'within_10pct_of_threshold','farther_from_threshold')
            for group_type,values in [('category',test.category.to_numpy()),('threshold_distance',bands)]:
                for value in np.unique(values):
                    ix=values==value;yy=truth[ix]
                    met=cls_metrics(yy,p[ix],s[ix]) if len(np.unique(yy))==2 else {}
                    classification_errors.append(dict(model=name,group_type=group_type,group=value,n=int(ix.sum()),
                        error_rate=float(np.mean(yy!=p[ix])),**{k:v for k,v in met.items() if k!='confusion_matrix'}))
        if name=='own_linear':
            weights=[dict(feature=f,coefficient=float(v)) for f,v in zip(m.encoder_.feature_names_,m.model_.w)]
        if name=='own_tree':
            importance += [dict(model=name,route='all',feature=f,importance=float(v))
                for f,v in zip(m.encoder_.feature_names_,m.model_.feature_importances_)]
        if name=='own_tree_regression':
            for route,tree in m.models_.items():
                importance += [dict(model=name,route=route,feature=f,importance=float(v))
                    for f,v in zip(m.encoders_[route].feature_names_,tree.feature_importances_)]
        print(f'Final fit and fixed evaluation: {name}',flush=True)
    # Naive baselines are training-only medians/majority, never tuning candidates.
    for subset,frame in [('validation',val),('test',test)]:
        truth=frame.price.to_numpy();c=(truth>threshold).astype(int)
        rows.append(dict(model='training_median',task='regression',subset=subset,n=len(frame),
                         **reg_metrics(truth,np.full(len(frame),float(train.price.median())))))
        maj=int(y.mean()>.5);p=np.full(len(frame),maj);s=np.full(len(frame),y.mean())
        rows.append(dict(model='training_majority',task='classification',subset=subset,n=len(frame),**cls_metrics(c,p,s)))
    dump(out/'final_metrics.json',rows)
    pd.DataFrame([{**r,'confusion_matrix':json.dumps(r.get('confusion_matrix'))} for r in rows]).to_csv(out/'final_metrics.csv',index=False)
    pd.DataFrame(hist).to_csv(out/'svm_objective_history.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'svm_diagnostics.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'final_feature_importance.csv',index=False)
    pd.DataFrame(weights).to_csv(out/'linear_svm_weights.csv',index=False)
    pd.DataFrame(support).to_csv(out/'svm_margin_violators.csv',index=False)
    pd.DataFrame(classification_errors).to_csv(out/'final_classification_error_groups.csv',index=False)
    for subset,frame in [('validation',val),('test',test)]:
        predictions[f'{subset}_price']=frame.price.to_numpy(float)
        predictions[f'{subset}_category']=frame.category.to_numpy(str)
    np.savez_compressed(out/'final_predictions.npz',**predictions)
    # Only our model classes are persisted, no local sklearn adapter pickles.
    modeldir=ROOT/'models';modeldir.mkdir(exist_ok=True)
    with (modeldir/'final_models.pkl').open('wb') as f:
        pickle.dump(dict(config=config,models={k:v for k,v in models.items() if k.startswith('own_')}),f)
    error_rows=[]
    for category,part in test.groupby('category'):
        ix=test.category.eq(category).to_numpy()
        p=predictions['test_own_tree_regression_prediction'][ix]
        error_rows.append(dict(category=category,n=len(part),**reg_metrics(part.price.to_numpy(),p)))
    pd.DataFrame(error_rows).to_csv(out/'final_regression_error_by_category.csv',index=False)
    ytest=test.price.to_numpy();ptest=predictions['test_own_tree_regression_prediction']
    e=(ytest-ptest)**2;k=max(1,int(np.ceil(len(e)*.01)))
    dump(out/'final_error_analysis.json',dict(top_one_percent_sse_share=float(np.sort(e)[-k:].sum()/e.sum()),
        n_top=k,test_max_price=float(ytest.max()),train_max_price=float(train.price.max()),
        rows_removed_from_evaluation=0,limitations=['Asking prices, not verified transaction prices.',
        'No independent manual validation of extreme price labels.',
        'Random grouped split does not establish future-period performance.',
        'CV is a fixed-configuration stability check, not nested model selection.']))
    from .final_eda import build
    build(train,test,ptest)


def scaling_experiment(train,config,out):
    rows=[];rng=np.random.default_rng(42);order=rng.permutation(len(train))
    for n in [2000,8000,20000]:
        part=train.iloc[order[:n]];y=(part.price.to_numpy()>config['threshold']).astype(int)
        for kind in ['tree','linear','rff']:
            m=OwnClassifier(config['classification'][kind]);t=fit_time(m,part,y)
            rows.append(dict(model='own_'+kind,n_train=n,fit_seconds=t,
                input_features=len(m.encoder_.feature_names_),
                mapped_features=config['classification'][kind].get('n_components',len(m.encoder_.feature_names_))))
    pd.DataFrame(rows).to_csv(out/'runtime_scaling.csv',index=False)


def run(data_path=DATA_PATH,stage='all'):
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    start=perf_counter()
    raw=load_raw(data_path);frame=clean(raw);split=make_splits(frame)
    dump(out/'data_quality.json',frame.attrs['audit'])
    train,val,test=map(residential_rows,[split.train,split.val,split.test])
    counts={name:dict(n=len(part),groups=int(part.group_id.nunique()),
             premium=int((part.price>train.price.median()).sum()))
            for name,part in [('train',train),('validation',val),('test',test)]}
    digest=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    dump(out/'final_data_summary.json',dict(raw_rows=len(raw),clean_rows=len(frame),
         residential_rows=len(train)+len(val)+len(test),split=counts,threshold=float(train.price.median()),
         original_full_scope_threshold=float(split.threshold),dataset_sha256=digest,seed=42,
         categories=list(train.category.unique()),feature_count=len(FeatureEncoder().fit(train).feature_names_),
         notes='Original full-dataset group split retained. Final residential training median defines the final classification task.'))
    with threadpool_limits(limits=2):
        if stage in ['search','all']:
            config=search(train,val,out)
            config.update(frozen_at=datetime.now(timezone.utc).isoformat(),dataset_sha256=digest)
            payload=json.dumps(config,sort_keys=True,ensure_ascii=False).encode()
            config['configuration_sha256']=hashlib.sha256(payload).hexdigest()
            dump(out/'frozen_config.json',config)
            print('CONFIGURATION FROZEN. No test outcomes used.',flush=True)
        else:
            config=json.loads((out/'frozen_config.json').read_text(encoding='utf-8'))
            if config['dataset_sha256']!=digest:raise ValueError('Dataset differs from frozen experiment')
        if stage in ['cv','all']:cv_evaluate(train,config,out)
        if stage in ['test','all']:
            fit_and_test(train,val,test,config,out)
            scaling_experiment(train,config,out)
            from .final_figures import build
            build()
    dump(out/f'run_{stage}_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,platform=platform.platform(),
        elapsed_seconds=perf_counter()-start,threads=2,completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(DATA_PATH))
    p.add_argument('--stage',choices=['search','cv','test','all'],default='all');a=p.parse_args()
    run(a.data,a.stage)
