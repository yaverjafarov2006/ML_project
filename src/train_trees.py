"""Train-only preprocessing, validation tuning; no test predictions in Stage 2."""
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from .data_prep import ROOT,DATA_PATH,load_raw,clean,make_splits,FeatureEncoder,make_tier_label
from .decision_tree import DecisionTree
from .evaluate import regression_metrics,classification_metrics

DEPTHS=[3,6,9,12]
LEAF_SIZES=[10,50]

def grouped_importance(names,values):
    table=pd.DataFrame({'feature':[n.split('=')[0].removesuffix('__missing') for n in names],'importance':values})
    return table.groupby('feature').importance.sum().sort_values(ascending=False)

def error_concentration(y,p):
    e=(np.asarray(y)-p)**2; k=max(1,int(np.ceil(len(e)*.01)))
    return dict(n_top=k,top_one_percent_sse_share=float(np.sort(e)[-k:].sum()/e.sum()) if e.sum() else 0.,max_price=float(np.max(y)))

def _fit(model,X,y,Xv):
    start=perf_counter();model.fit(X,y);fit=perf_counter()-start
    start=perf_counter();p=model.predict(Xv);pred=perf_counter()-start
    return p,{'fit_seconds':fit,'predict_seconds':pred}

def run(data_path=DATA_PATH,output_dir=None):
    out=Path(output_dir or ROOT/'outputs');out.mkdir(parents=True,exist_ok=True)
    figdir=ROOT/'report'/'figures';figdir.mkdir(parents=True,exist_ok=True)
    split=make_splits(clean(load_raw(data_path)))
    enc=FeatureEncoder().fit(split.train)
    X,Xv=enc.transform(split.train),enc.transform(split.val)
    y,yv=split.train.price.to_numpy(float),split.val.price.to_numpy(float)
    c,_=make_tier_label(y,split.threshold);cv,_=make_tier_label(yv,split.threshold)
    rows=[]
    for target in ['price','log_price']:
        yt=np.log(y) if target=='log_price' else y
        for leaf in LEAF_SIZES:
            for depth in DEPTHS:
                model=DecisionTree(task='regression',max_depth=depth,min_samples_leaf=leaf)
                pv,times=_fit(model,X,yt,Xv);pt=model.predict(X)
                if target=='log_price': pv,pt=np.exp(pv),np.exp(pt)
                row=dict(task='regression',variant=target,max_depth=depth,min_samples_leaf=leaf,n_leaves=model.n_leaves_,**times)
                for prefix,a,p in [('train',y,pt),('val',yv,pv)]:
                    row.update({prefix+'_'+k:v for k,v in regression_metrics(a,p).items()})
                rows.append(row);print(f'{len(rows):02}/32 {target} depth={depth} leaf={leaf} val RMSE={row["val_rmse"]:.0f}',flush=True)
    for criterion in ['gini','entropy']:
        for leaf in LEAF_SIZES:
            for depth in DEPTHS:
                model=DecisionTree(criterion=criterion,max_depth=depth,min_samples_leaf=leaf)
                pv,times=_fit(model,X,c,Xv)
                row=dict(task='classification',variant=criterion,max_depth=depth,min_samples_leaf=leaf,n_leaves=model.n_leaves_,**times)
                for prefix,a,p,s in [('train',c,model.predict(X),model.predict_proba(X)[:,1]),('val',cv,pv,model.predict_proba(Xv)[:,1])]:
                    row.update({prefix+'_'+k:v for k,v in classification_metrics(a,p,s).items() if k!='confusion_matrix'})
                rows.append(row);print(f'{len(rows):02}/32 {criterion} depth={depth} leaf={leaf} val F1={row["val_f1"]:.4f}',flush=True)
    grid=pd.DataFrame(rows);grid.to_csv(out/'tree_search.csv',index=False)
    br=grid[grid.task.eq('regression')].sort_values(['val_rmse','max_depth','min_samples_leaf']).iloc[0]
    bc=grid[grid.task.eq('classification')].sort_values(['val_f1','val_roc_auc','max_depth','min_samples_leaf'],ascending=[False,False,True,True]).iloc[0]
    rp=dict(max_depth=int(br.max_depth),min_samples_leaf=int(br.min_samples_leaf),random_state=42)
    cp=dict(criterion=bc.variant,max_depth=int(bc.max_depth),min_samples_leaf=int(bc.min_samples_leaf),random_state=42)
    log=br.variant=='log_price';yt=np.log(y) if log else y
    reg=DecisionTree(task='regression',**rp);pred,rt=_fit(reg,X,yt,Xv)
    skreg=DecisionTreeRegressor(**rp);spred,srt=_fit(skreg,X,yt,Xv)
    trainpred=reg.predict(X)
    if log: pred,spred,trainpred=np.exp(pred),np.exp(spred),np.exp(trainpred)
    clf=DecisionTree(**cp);pc,ct=_fit(clf,X,c,Xv);prob=clf.predict_proba(Xv)[:,1]
    skclf=DecisionTreeClassifier(**cp);spc,sct=_fit(skclf,X,c,Xv)
    majority=int(np.bincount(c).argmax())
    summary=dict(stage=2,evaluation_split='validation',test_evaluated=False,
                 train_rows=len(X),validation_rows=len(Xv),test_rows=len(split.test),features=X.shape[1],threshold=split.threshold,
                 environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,platform=platform.platform()),
                 search=dict(configurations=len(grid),depths=DEPTHS,min_samples_leaf=LEAF_SIZES,regression_selection='minimum validation RMSE',classification_selection='maximum validation F1; AUC tie-break'),
                 regression=dict(target=br.variant,parameters=rp,ours=regression_metrics(yv,pred),sklearn=regression_metrics(yv,spred),dummy=regression_metrics(yv,np.full(len(yv),y.mean())),ours_runtime=rt,sklearn_runtime=srt,train=regression_metrics(y,trainpred),train_tail=error_concentration(y,trainpred),validation_tail=error_concentration(yv,pred),max_prediction_difference=float(np.max(np.abs(pred-spred)))),
                 classification=dict(parameters=cp,ours=classification_metrics(cv,pc,prob),sklearn=classification_metrics(cv,spc,skclf.predict_proba(Xv)[:,1]),dummy=classification_metrics(cv,np.full(len(cv),majority),np.full(len(cv),c.mean())),ours_runtime=ct,sklearn_runtime=sct,prediction_agreement=float(np.mean(pc==spc))))
    (out/'tree_validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'feature':enc.feature_names_,'regression':reg.feature_importances_,'classification':clf.feature_importances_}).to_csv(out/'tree_feature_importance.csv',index=False)
    errors=[]
    cats=split.val.category.fillna('<MISSING>').to_numpy()
    for category in sorted(set(cats)):
        mask=cats==category;errors.append(dict(category=category,n=int(mask.sum()),**regression_metrics(yv[mask],pred[mask])))
    pd.DataFrame(errors).to_csv(out/'tree_error_by_category.csv',index=False)
    _plots(grid,summary,enc,reg,clf,yv,pred,figdir)
    _report(summary,errors,out)
    return summary

def _plots(grid,s,enc,reg,clf,yv,pred,figdir):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,variant in zip(axes.flat,['price','log_price','gini','entropy']):
        metric='rmse' if variant in ['price','log_price'] else 'f1'
        for leaf,color in [(10,'#2667a5'),(50,'#21918c')]:
            subset=grid[(grid.variant==variant)&(grid.min_samples_leaf==leaf)].sort_values('max_depth')
            for split,style in [('train','--'),('val','-')]:
                ax.plot(subset.max_depth,subset[split+'_'+metric],style,marker='o',color=color,label=f'{split}, leaf={leaf}')
        if metric=='rmse':ax.set_yscale('log')
        ax.set(title=variant,xlabel='max_depth',ylabel=metric.upper());ax.set_xticks(DEPTHS);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Training and validation: 32 configurations');fig.savefig(figdir/'04_tree_validation_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0];ax.hexbin(yv,pred,xscale='log',yscale='log',gridsize=45,mincnt=1,cmap='viridis')
    limits=[min(yv.min(),pred.min()),max(yv.max(),pred.max())];ax.plot(limits,limits,'r--',lw=1)
    ax.set(title='Validation price predictions',xlabel='Actual AZN (log scale)',ylabel='Predicted AZN (log scale)')
    cm=np.array(s['classification']['ours']['confusion_matrix']);ax=axes[0,1];ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=15)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Standard','Premium'],yticklabels=['Standard','Premium'],xlabel='Predicted',ylabel='Actual',title='Validation confusion matrix')
    for ax,model,title in [(axes[1,0],reg,'Regression'),(axes[1,1],clf,'Classification')]:
        imp=grouped_importance(enc.feature_names_,model.feature_importances_).head(10).sort_values()
        ax.barh(imp.index,imp.values,color='#2667a5');ax.set(title=title+' grouped importance',xlabel='Normalized impurity decrease')
    fig.savefig(figdir/'05_tree_diagnostics.png',dpi=150);plt.close(fig)

def _report(s,errors,out):
    r,c=s['regression'],s['classification'];lines=['# Stage 2: Decision Tree validation report','',
    'This is an intermediate validation report. The held-out test set has not been evaluated.',
    '', '## Protocol',f"Train: {s['train_rows']:,}; validation: {s['validation_rows']:,}; held-out test: {s['test_rows']:,}. Encoded features: {s['features']}. Premium = price > training median ({s['threshold']:,.0f} AZN).",
    'Group-disjoint splits and training-only preprocessing are inherited unchanged from Stage 1. Seed is 42. No repeated seed selection. All training rows are used in every search fit.',
    '32 candidates: regression price/log-price and classification Gini/entropy, each with depths 3/6/9/12 and minimum leaf sizes 10/50. Regression is selected by validation RMSE; classification by validation F1, then AUC. These selection scores are optimistic relative to unseen data.',
    '', '## Implementation',
    'NumPy implementation evaluates exact sorted adjacent-value thresholds. Regression uses centered prefix sums to reduce cancellation. Binary features use a single exact partition. Leaves predict target means or empirical class probabilities. Zero-gain splits are permitted when the stopping rules allow them, enabling XOR learning. Minimum impurity decrease uses the root-sample-weighted gain. Feature importance sums these gains and normalizes to one.',
    'Metrics (RMSE, MAE, R2, precision, recall, F1, tie-aware rank AUC and confusion matrix) are implemented in NumPy. sklearn is imported only for benchmarks and tests.',
    '', '## Selected configurations',f"Regression target: {r['target']}; parameters: {r['parameters']}",f"Classification: {c['parameters']}",
    'For log-price, predictions are exp(mean leaf log-price); no smearing correction is fitted. This optimizes log-scale squared loss and is evaluated on the original AZN scale.',
    '', '| Model | RMSE (AZN) | MAE (AZN) | R2 |','|---|---:|---:|---:|']
    for key in ['dummy','ours','sklearn']:
        m=r[key];lines.append(f"| {key} | {m['rmse']:,.2f} | {m['mae']:,.2f} | {m['r2']:.6f} |")
    lines+=['','| Model | Precision | Recall | F1 | AUC |','|---|---:|---:|---:|---:|']
    for key in ['dummy','ours','sklearn']:
        m=c[key];lines.append(f"| {key} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['roc_auc']:.6f} |")
    lines+=['','## Error analysis',f"The largest training price is {r['train_tail']['max_price']:,.0f} AZN; validation maximum is {r['validation_tail']['max_price']:,.0f} AZN. Top 1% of squared-error observations account for {100*r['train_tail']['top_one_percent_sse_share']:.2f}% of training SSE and {100*r['validation_tail']['top_one_percent_sse_share']:.2f}% of validation SSE.",
    'This severe price tail makes RMSE sensitive to a few observations and can make training RMSE exceed validation RMSE. All positive-price observations were retained; none were removed based on model error. Raw-price trees become unstable as depth increases; log-price trees are more robust on this validation split.',
    '', '| Category | n | RMSE AZN | MAE AZN |','|---|---:|---:|---:|']
    for row in errors:lines.append(f"| {row['category']} | {row['n']} | {row['rmse']:,.0f} | {row['mae']:,.0f} |")
    lines+=['','## Benchmark and timing',
    'sklearn models use identical data, target transformation and selected hyperparameters. Large-data predictions need not be identical: float32 input conversion in sklearn versus NumPy float64 and tie-breaking can produce different splits. Deterministic tiny examples verify the exact expected split and probability values.',
    f"Own/sklearn classification prediction agreement: {100*c['prediction_agreement']:.2f}%.",
    f"Own regression fit: {r['ours_runtime']['fit_seconds']:.3f}s; classification fit: {c['ours_runtime']['fit_seconds']:.3f}s. Single-run timings are hardware-dependent, not uncertainty estimates. Full timings and environment are in tree_validation_summary.json.",
    'Impurity feature importances are descriptive, may favor continuous or high-cardinality features, and are not causal effects.',
    '', '## Remaining work',
    'Implement linear Pegasos and nonlinear extension; validation tuning; group-aware CV and uncertainty; freeze configurations; evaluate the held-out test once; IEEE report and slides; contribution records and final tagged repository. Stage 2 alone is not the final submission.',
    '', '## AI assistance disclosure',
    'AI assisted with implementation, explanations and validation. Team members must review the code, reproduce the experiments, understand the derivations, and record their actual contributions. Do not claim AI-generated code as unaided work.',
    '', '## Reference documentation',
    '- https://scikit-learn.org/1.8/modules/tree.html',
    '- https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeRegressor.html']
    (out/'TREE_VALIDATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':run()
