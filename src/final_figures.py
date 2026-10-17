"""Regenerate publication figures from numerical outputs only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from .data_prep import ROOT

LABELS={'own_tree':'Own tree','own_linear':'Own linear SVM','own_rff':'Own RFF SVM',
        'sklearn_tree':'sklearn tree','sklearn_linear':'SGD linear','sklearn_rff':'SGD + same RFF'}
COLORS={'own_tree':'#2563a6','own_linear':'#a36819','own_rff':'#167d72',
        'sklearn_tree':'#9fc2df','sklearn_linear':'#d9b680','sklearn_rff':'#92c9bd'}


def build():
    out=ROOT/'outputs';figdir=ROOT/'report'/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titlesize':12,'axes.labelsize':10,'savefig.dpi':170})
    met=pd.DataFrame(json.loads((out/'final_metrics.json').read_text()))
    cfg=json.loads((out/'frozen_config.json').read_text());npz=np.load(out/'final_predictions.npz')
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    cls=met[met.subset.eq('test')&met.model.isin(LABELS)]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,key,title in zip(axes,['f1','roc_auc'],['Held-out test F1','Held-out test ROC AUC']):
        names=cls.model.to_list();vals=cls[key].to_numpy()
        ax.barh([LABELS[n] for n in names],vals,color=[COLORS[n] for n in names])
        ax.set_xlim(0,1.08);ax.set_title(title);ax.invert_yaxis()
        for i,v in enumerate(vals):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
    save(fig,'final_classification')
    hist=pd.read_csv(out/'svm_objective_history.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,name in zip(axes,['own_linear','own_rff']):
        h=hist[hist.model.eq(name)]
        ax.plot(h.epoch,h.objective,label='Last iterate',color=COLORS[name],alpha=.5)
        ax.plot(h.epoch,h.averaged_objective,label='Suffix average',color=COLORS[name])
        ax.axhline(1,color='grey',ls=':',label='Zero-vector objective')
        ax.set(title=LABELS[name],xlabel='Epoch',ylabel='Training primal objective',yscale='log');ax.legend(fontsize=8)
    save(fig,'final_svm_objective')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    y=(npz['test_price']>cfg['threshold']).astype(int)
    for name in ['own_tree','own_linear','own_rff']:
        s=npz[f'test_{name}_score'];fpr,tpr,_=roc_curve(y,s);p,r,_=precision_recall_curve(y,s)
        axes[0].plot(fpr,tpr,label=LABELS[name],color=COLORS[name]);axes[1].plot(r,p,label=LABELS[name],color=COLORS[name])
    axes[0].plot([0,1],[0,1],':',color='grey');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC curves')
    axes[1].axhline(y.mean(),ls=':',color='grey');axes[1].set(xlabel='Recall',ylabel='Precision',title='Test precision–recall curves')
    for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.03)
    save(fig,'final_roc_pr')
    winner='own_'+cfg['classification_winner'];r=cls[cls.model.eq(winner)].iloc[0]
    cm=np.array(r.confusion_matrix)
    fig,ax=plt.subplots(figsize=(4,3.6),layout='constrained');ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=17,color='white' if cm[i,j]>cm.max()/2 else '#142b42')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Standard','Premium'],yticklabels=['Standard','Premium'],xlabel='Predicted',ylabel='Actual',title=LABELS[winner]+' — test')
    save(fig,'final_confusion')
    yt=npz['test_price'];pt=npz['test_own_tree_regression_prediction'];ape=np.abs(yt-pt)/yt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    axes[0].scatter(yt,pt,s=4,alpha=.2,color='#2563a6',rasterized=True)
    low=min(yt.min(),pt.min());high=max(yt.max(),pt.max())
    axes[0].plot([low,high],[low,high],color='#a36819',lw=1)
    axes[0].set(xscale='log',yscale='log',xlabel='Actual asking price (AZN)',ylabel='Predicted price (AZN)',title='Own tree: all test rows')
    axes[1].plot(np.sort(ape)*100,np.arange(1,len(ape)+1)/len(ape)*100,color='#167d72')
    axes[1].axvline(20,color='grey',ls=':');axes[1].set(xlim=(0,100),ylim=(0,100),xlabel='Absolute percentage error (%)',ylabel='Test rows within error (%)',title='Error distribution; all rows retained')
    save(fig,'final_regression')
    imp=pd.read_csv(out/'final_feature_importance.csv');weights=pd.read_csv(out/'linear_svm_weights.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    part=imp[imp.model.eq('own_tree')].copy();part['group']=part.feature.str.split('=').str[0].str.replace('__missing','',regex=False)
    g=part.groupby('group').importance.sum().nlargest(8).sort_values()
    axes[0].barh(g.index,g.values,color='#2563a6');axes[0].set(title='Tree classifier importance',xlabel='Normalized impurity reduction')
    w=weights.iloc[np.argsort(np.abs(weights.coefficient))[-8:]].sort_values('coefficient')
    axes[1].barh(w.feature.str.slice(0,35),w.coefficient,color=['#a36819' if v<0 else '#167d72' for v in w.coefficient]);axes[1].set(title='Linear SVM: largest |weights|',xlabel='Coefficient on standardized feature')
    save(fig,'final_interpretation')
    timing=pd.read_csv(out/'runtime_scaling.csv');fig,ax=plt.subplots(figsize=(5,3.6),layout='constrained')
    for name,g in timing.groupby('model'):
        ax.plot(g.n_train,g.fit_seconds,'o-',label=LABELS[name],color=COLORS[name])
    ax.set(xlabel='Training rows',ylabel='Fit seconds, including preprocessing',title='Measured scaling (two BLAS threads)');ax.legend(fontsize=8)
    save(fig,'final_runtime')
    if (out/'final_cv_folds.csv').exists():
        cv=pd.read_csv(out/'final_cv_folds.csv');r=cv[cv.model.eq('own_tree_regression')]
        fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
        axes[0].bar(r.fold,r.rmse/1e6,color='#2563a6');axes[0].set(title='Grouped CV exposes price-label risk',xlabel='Fold',ylabel='RMSE (million AZN)')
        axes[1].bar(r.fold,r.median_ape_pct,color='#167d72');axes[1].set(title='Typical relative error is more stable',xlabel='Fold',ylabel='Median absolute percentage error (%)')
        save(fig,'final_cv_stability')
    search=pd.read_csv(out/'final_classification_search.csv')
    params=pd.json_normalize(search.config.map(json.loads));search=pd.concat([search.drop(columns='config'),params],axis=1)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for criterion in ['gini','entropy']:
        g=search[search.kind.eq('tree')&search.criterion.eq(criterion)&search.min_samples_leaf.eq(5)].sort_values('max_depth')
        axes[0].plot(g.max_depth,g.train_f1,'--',label=criterion+' train')
        axes[0].plot(g.max_depth,g.val_f1,'o-',label=criterion+' validation')
    axes[0].set(xlabel='Maximum depth (minimum leaf = 5)',ylabel='F1',title='Tree capacity and generalization');axes[0].legend(fontsize=8)
    for kind in ['linear','rff']:
        g=search[search.kind.eq(kind)&search.n_epochs.eq(60)&search.schedule.eq('pegasos')]
        if kind=='rff':g=g[g.gamma.eq(cfg['classification']['rff']['gamma'])&g.n_components.eq(1024)]
        g=g.sort_values('lambda_');axes[1].plot(g.lambda_,g.val_f1,'o-',label=kind,color=COLORS['own_'+kind])
    axes[1].set(xscale='log',xlabel='Regularization lambda',ylabel='Validation F1',title='SVM regularization trade-off');axes[1].legend(fontsize=8)
    save(fig,'final_tuning')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    g=search[search.kind.eq('linear')&search.n_epochs.eq(60)&search.schedule.eq('pegasos')].sort_values('lambda_')
    axes[0].plot(g.lambda_,g.margin_width,'o-',color='#a36819');axes[0].set(xscale='log',xlabel='Regularization lambda',ylabel='2 / ||w||',title='Linear margin in standardized space')
    g=search[search.kind.eq('linear')&search.lambda_.eq(cfg['classification']['linear']['lambda_'])]
    labs=[f'{r.schedule}\n{int(r.n_epochs)} epochs' for _,r in g.iterrows()]
    axes[1].bar(labs,g.val_f1,color='#2563a6');axes[1].set(ylim=(0,1),ylabel='Validation F1',title='Epoch and learning-rate diagnostics');axes[1].tick_params(axis='x',labelsize=7)
    save(fig,'final_margin_schedule')
    # Single-column variants keep labels legible in the IEEE-style report.
    fig,ax=plt.subplots(figsize=(4,2.7),layout='constrained')
    for criterion in ['gini','entropy']:
        g=search[search.kind.eq('tree')&search.criterion.eq(criterion)&search.min_samples_leaf.eq(5)].sort_values('max_depth')
        ax.plot(g.max_depth,g.train_f1,'--',label=criterion+' train')
        ax.plot(g.max_depth,g.val_f1,'o-',label=criterion+' validation')
    ax.set(xlabel='Maximum depth (leaf = 5)',ylabel='F1');ax.legend(fontsize=7)
    save(fig,'report_tuning')
    fig,ax=plt.subplots(figsize=(4,2.7),layout='constrained')
    ax.plot(np.sort(ape)*100,np.arange(1,len(ape)+1)/len(ape)*100,color='#167d72')
    ax.axvline(20,color='grey',ls=':');ax.set(xlim=(0,100),ylim=(0,100),xlabel='Absolute percentage error (%)',ylabel='Test rows within error (%)')
    save(fig,'report_regression')
    fig,ax=plt.subplots(figsize=(4,2.7),layout='constrained')
    for name in ['own_linear','own_rff']:
        h=hist[hist.model.eq(name)];ax.plot(h.epoch,h.averaged_objective,label=LABELS[name],color=COLORS[name])
    ax.axhline(1,color='grey',ls=':',label='Zero vector');ax.set(yscale='log',xlabel='Epoch',ylabel='Training objective');ax.legend(fontsize=8)
    save(fig,'report_svm_objective')
    fig,ax=plt.subplots(figsize=(4,2.7),layout='constrained')
    for name in ['own_tree','own_linear','own_rff']:
        fpr,tpr,_=roc_curve(y,npz[f'test_{name}_score']);ax.plot(fpr,tpr,label=LABELS[name],color=COLORS[name])
    ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8)
    save(fig,'report_roc')
    if (out/'final_cv_folds.csv').exists():
        fig,ax=plt.subplots(figsize=(4,2.5),layout='constrained')
        ax.bar(r.fold,r.rmse/1e6,color='#2563a6');ax.set(xlabel='Grouped fold',ylabel='RMSE (million AZN)')
        save(fig,'report_cv')
    fig,axes=plt.subplots(2,1,figsize=(4,5.2),layout='constrained')
    part=imp[imp.model.eq('own_tree')].copy();part['group']=part.feature.str.split('=').str[0].str.replace('__missing','',regex=False)
    g=part.groupby('group').importance.sum().nlargest(6).sort_values()
    axes[0].barh(g.index,g.values,color='#2563a6');axes[0].set(xlabel='Tree impurity importance')
    w=weights.iloc[np.argsort(np.abs(weights.coefficient))[-6:]].sort_values('coefficient')
    axes[1].barh(w.feature.str.slice(0,27),w.coefficient,color='#167d72');axes[1].set(xlabel='Linear SVM coefficient')
    for ax in axes:ax.tick_params(labelsize=8)
    save(fig,'report_interpretation')


if __name__=='__main__':build()
