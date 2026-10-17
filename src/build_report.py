"""Build report data from saved experiment results; optionally compile LaTeX."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from .data_prep import ROOT

NAMES={'own_tree_regression':'Own spatial trees','sklearn_tree_regression':'sklearn spatial trees',
 'training_median':'Training median','own_tree':'Own tree','sklearn_tree':'sklearn tree',
 'own_linear':'Own linear SVM','sklearn_linear':'SGD linear','own_rff':'Own RFF SVM',
 'sklearn_rff':'SGD + same RFF','training_majority':'Training majority'}


def tex_escape(value):
    return str(value).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#')


def build(compile_pdf=False):
    out=ROOT/'outputs';report=ROOT/'report';generated=report/'generated';generated.mkdir(exist_ok=True)
    rows=json.loads((out/'final_metrics.json').read_text());cfg=json.loads((out/'frozen_config.json').read_text())
    data=json.loads((out/'final_data_summary.json').read_text());err=json.loads((out/'final_error_analysis.json').read_text())
    cv=pd.read_csv(out/'final_cv_summary.csv');diag=pd.read_csv(out/'svm_diagnostics.csv')
    test={r['model']:r for r in rows if r['subset']=='test'}
    val={r['model']:r for r in rows if r['subset']=='validation'}
    def write(name,lines):
        (generated/(name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def row(*values):return ' & '.join(map(str,values))+r' \\'
    def table(name,header,entries,cols):
        write(name,[r'\begin{tabular}{@{}'+cols+r'@{}}',r'\toprule',row(*header),r'\midrule',*[row(*r) for r in entries],r'\bottomrule',r'\end{tabular}'])
    table('regression_table',['Model','RMSE','MAE',r'$R^2$','MedAPE'],[
        [NAMES[n],f"{test[n]['rmse']:,.0f}",f"{test[n]['mae']:,.0f}",f"{test[n]['r2']:.3f}",f"{test[n]['median_ape_pct']:.2f}\\%"]
        for n in ['training_median','own_tree_regression','sklearn_tree_regression']],'lrrrr')
    table('classification_table',['Model','Precision','Recall','F1','ROC AUC','AP'],[
        [NAMES[n],*[f"{test[n][k]:.3f}" for k in ['precision','recall','f1','roc_auc','pr_auc_ap']]]
        for n in ['training_majority','own_tree','sklearn_tree','own_linear','sklearn_linear','own_rff','sklearn_rff']],'lrrrrr')
    table('cv_table',['Model','RMSE mean $\\pm$ SD','F1 mean $\\pm$ SD'],[
        [NAMES[r.model],f'{r.rmse_mean:,.0f} $\\pm$ {r.rmse_std:,.0f}' if r.task=='regression' else '--',
         f'{r.f1_mean:.4f} $\\pm$ {r.f1_std:.4f}' if r.task=='classification' else '--'] for r in cv.itertuples()],'lrr')
    table('timing_table',['Model','Fit (s)','Predict (s)'],[
        [NAMES[n],f"{test[n]['fit_seconds']:.2f}",f"{test[n]['predict_seconds']:.3f}"]
        for n in ['own_tree_regression','sklearn_tree_regression','own_tree','sklearn_tree','own_linear','sklearn_linear','own_rff','sklearn_rff']],'lrr')
    table('svm_table',['Model',r'$J$',r'$\|w\|$',r'$2/\|w\|$','Margin set'],[
        [NAMES[r.model],f'{r.objective:.4f}',f'{r.weight_norm:.3f}',f'{r.margin_width:.3f}',f'{r.margin_violator_fraction*100:.1f}\\%']
        for r in diag.itertuples()],'lrrrr')
    bycat=pd.read_csv(out/'final_regression_error_by_category.csv')
    cats={'yeni tikili':'New apartments','köhnə tikili':'Old apartments','həyət evi/bağ evi':'Detached / garden'}
    table('category_table',['Category','n','MAE (AZN)','MedAPE'],[
        [cats[r.category],f'{r.n:,}',f'{r.mae:,.0f}',f'{r.median_ape_pct:.2f}\\%'] for r in bycat.itertuples()],'lrrr')
    table('split_table',['Subset','Rows','Groups','Premium'],[
        [k.title(),f"{v['n']:,}",f"{v['groups']:,}",f"{100*v['premium']/v['n']:.2f}\\%"] for k,v in data['split'].items()],'lrrr')
    g=pd.read_csv(out/'final_classification_error_groups.csv');near=g[g.group_type.eq('threshold_distance')&g.model.eq('own_tree')]
    near_error=float(near[near.group.eq('within_10pct_of_threshold')].error_rate.iloc[0])*100
    far_error=float(near[near.group.eq('farther_from_threshold')].error_rate.iloc[0])*100
    treecv=cv[cv.model.eq('own_tree_regression')].iloc[0]
    importance=pd.read_csv(out/'final_feature_importance.csv')
    weights=pd.read_csv(out/'linear_svm_weights.csv').set_index('feature').coefficient
    values={'regRMSE':f"{test['own_tree_regression']['rmse']:,.0f}",
        'regMAE':f"{test['own_tree_regression']['mae']:,.0f}",
        'regRsq':f"{test['own_tree_regression']['r2']:.3f}",
        'regMedAPE':f"{test['own_tree_regression']['median_ape_pct']:.2f}",
        'regWithin':f"{test['own_tree_regression']['within_20_pct']:.1f}",
        'treeFone':f"{test['own_tree']['f1']:.4f}",'treeAUC':f"{test['own_tree']['roc_auc']:.4f}",
        'linearFone':f"{test['own_linear']['f1']:.4f}",'rffFone':f"{test['own_rff']['f1']:.4f}",
        'treeValFone':f"{val['own_tree']['f1']:.4f}",'rffValFone':f"{val['own_rff']['f1']:.4f}",
        'linearValFone':f"{val['own_linear']['f1']:.4f}",
        'sseShare':f"{100*err['top_one_percent_sse_share']:.1f}",
        'testMax':f"{err['test_max_price']:,.0f}",
        'nearError':f'{near_error:.1f}','farError':f'{far_error:.1f}',
        'cvMedAPE':f'{treecv.median_ape_pct_mean:.2f}', 'cvMedAPEsd':f'{treecv.median_ape_pct_std:.2f}',
        'areaImportance':f"{100*importance[importance.model.eq('own_tree')&importance.feature.eq('area_m2')].importance.sum():.1f}",
        'logAreaWeight':f"{weights['log_area_m2']:.3f}", 'rawAreaWeight':f"{weights['area_m2']:.3f}"}
    meta_path=ROOT/'submission_metadata.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    authors=meta.get('authors',[])
    values['authorline']=tex_escape('; '.join(authors)) if authors else 'MLE-AI-201 Course Project'
    repo=meta.get('repository_url')
    values['repositoryline']=(r'Repository: \url{'+repo+r'} (tag \texttt{v1.0-final}).') if repo else 'Repository URL has not been supplied; reproducible source is included in the companion project package.'
    write('values',[chr(92)+'newcommand{'+chr(92)+k+'}{'+v+'}' for k,v in values.items()])
    write('confusion',[r'\begin{tabular}{lrr}',row('Actual / predicted','Standard','Premium'),r'\hline',
        row('Standard',*test['own_tree']['confusion_matrix'][0]),row('Premium',*test['own_tree']['confusion_matrix'][1]),r'\end{tabular}'])
    if compile_pdf:
        compiler=shutil.which('pdflatex')
        if not compiler:raise RuntimeError('pdflatex is needed to compile report.tex; existing report.pdf is included.')
        for _ in range(2):subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=report,check=True,stdout=subprocess.DEVNULL)
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compile',action='store_true');a=p.parse_args();build(a.compile)
