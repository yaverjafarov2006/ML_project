"""Metrics implemented in NumPy; positive class is premium=1."""
import numpy as np

def _pair(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=1 or b.shape!=a.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching finite nonempty vectors')
    return a,b

def rmse(y_true,y_pred):
    a,b=_pair(y_true,y_pred); return float(np.sqrt(np.mean((a-b)**2)))

def mae(y_true,y_pred):
    a,b=_pair(y_true,y_pred); return float(np.mean(np.abs(a-b)))

def r2(y_true,y_pred):
    a,b=_pair(y_true,y_pred)
    if len(a)<2: return float('nan')
    residual=np.sum((a-b)**2); total=np.sum((a-a.mean())**2)
    return float(1-residual/total) if total else float(residual==0)

def confusion_matrix(y_true,y_pred):
    a,b=_pair(y_true,y_pred)
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all(): raise ValueError('Binary labels required')
    return np.bincount(2*a.astype(int)+b.astype(int),minlength=4).reshape(2,2)

def precision_recall_f1(y_true,y_pred):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
    p=float(tp/(tp+fp)) if tp+fp else 0.
    r=float(tp/(tp+fn)) if tp+fn else 0.
    return p,r,2*p*r/(p+r) if p+r else 0.

def roc_auc(y_true,scores):
    y,s=_pair(y_true,scores)
    if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2: raise ValueError('AUC needs both binary classes')
    order=np.argsort(s,kind='stable'); sorted_s=s[order]
    starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1]; ends=np.r_[starts[1:],len(y)]
    ranks=np.empty(len(y)); ranks[order]=np.repeat((starts+1+ends)/2,ends-starts)
    npos=int(y.sum()); nneg=len(y)-npos
    return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def regression_metrics(y,p): return {'rmse':rmse(y,p),'mae':mae(y,p),'r2':r2(y,p)}
def classification_metrics(y,p,s):
    precision,recall,f1=precision_recall_f1(y,p)
    return dict(precision=precision,recall=recall,f1=f1,roc_auc=roc_auc(y,s),confusion_matrix=confusion_matrix(y,p).tolist())
def compare(name_to_metrics):
    for name,metrics in name_to_metrics.items():
        print(name, ' | '.join(f'{k}: {v:.6g}' for k,v in metrics.items() if np.isscalar(v)))
