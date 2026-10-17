"""Exact CART-style tree; NumPy only, with weighted impurity importance."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    n_samples: int
    impurity: float
    value: object
    proba: object = None
    feature: object = None
    threshold: object = None
    left: object = None
    right: object = None
    def is_leaf(self):
        return self.left is None and self.right is None

class DecisionTree:
    def __init__(self, task='classification', criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0., random_state=42):
        self.task=task
        self.criterion=criterion or ('mse' if task=='regression' else 'gini')
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.min_impurity_decrease=min_impurity_decrease
        self.random_state=random_state
        self.root=None
        if task not in ('classification','regression'):
            raise ValueError('Unknown task')
        if self.criterion not in (('mse',) if task=='regression' else ('gini','entropy')):
            raise ValueError('Criterion does not match task')
        for name, value, minimum in [('max_depth',max_depth,0),('min_samples_split',min_samples_split,2),('min_samples_leaf',min_samples_leaf,1)]:
            if name=='max_depth' and value is None: continue
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum:
                raise ValueError(name+' must be an integer >= '+str(minimum))
        if not np.isfinite(min_impurity_decrease) or min_impurity_decrease<0:
            raise ValueError('Invalid min_impurity_decrease')

    def _class_impurity(self, counts):
        p=counts/counts.sum(axis=-1,keepdims=True)
        if self.criterion=='gini': return 1-np.sum(p*p,axis=-1)
        logs=np.zeros_like(p,dtype=float)
        np.log2(p,out=logs,where=p>0)
        return -np.sum(p*logs,axis=-1)

    def _impurity(self,y):
        if self.task=='regression': return float(np.mean((y-y.mean())**2))
        return float(self._class_impurity(np.bincount(y,minlength=len(self.classes_))))

    def _best_split(self,X,y):
        n=len(y); parent=self._impurity(y)
        best=None; best_gain=0.
        tol=np.finfo(float).eps*max(1.,parent)*16
        for j in self._rng.permutation(X.shape[1]):
            values=X[:,j]; lo,hi=values.min(),values.max()
            if lo==hi: continue
            binary=(values==lo)|(values==hi)
            if binary.all():
                mask=values==lo; nl=int(mask.sum())
                if min(nl,n-nl)<self.min_samples_leaf: continue
                gain=parent-(nl*self._impurity(y[mask])+(n-nl)*self._impurity(y[~mask]))/n
                lower,upper=lo,hi
            else:
                order=np.argsort(values,kind='stable'); xs=values[order]; ys=y[order]
                cut=np.flatnonzero(xs[:-1]<xs[1:])+1
                cut=cut[(cut>=self.min_samples_leaf)&(n-cut>=self.min_samples_leaf)]
                if not len(cut): continue
                if self.task=='regression':
                    # Center first to avoid cancellation for large target offsets.
                    z=ys-ys.mean(); s=np.cumsum(z); q=np.cumsum(z*z)
                    ls=s[cut-1]; lq=q[cut-1]
                    cost=(np.maximum(0,lq-ls*ls/cut)+np.maximum(0,q[-1]-lq-(s[-1]-ls)**2/(n-cut)))/n
                else:
                    counts=np.cumsum(np.eye(len(self.classes_))[ys],axis=0)
                    left=counts[cut-1]; right=counts[-1]-left
                    cost=(cut*self._class_impurity(left)+(n-cut)*self._class_impurity(right))/n
                k=int(np.argmin(cost)); pos=cut[k]
                gain=parent-float(cost[k]); lower,upper=xs[pos-1],xs[pos]
            if (best is None and gain>=-tol) or gain>best_gain+tol:
                threshold=lower/2+upper/2
                if threshold>=upper: threshold=lower
                best_gain=max(0.,float(gain)); best=(int(j),float(threshold),best_gain)
        return best

    def _grow(self,X,y,depth):
        impurity=self._impurity(y)
        if self.task=='classification':
            counts=np.bincount(y,minlength=len(self.classes_)); proba=counts/len(y)
            node=Node(len(y),impurity,self.classes_[counts.argmax()],proba)
        else: node=Node(len(y),impurity,float(y.mean()))
        self.depth_=max(self.depth_,depth)
        stop=(self.max_depth is not None and depth>=self.max_depth) or len(y)<max(self.min_samples_split,2*self.min_samples_leaf) or impurity<=0
        split=None if stop else self._best_split(X,y)
        if split is None or len(y)/self.n_samples_*split[2]<self.min_impurity_decrease:
            self.n_leaves_+=1; return node
        node.feature,node.threshold,gain=split
        self.feature_importances_[node.feature]+=len(y)/self.n_samples_*gain
        mask=X[:,node.feature]<=node.threshold
        node.left=self._grow(X[mask],y[mask],depth+1)
        node.right=self._grow(X[~mask],y[~mask],depth+1)
        return node

    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y)
        if X.ndim!=2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError('X must be a finite, nonempty 2D array')
        if y.ndim!=1 or len(y)!=len(X): raise ValueError('y must match X rows')
        if self.task=='classification':
            if y.dtype.kind in 'fc' and not np.isfinite(y).all(): raise ValueError('Nonfinite labels')
            self.classes_,y=np.unique(y,return_inverse=True)
        else:
            y=y.astype(float)
            if not np.isfinite(y).all(): raise ValueError('Nonfinite target')
        self.n_samples_,self.n_features_in_=X.shape
        self.depth_=0; self.n_leaves_=0
        self.feature_importances_=np.zeros(X.shape[1]); self._rng=np.random.default_rng(self.random_state)
        self.root=self._grow(X,y,0)
        total=self.feature_importances_.sum()
        if total>0: self.feature_importances_/=total
        return self

    def _leaves(self,X):
        if self.root is None: raise ValueError('Fit the model first')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('Invalid prediction features')
        stack=[(self.root,np.arange(len(X)))]
        while stack:
            node,idx=stack.pop()
            if not len(idx): continue
            if node.is_leaf(): yield idx,node
            else:
                mask=X[idx,node.feature]<=node.threshold
                stack.extend([(node.left,idx[mask]),(node.right,idx[~mask])])

    def predict(self,X):
        X=np.asarray(X); leaves=list(self._leaves(X))
        out=np.empty(len(X),dtype=self.classes_.dtype if self.task=='classification' else float)
        for idx,node in leaves: out[idx]=node.value
        return out

    def predict_proba(self,X):
        if self.task!='classification': raise ValueError('Classification only')
        X=np.asarray(X); leaves=list(self._leaves(X))
        out=np.empty((len(X),len(self.classes_)))
        for idx,node in leaves: out[idx]=node.proba
        return out
