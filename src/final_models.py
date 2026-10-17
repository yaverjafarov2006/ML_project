"""Complete pipelines. sklearn imports below belong to baselines only."""
import numpy as np
from .data_prep import FeatureEncoder
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures


class OwnClassifier:
    def __init__(self, config):
        self.config = dict(config)

    def fit(self, frame, y):
        self.encoder_ = FeatureEncoder().fit(frame)
        self.is_tree_ = self.config['kind'] == 'tree'
        X = self.encoder_.transform(frame, scaled=not self.is_tree_)
        self.rff_ = None
        if self.config['kind'] == 'rff':
            self.rff_ = RandomFourierFeatures(gamma=self.config['gamma'],
                n_components=self.config['n_components'], random_state=42)
            X = self.rff_.fit_transform(X)
        if self.is_tree_:
            self.model_ = DecisionTree(criterion=self.config['criterion'],
                max_depth=self.config['max_depth'], min_samples_leaf=self.config['min_samples_leaf'],
                random_state=42)
        else:
            self.model_ = PegasosSVM(lambda_=self.config['lambda_'],
                n_epochs=self.config['n_epochs'], schedule=self.config.get('schedule','pegasos'),
                batch_size=256, random_state=42)
        self.model_.fit(X, y)
        return self

    def _transform(self, frame):
        X = self.encoder_.transform(frame, scaled=not self.is_tree_)
        return self.rff_.transform(X) if self.rff_ is not None else X

    def decision_function(self, frame):
        X = self._transform(frame)
        return self.model_.predict_proba(X)[:,1] if self.is_tree_ else self.model_.decision_function(X)

    def predict(self, frame):
        return self.model_.predict(self._transform(frame))


def sklearn_classifier(config):
    """Preprocessing fitted inside Pipeline, including scaling for SVM.

    Own RFF is intentionally shared to isolate optimizer differences. The
    baseline is sklearn's SGD hinge solver, not a claim of identical updates.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier

    class FrameFeatures(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.encoder_ = FeatureEncoder().fit(X)
            return self
        def transform(self, X):
            return self.encoder_.transform(X)

    class RFFFeatures(TransformerMixin, BaseEstimator):
        def __init__(self, gamma=.003, n_components=1024):
            self.gamma=gamma; self.n_components=n_components
        def fit(self, X, y=None):
            self.mapper_=RandomFourierFeatures(self.gamma,self.n_components).fit(X)
            return self
        def transform(self, X):
            return self.mapper_.transform(X)

    steps=[('features',FrameFeatures())]
    if config['kind']=='tree':
        steps += [('model',DecisionTreeClassifier(criterion=config['criterion'],
            max_depth=config['max_depth'],min_samples_leaf=config['min_samples_leaf'],random_state=42))]
    else:
        steps += [('scaler',StandardScaler())]
        if config['kind']=='rff':
            steps += [('rff',RFFFeatures(config['gamma'],config['n_components']))]
        steps += [('model',SGDClassifier(loss='hinge',penalty='l2',alpha=config['lambda_'],
            max_iter=config['n_epochs'],tol=None,fit_intercept=True,shuffle=True,
            average=True,random_state=42,learning_rate='optimal'))]
    return Pipeline(steps)


def scores(model, frame, kind):
    if kind=='tree' and hasattr(model,'predict_proba'):
        return model.predict_proba(frame)[:,1]
    return model.decision_function(frame)
