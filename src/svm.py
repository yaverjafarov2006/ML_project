"""NumPy-only mini-batch Pegasos and random Fourier features.

The bias is separate and unregularized. Each epoch shuffles every training
row once. Updates use the batch mean of hinge subgradients and project w onto
the radius 1/sqrt(lambda) ball. By default eta_t=1/(lambda*t); alternative
schedules are exposed for a controlled ablation, not called Pegasos.
Final coefficients average iterates from the second half of training.
"""
import numpy as np


def _matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError('X must be a nonempty finite matrix')
    return X


class PegasosSVM:
    def __init__(self, lambda_=1e-3, n_epochs=60, batch_size=256,
                 schedule='pegasos', eta0=0.05, average=True, random_state=42):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.schedule = schedule
        self.eta0 = eta0
        self.average = average
        self.random_state = random_state

    @staticmethod
    def objective(X, y_pm, w, b, lambda_):
        return float(lambda_ * (w @ w) / 2 + np.maximum(0, 1-y_pm*(X@w+b)).mean())

    def learning_rate(self, t):
        if self.schedule == 'pegasos':
            return 1. / (self.lambda_ * t)
        if self.schedule == 'constant':
            return self.eta0
        if self.schedule == 'inverse_sqrt':
            return self.eta0 / np.sqrt(t)
        raise ValueError('Unknown learning-rate schedule')

    def fit(self, X, y):
        X = _matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('One label per row required')
        if not np.isfinite(self.lambda_) or self.lambda_ <= 0:
            raise ValueError('lambda_ must be positive and finite')
        if not isinstance(self.n_epochs, (int, np.integer)) or self.n_epochs < 1:
            raise ValueError('n_epochs must be a positive integer')
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not np.isfinite(self.eta0) or self.eta0 <= 0:
            raise ValueError('eta0 must be positive and finite')
        self.learning_rate(1)
        self.classes_, yi = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError('Exactly two classes required')
        yp = 2 * yi - 1
        self.n_features_in_ = X.shape[1]
        w = np.zeros(X.shape[1]); b = 0.
        wsum = np.zeros_like(w); bsum = 0.; n_average = 0; t = 0
        rng = np.random.default_rng(self.random_state)
        total_steps = self.n_epochs * int(np.ceil(len(X) / self.batch_size))
        self.history_ = [dict(epoch=0, objective=1., averaged_objective=1., weight_norm=0.)]
        for epoch in range(self.n_epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                ix = order[start:start+self.batch_size]
                xb, yb = X[ix], yp[ix]
                active = yb * (xb @ w + b) < 1.
                gradient = xb.T @ (yb * active) / len(ix)
                t += 1; eta = self.learning_rate(t)
                w = (1. - eta*self.lambda_) * w + eta * gradient
                b += eta * float(np.mean(yb * active))
                norm = np.linalg.norm(w)
                w *= min(1., 1./(np.sqrt(self.lambda_) * norm)) if norm else 1.
                if self.average and t > total_steps // 2:
                    wsum += w; bsum += b; n_average += 1
            wa, ba = (wsum/n_average, bsum/n_average) if n_average else (w, b)
            self.history_.append(dict(epoch=epoch+1,
                objective=self.objective(X, yp, w, b, self.lambda_),
                averaged_objective=self.objective(X, yp, wa, ba, self.lambda_),
                weight_norm=float(np.linalg.norm(w))))
        self.raw_w_, self.raw_b_ = w.copy(), b
        self.w, self.b = (wsum/n_average, bsum/n_average) if n_average else (w, b)
        self.n_steps_ = t
        self.weight_norm_ = float(np.linalg.norm(self.w))
        self.margin_width_ = float(2/self.weight_norm_) if self.weight_norm_ else float('inf')
        self.support_mask_ = yp * (X @ self.w + self.b) <= 1.
        self.objective_ = self.objective(X, yp, self.w, self.b, self.lambda_)
        return self

    def decision_function(self, X):
        if not hasattr(self, 'w'):
            raise RuntimeError('Call fit first')
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count mismatch')
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]


class RandomFourierFeatures:
    """Approximate exp(-gamma*||x-z||^2) with sqrt(2/m)*cos(xW+b)."""
    def __init__(self, gamma=0.003, n_components=1024, random_state=42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = _matrix(X)
        if not np.isfinite(self.gamma) or self.gamma <= 0 or self.n_components < 1:
            raise ValueError('Positive gamma and n_components required')
        if not isinstance(self.n_components, (int, np.integer)):
            raise ValueError('n_components must be an integer')
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0., np.sqrt(2*self.gamma), (X.shape[1], self.n_components))
        self.offset_ = rng.uniform(0, 2*np.pi, self.n_components)
        return self

    def transform(self, X):
        if not hasattr(self, 'weights_'):
            raise RuntimeError('Call fit first')
        X = _matrix(X)
        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError('Feature count mismatch')
        result = np.empty((len(X), self.n_components))
        for start in range(0, len(X), 4096):
            result[start:start+4096] = np.cos(X[start:start+4096] @ self.weights_ + self.offset_)
        result *= np.sqrt(2/self.n_components)
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
