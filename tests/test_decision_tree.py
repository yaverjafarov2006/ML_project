import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from src.decision_tree import DecisionTree

class TreeTests(unittest.TestCase):
    def test_xor_zero_gain_split(self):
        X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,1,1,0])
        np.testing.assert_array_equal(DecisionTree(max_depth=2).fit(X,y).predict(X),y)
    def test_regression_known_split(self):
        X=np.array([[0],[1],[4],[5]]); y=np.array([2,2,9,9])
        a=DecisionTree(task='regression',max_depth=1).fit(X,y)
        b=DecisionTreeRegressor(max_depth=1).fit(X,y)
        self.assertEqual(a.root.threshold,2.5); self.assertEqual(a.root.threshold,b.tree_.threshold[0])
        np.testing.assert_allclose(a.predict(X),b.predict(X))
    def test_gini_and_entropy(self):
        X=np.arange(6)[:,None]; y=np.array([0,0,0,1,1,1])
        for criterion in ['gini','entropy']:
            a=DecisionTree(criterion=criterion,max_depth=1).fit(X,y)
            b=DecisionTreeClassifier(criterion=criterion,max_depth=1).fit(X,y)
            self.assertEqual(a.root.threshold,b.tree_.threshold[0])
            np.testing.assert_allclose(a.predict_proba(X),b.predict_proba(X))
    def test_constant_features_labels(self):
        a=DecisionTree().fit(np.zeros((4,2)),['a','b','b','b'])
        self.assertEqual(a.n_leaves_,1)
        np.testing.assert_allclose(a.predict_proba([[0,0]]),[[.25,.75]])
        self.assertEqual(a.predict([[0,0]])[0],'b')
    def test_regression_leaf_mean(self):
        a=DecisionTree(task='regression',max_depth=0).fit([[0],[1]],[1,9])
        self.assertEqual(a.predict([[.5]])[0],5)
    def test_stopping_rules(self):
        X=np.arange(20)[:,None]; y=np.arange(20)%2
        base=DecisionTree().fit(X,y)
        for kwargs in [dict(max_depth=1),dict(min_samples_split=30),dict(min_samples_leaf=8),dict(min_impurity_decrease=.1)]:
            a=DecisionTree(**kwargs).fit(X,y)
            self.assertLess(a.n_leaves_,base.n_leaves_)
    def test_global_impurity_decrease(self):
        X=np.arange(12)[:,None]; y=np.array([0,0,0,0,0,0,1,1,1,0,1,1])
        for decrease in [0,.02,.1,.5]:
            a=DecisionTree(min_impurity_decrease=decrease).fit(X,y)
            b=DecisionTreeClassifier(min_impurity_decrease=decrease,random_state=42).fit(X,y)
            np.testing.assert_array_equal(a.predict(X),b.predict(X))
    def test_importance_and_leaf_size(self):
        X=np.c_[np.arange(30),np.ones(30)]; y=np.arange(30)
        a=DecisionTree(task='regression',min_samples_leaf=5).fit(X,y)
        np.testing.assert_allclose(a.feature_importances_,[1,0])
        def check(node):
            if node.is_leaf(): self.assertGreaterEqual(node.n_samples,5)
            else: check(node.left);check(node.right)
        check(a.root)
    def test_large_offset(self):
        X=np.arange(8)[:,None]; y=1e12+np.array([0,0,0,0,100,100,100,100])
        a=DecisionTree(task='regression',max_depth=1).fit(X,y)
        self.assertEqual(a.root.threshold,3.5); np.testing.assert_allclose(a.predict(X)-1e12,y-1e12)
    def test_seed_and_refit(self):
        X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,1,1,0])
        a=DecisionTree().fit(X,y); p=a.predict(X).copy()
        a.fit(X,y);np.testing.assert_array_equal(a.predict(X),p)
        a.fit(X,np.ones(4));self.assertEqual(a.n_leaves_,1);self.assertEqual(a.feature_importances_.sum(),0)
    def test_invalid_inputs(self):
        for kw in [dict(task='bad'),dict(task='regression',criterion='gini'),dict(min_samples_leaf=0),dict(max_depth=-1),dict(min_samples_split=1),dict(min_impurity_decrease=-1)]:
            with self.assertRaises(ValueError): DecisionTree(**kw)
        for X,y in [([],[]),([[np.nan]],[0]),([[1],[2]],[1])]:
            with self.assertRaises(ValueError): DecisionTree().fit(X,y)
        with self.assertRaises(ValueError): DecisionTree().predict([[1]])
        with self.assertRaises(ValueError): DecisionTree(task='regression').fit([[1]],[1]).predict_proba([[1]])
    def test_empty_prediction_and_pure_leaf(self):
        a=DecisionTree().fit([[0],[1]],[1,1]);self.assertEqual(a.predict(np.empty((0,1))).shape,(0,))
        np.testing.assert_allclose(a.predict_proba([[5]]),[[1]])

if __name__=='__main__': unittest.main()
