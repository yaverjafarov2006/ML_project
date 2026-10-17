import unittest
import numpy as np
from sklearn import metrics
from src.evaluate import *

class MetricTests(unittest.TestCase):
    def test_regression(self):
        y=np.array([1,5,9,11]);p=np.array([2,5,7,12])
        self.assertAlmostEqual(rmse(y,p),metrics.root_mean_squared_error(y,p))
        self.assertAlmostEqual(mae(y,p),metrics.mean_absolute_error(y,p))
        self.assertAlmostEqual(r2(y,p),metrics.r2_score(y,p))
    def test_confusion_and_rates(self):
        y=[0,0,1,1,1];p=[0,1,0,1,1]
        np.testing.assert_array_equal(confusion_matrix(y,p),metrics.confusion_matrix(y,p))
        np.testing.assert_allclose(precision_recall_f1(y,p),[metrics.precision_score(y,p),metrics.recall_score(y,p),metrics.f1_score(y,p)])
    def test_tied_auc(self):
        self.assertEqual(roc_auc([0,1,0,1],[.5,.5,.5,.5]),.5)
        self.assertEqual(roc_auc([0,1],[0,1]),1)
        self.assertEqual(roc_auc([0,1],[1,0]),0)
    def test_random_auc(self):
        rng=np.random.default_rng(42)
        for _ in range(10):
            y=rng.integers(0,2,200);s=np.round(rng.normal(size=200),1)
            self.assertAlmostEqual(roc_auc(y,s),metrics.roc_auc_score(y,s))
    def test_edge_cases(self):
        self.assertEqual(r2([3,3],[3,3]),1);self.assertEqual(r2([3,3],[2,2]),0)
        self.assertTrue(np.isnan(r2([1],[1])))
        self.assertEqual(precision_recall_f1([0,1],[0,0]),(0,0,0))
        for fn,args in [(rmse,([],[])),(roc_auc,([0,0],[1,2])),(confusion_matrix,([2],[0]))]:
            with self.assertRaises(ValueError):fn(*args)

if __name__=='__main__': unittest.main()
