import unittest
import numpy as np
from sklearn.linear_model import SGDClassifier
from src.svm import PegasosSVM, RandomFourierFeatures


class SVMTests(unittest.TestCase):
    def test_single_step_and_unregularized_bias(self):
        X=np.array([[-2.], [1.]])
        m=PegasosSVM(lambda_=1., n_epochs=1, batch_size=2, average=False).fit(X,[0,1])
        np.testing.assert_allclose(m.w,[1.])  # gradient 1.5, projection radius 1
        self.assertEqual(m.b,0.)
        self.assertAlmostEqual(m.objective_,.5)

    def test_objective_known_value(self):
        self.assertAlmostEqual(PegasosSVM.objective(np.array([[1.],[2.]]),
            np.array([-1,1]),np.array([.5]),.2,.1),.8625)

    def test_separable_and_baseline_agreement(self):
        rng=np.random.default_rng(3)
        X=np.r_[rng.normal(-2,.25,(80,2)),rng.normal(2,.25,(80,2))]
        y=np.r_[np.zeros(80),np.ones(80)]
        m=PegasosSVM(lambda_=.01,n_epochs=60,batch_size=32).fit(X,y)
        b=SGDClassifier(loss='hinge',alpha=.01,random_state=42).fit(X,y)
        np.testing.assert_array_equal(m.predict(X),b.predict(X))
        self.assertLess(m.objective_,m.history_[0]['objective'])
        self.assertLessEqual(np.linalg.norm(m.w),10+1e-10)
        self.assertAlmostEqual(m.margin_width_,2/np.linalg.norm(m.w))
        np.testing.assert_array_equal(m.support_mask_,(2*y-1)*m.decision_function(X)<=1)

    def test_rff_xor(self):
        X=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]]),(40,1))
        y=np.tile(np.array([0,1,1,0]),40)
        phi=RandomFourierFeatures(gamma=.5,n_components=256).fit_transform(X)
        m=PegasosSVM(lambda_=.001,n_epochs=50,batch_size=32).fit(phi,y)
        np.testing.assert_array_equal(m.predict(phi),y)

    def test_rff_approximates_rbf(self):
        X=np.array([[0.,0.],[1.,0.],[1.,1.]])
        z=RandomFourierFeatures(gamma=.3,n_components=20000).fit_transform(X)
        expected=np.exp(-.3*np.sum((X[:,None]-X[None,:])**2,axis=2))
        np.testing.assert_allclose(z@z.T,expected,atol=.025)

    def test_labels_refit_and_determinism(self):
        X=np.array([[-2.],[-1.],[1.],[2.]])
        m=PegasosSVM(lambda_=.01,n_epochs=30,batch_size=2).fit(X,['no','no','yes','yes'])
        p=m.decision_function(X).copy();m.fit(X,['no','no','yes','yes'])
        np.testing.assert_array_equal(m.decision_function(X),p)
        np.testing.assert_array_equal(m.predict(X),['no','no','yes','yes'])

    def test_schedule(self):
        self.assertEqual(PegasosSVM(lambda_=.1).learning_rate(2),5)
        self.assertEqual(PegasosSVM(schedule='constant',eta0=.2).learning_rate(4),.2)
        self.assertEqual(PegasosSVM(schedule='inverse_sqrt',eta0=.2).learning_rate(4),.1)

    def test_invalid(self):
        for kw in [dict(lambda_=0),dict(n_epochs=0),dict(batch_size=0),dict(schedule='bad')]:
            with self.assertRaises(ValueError):PegasosSVM(**kw).fit([[-1],[1]],[0,1])
        for X,y in [([[0],[1]],[1,1]),([[np.nan],[1]],[0,1]),([[0],[1]],[0])]:
            with self.assertRaises(ValueError):PegasosSVM().fit(X,y)
        with self.assertRaises(RuntimeError):PegasosSVM().predict([[0]])

if __name__=='__main__':unittest.main()
