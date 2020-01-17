from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import inspect
from util import (
    assert_inputs_are_valid,
    value_counts,
)
from latent_estimation import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
)
from latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from pruning import get_noise_indices


class LearningWithNoisyLabels(BaseEstimator): # Inherits sklearn classifier
    '''Rank Pruning is a state-of-the-art algorithm (2017) for
      multiclass classification with (potentially extreme) mislabeling 
      across any or all pairs of class labels. It works with ANY classifier,
      including deep neural networks. See clf parameter.
    This subfield of machine learning is referred to as Confident Learning.
    Rank Pruning also achieves state-of-the-art performance for binary
      classification with noisy labels and positive-unlabeled
      learning (PU learning) where a subset of positive examples is given and
      all other examples are unlabeled and assumed to be negative examples.
    Rank Pruning works by "learning from confident examples." Confident examples are
      identified as examples with high predicted probability for their training label.
    Given any classifier having the predict_proba() method, an input feature matrix, X, 
      and a discrete vector of labels, s, which may contain mislabeling, Rank Pruning
      estimates the classifications that would be obtained if the hidden, true labels, y,
      had instead been provided to the classifier during training.
    "s" denotes the noisy label instead of \tilde(y), for ASCII encoding reasons.

    Parameters
    ----------
    clf : sklearn.classifier compliant class (e.g. skorch wraps around PyTorch)
      See cleanlab.models for examples of sklearn wrappers around FastText, PyTorch, etc.
      The clf object must have the following three functions defined:
      1. clf.predict_proba(X) # Predicted probabilities
      2. clf.predict(X) # Predict labels
      3. clf.fit(X, y, sample_weight) # Train classifier
      Stores the classifier used in Rank Pruning.
      Default classifier used is logistic regression.

    seed : int (default = None)
      Number to set the default state of the random number generator used to split
      the cross-validated folds. If None, uses np.random current random state.

    cv_n_folds : int
      This class needs holdout predicted probabilities for every training example
      and f not provided, uses cross-validation to compute them. cv_n_folds sets
      the number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    prune_method : str (default: 'prune_by_noise_rate')
      'prune_by_class', 'prune_by_noise_rate', or 'both'. Method used for pruning.
      1. 'prune_by_noise_rate': works by removing examples with *high probability* of
      being mislabeled for every non-diagonal in the prune_counts_matrix (see pruning.py).
      2. 'prune_by_class': works by removing the examples with *smallest probability* of
      belonging to their given class label for every class.
      3. 'both': Finds the examples satisfying (1) AND (2) and removes their set conjunction.

    converge_latent_estimates : bool (Default: False)
      If true, forces numerical consistency of latent estimates. Each is estimated
      independently, but they are related mathematically with closed form 
      equivalences. This will iteratively enforce mathematically consistency.

    pulearning : int (0 or 1, default: None)
      Only works for 2 class datasets. Set to the integer of the class that is
      perfectly labeled (certain no errors in that class). If unsure, set to None.''' 
  
  
    def __init__(
        self,
        clf = None,
        seed = None,
        # Hyper-parameters (used by .fit() function)
        cv_n_folds = 5,
        prune_method = 'prune_by_noise_rate',
        converge_latent_estimates = False,
        pulearning = None,
    ):

        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = logreg(multi_class = 'auto', solver = 'lbfgs')

        # Make sure the passed in classifier has the appropriate methods defined.
        if not hasattr(clf, "fit"):
            raise ValueError('The classifier (clf) must define a .fit() method.')
        if not hasattr(clf, "predict_proba"):
            raise ValueError('The classifier (clf) must define a .predict_proba() method.')
        if not hasattr(clf, "predict"):
            raise ValueError('The classifier (clf) must define a .predict() method.')

        if seed is not None:
            np.random.seed(seed = seed)
        
        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.prune_method = prune_method
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
  
  
    def fit(
        self, 
        X,
        s,
        psx = None,
        thresholds = None,
        noise_matrix = None,
        inverse_noise_matrix = None, 
    ):
        # Check inputs
        assert_inputs_are_valid(X, s, psx)
        #没用
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and np.trace(inverse_noise_matrix) <= 1:
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}, but must exceed 1.".format(t))

        # Number of classes
        self.K = len(np.unique(s))

        # 'ps' is p(s=k)
        self.ps = value_counts(s) / float(len(s))

        self.confident_joint = None
        # If needed, compute noise rates (fraction of mislabeling) for all classes. 
        # Also, if needed, compute P(s=k|x), denoted psx.
        
        # Set / re-set noise matrices / psx; estimate if not provided.
        if noise_matrix is not None:
            self.noise_matrix = noise_matrix
            if inverse_noise_matrix is None:
                self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(self.ps, self.noise_matrix)
        if inverse_noise_matrix is not None:
            self.inverse_noise_matrix = inverse_noise_matrix
            if noise_matrix is None:
                self.noise_matrix = compute_noise_matrix_from_inverse(self.ps, self.inverse_noise_matrix)
        if noise_matrix is None and inverse_noise_matrix is None:
            if psx is None:
                self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint, psx = \
                estimate_py_noise_matrices_and_cv_pred_proba(
                    X = X,
                    s = s,
                    clf = self.clf,
                    cv_n_folds = self.cv_n_folds,
                    thresholds = thresholds,
                    converge_latent_estimates = self.converge_latent_estimates,
                    seed = self.seed,
                )
            else: # 只执行这里psx is provided by user (assumed holdout probabilities)
                self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint = \
                estimate_py_and_noise_matrices_from_probabilities(
                    s = s, 
                    psx = psx,
                    thresholds = thresholds,
                    converge_latent_estimates = self.converge_latent_estimates,
                )

        #没用
        if psx is None: 
            psx = estimate_cv_predicted_probabilities(
                X = X,
                labels = s,
                clf = self.clf,
                cv_n_folds = self.cv_n_folds,
                seed = self.seed,
            )

        # 没用if pulearning == the integer specifying the class without noise.
        if self.K == 2 and self.pulearning is not None: # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            self.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        self.noise_mask = get_noise_indices(
            s,
            psx,
            inverse_noise_matrix = self.inverse_noise_matrix,
            confident_joint = self.confident_joint,
            prune_method = self.prune_method,
        ) 

#在剪枝算法中，将noise_mask是噪声样本下标，去掉噪声样本下标得到X_mask,只保留X和s中的正确样本下标
        # X_mask = ~self.noise_mask
        # X_pruned = X[X_mask]
        # s_pruned = s[X_mask]

#在重标注算法中
        psx_cha=psx[self.noise_mask]
        trueindex=np.where(self.noise_mask==True)
        trueindex=trueindex[0].tolist()
        x=0
        for i in psx_cha:
          index=trueindex[x]
          maxindex=np.argsort(-i)[0]
          if maxindex==s[index]:
            s[index]=np.argsort(-i)[1]
          else:
            s[index]=np.argsort(-i)[0]
          x+=1
        X_pruned=X
        s_pruned=s
        relabel_s=pd.DataFrame({'relabel_s':s})
        relabel_s.to_csv('relabel.csv',index=None)



        # Check if sample_weight in clf.fit(). Compatible with Python 2/3.
        if hasattr(inspect, 'getfullargspec') and \
                'sample_weight' in inspect.getfullargspec(self.clf.fit).args or \
                hasattr(inspect, 'getargspec') and \
                'sample_weight' in inspect.getargspec(self.clf.fit).args:
            # Re-weight examples in the loss function for the final fitting
            # s.t. the "apparent" original number of examples in each class
            # is preserved, even though the pruned sets may differ.
            self.sample_weight = np.ones(np.shape(s_pruned))
            for k in range(self.K): 
                self.sample_weight[s_pruned == k] = 1.0 / self.noise_matrix[k][k]

            self.clf.fit(X_pruned, s_pruned, sample_weight=self.sample_weight)
        else:
            # This is less accurate, but its all we can do if sample_weight isn't available.
            self.clf.fit(X_pruned, s_pruned)
            
        return self.clf


    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)


    def predict_proba(self, *args, **kwargs):
        return self.clf.predict_proba(*args, **kwargs)


    def score(self, X, y, sample_weight=None):    
        if hasattr(self.clf, 'score'):
        
            # Check if sample_weight in clf.score(). Compatible with Python 2/3.
            if hasattr(inspect, 'getfullargspec') and \
                    'sample_weight' in inspect.getfullargspec(self.clf.score).args or \
                    hasattr(inspect, 'getargspec') and \
                    'sample_weight' in inspect.getargspec(self.clf.score).args:
                return self.clf.score(X, y, sample_weight=sample_weight)
            else:
                return self.clf.score(X, y)
        else:
            return accuracy_score(y, self.clf.predict(X), sample_weight=sample_weight)
