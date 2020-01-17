
# coding: utf-8

# ## Latent Estimation
# 
# #### Contains methods for estimating four latent structures used for confident learning.
# * The latent prior of the unobserved, errorless labels $y$: denoted $p(y)$ (latex) & '```py```' (code).
# * The latent noisy channel (noise matrix) characterizing the flipping rates: denoted $P_{s \vert y }$ (latex) & '```nm```' (code).
# * The latent inverse noise matrix characterizing flipping process: denoted $P_{y \vert s}$ (latex) & '```inv```' (code).
# * The latent ```confident_joint```, an unnormalized counts matrix of counting a confident subset of the joint counts of label errors.


from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement
)
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import warnings

from util import (
    value_counts, clip_values, clip_noise_rates, round_preserving_row_totals,
    assert_inputs_are_valid,
)
from latent_algebra import (
    compute_inv_noise_matrix, compute_py, compute_noise_matrix_from_inverse
)


def num_label_errors(
    labels,
    psx,
    confident_joint=None,
):


    if confident_joint is None:
        confident_joint = compute_confident_joint(s=labels, psx=psx)
    # Normalize confident joint so that it estimates the joint, p(s,y)
    joint = confident_joint / float(np.sum(confident_joint))
    frac_errors = 1. - joint.trace()
    num_errors = int(frac_errors * len(labels))

    return num_errors


def calibrate_confident_joint(confident_joint, s, multi_label=False):

    if multi_label:
        s_counts = value_counts([x for l in s for x in l])
    else:
        s_counts = value_counts(s)
    # Calibrate confident joint to have correct p(s) prior on noisy labels.
    calibrated_cj = (confident_joint.T / confident_joint.sum(axis=1) * s_counts).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = calibrated_cj / np.sum(calibrated_cj) * sum(s_counts)
    return round_preserving_row_totals(calibrated_cj)


def estimate_joint(s, psx=None, confident_joint=None, multi_label=False):

    
    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            s,
            psx,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        calibrated_cj = calibrate_confident_joint(confident_joint, s)

    return calibrated_cj / float(np.sum(calibrated_cj))


def _compute_confident_joint_multi_label(
    labels,
    psx,
    thresholds=None,
    calibrate=True,
):

    # Compute unique number of classes K by flattening labels (list of lists)
    K = len(np.unique([i for l in labels for i in l]))
    # Compute thresholds = p(s=k | k in set of given labels)
    # This is the average probability of class given that the label is represented.
    k_in_l = np.array([[k in l for l in labels] for k in range(K)])
    thresholds = [np.mean(psx[:,k][k_in_l[k]]) for k in range(K)]
    # Create mask for every example if for each class, prob >= threshold
    psx_bool = psx >= thresholds
    # Compute confident joint
    # (no need to avoid collisions for multi-label, double counting is okay!)
    confident_joint = np.array([psx_bool[k_in_l[k]].sum(axis = 0) for k in range(K)])
    if calibrate:
        return calibrate_confident_joint(confident_joint, labels, multi_label=True)

    return confident_joint


def compute_confident_joint(
    s,
    psx,
    K=None,
    thresholds=None,
    calibrate=True,
    multi_label=False,
    return_indices_of_off_diagonals=False,
):
    if multi_label:
        return _compute_confident_joint_multi_label(
            labels=s,
            psx=psx,
            thresholds=thresholds,
            calibrate=calibrate,
        )

    # s needs to be a numpy array
    s = np.asarray(s)

    # Find the number of unique classes if K is not given
    if K is None:
        K = len(np.unique(s))

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
    thresholds = np.asarray(thresholds)

    # The following code computes the confident joint.
    # The code is optimized with vectorized functions.
    # For ease of understanding, here is (a slow) implementation with for loops.
    #     confident_joint = np.zeros((K, K), dtype = int)
    #     for i, row in enumerate(psx):
    #         s_label = s[i]
    #         confident_bins = row >= thresholds - 1e-6
    #         num_confident_bins = sum(confident_bins)
    #         if num_confident_bins == 1:
    #             confident_joint[s_label][np.argmax(confident_bins)] += 1
    #         elif num_confident_bins > 1:
    #             confident_joint[s_label][np.argmax(row)] += 1

    # Compute confident joint (vectorized for speed).

    # psx_bool is a bool matrix where each row represents a training example as
    # a boolean vector of size K, with True if the example confidentally belongs
    # to that class and False if not.
    psx_bool = (psx >= thresholds - 1e-6)
    num_confident_bins = psx_bool.sum(axis = 1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    psx_argmax = psx.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = psx_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is more than one confident class, choose the class with largest prob.
    true_label_guess = np.where(more_than_one_confident, psx_argmax, confident_argmax)
    y_confident = true_label_guess[at_least_one_confident] # Omits meaningless all-False rows
    s_confident = s[at_least_one_confident]
    confident_joint = confusion_matrix(y_confident, s_confident).T
    
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, s)

    if return_indices_of_off_diagonals:
        indices = np.arange(len(s))[at_least_one_confident][y_confident != s_confident]

        return confident_joint, indices

    return confident_joint


def estimate_latent(
    confident_joint,
    s,
    py_method='cnt',
    converge_latent_estimates=False,
):
    # Number of classes
    K = len(np.unique(s))
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))
    # Ensure labels are of type np.array()
    s = np.asarray(s)
    # Number of training examples confidently counted from each noisy class
    s_count = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    y_count = confident_joint.sum(axis=0).astype(float)
    # Confident Counts Estimator for p(s=k_s|y=k_y) ~ |s=k_s and y=k_y| / |y=k_y|
    noise_matrix = confident_joint / y_count
    # Confident Counts Estimator for p(y=k_y|s=k_s) ~ |y=k_y and s=k_s| / |s=k_s|
    inv_noise_matrix = confident_joint.T / s_count
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(ps, noise_matrix, inv_noise_matrix, py_method, y_count)
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = converge_estimates(ps, py, noise_matrix, inv_noise_matrix)
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix


def estimate_py_and_noise_matrices_from_probabilities(
    s,
    psx,
    thresholds=None,
    converge_latent_estimates=True,
    py_method='cnt',
    calibrate=True,
):
    confident_joint = compute_confident_joint(
        s=s,
        psx=psx,
        thresholds=thresholds,
        calibrate=calibrate,
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=s,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint


def estimate_confident_joint_and_cv_pred_proba(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
):
    assert_inputs_are_valid(X, s)
    # Number of classes
    K = len(np.unique(s))
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))

    # Ensure labels are of type np.array()
    s = np.asarray(s)

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Intialize psx array
    psx = np.zeros((len(s), K))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, s)):

        clf_copy = copy.deepcopy(clf)

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        clf_copy.fit(X_train_cv, s_train_cv)
        psx_cv = clf_copy.predict_proba(X_holdout_cv) # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv

    # Compute the confident counts of all pairwise label-flipping mislabeling rates.
    confident_joint = compute_confident_joint(
        s=s,
        psx=psx, # P(s = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )

    return confident_joint, psx


def estimate_py_noise_matrices_and_cv_pred_proba(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=False,
    py_method='cnt',
    seed=None,
):
    print("hello2")
    confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
        X=X,
        s=s,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
    )
    print("hello3")
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=s,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )
    print("hello4")

    return py, noise_matrix, inv_noise_matrix, confident_joint, psx


def estimate_cv_predicted_probabilities(
    X,
    labels, # class labels can be noisy (s) or not noisy (y).
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    seed=None,
):
    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        s=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        seed=seed,
    )[-1]


def estimate_noise_matrices(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=True,
    seed=None,
):

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        s=s,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        converge_latent_estimates=converge_latent_estimates,
        seed=seed,
    )[1:-2]


def converge_estimates(
    ps,
    py,
    noise_matrix,
    inverse_noise_matrix,
    inv_noise_matrix_iterations=5,
    noise_matrix_iterations=3,
):

    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(py, noise_matrix, ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, py)

    return py, noise_matrix, inverse_noise_matrix
