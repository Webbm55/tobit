"""Maximum-likelihood Tobit regression for censored data.

This module implements Tobit regression (censored regression) using maximum
likelihood estimation. It handles left-censored, right-censored, and uncensored
observations to model dependent variables with censoring limits.

Typical usage example:

  model = TobitModel(fit_intercept=True)
  model.fit(X, y, censoring_indicators)
  predictions = model.predict(X_test)
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats
from scipy.special import log_ndtr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def split_left_right_censored(x, y, cens):
    """Partitions data into left-censored, uncensored, and right-censored subsets.

    Args:
        x: DataFrame of predictor variables.
        y: Series of target variable values.
        cens: Series with censoring indicators (-1: left, 0: none, 1: right).

    Returns:
        Tuple of (xs, ys) where each is a list of [left, middle, right] arrays.
        Elements are None if no observations exist for that censoring type.

    Warns:
        UserWarning: If no censored observations exist in the data.
    """
    counts = cens.value_counts()
    if -1 not in counts and 1 not in counts:
        warnings.warn("No censored observations; use regression methods for uncensored data")
    xs = []
    ys = []

    for value in [-1, 0, 1]:
        if value in counts:
            split = cens == value
            y_split = np.squeeze(y[split].values)
            x_split = x[split].values

        else:
            y_split, x_split = None, None
        xs.append(x_split)
        ys.append(y_split)
    return xs, ys


def tobit_neg_log_likelihood(xs, ys, params):
    """Computes negative log-likelihood for Tobit model.

    Uses censored normal distribution for censored observations and standard
    normal density for uncensored observations.

    Args:
        xs: List of [left, middle, right] predictor arrays from split_left_right_censored.
        ys: List of [left, middle, right] target arrays from split_left_right_censored.
        params: Array of model parameters [coefficients..., sigma].

    Returns:
        Negative log-likelihood value for optimization.
    """
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    to_cat = []

    cens = False
    if y_left is not None:
        cens = True
        left = (y_left - np.dot(x_left, b))
        to_cat.append(left)
    if y_right is not None:
        cens = True
        right = (np.dot(x_right, b) - y_right)
        to_cat.append(right)
    if cens:
        concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = scipy.stats.norm.logcdf(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        # Prevent log(0) by enforcing minimum sigma value
        mid = scipy.stats.norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
        mid_sum = mid.sum()
    else:
        mid_sum = 0

    loglik = cens_sum + mid_sum

    return - loglik


def tobit_neg_log_likelihood_der(xs, ys, params):
    """Computes gradient of negative log-likelihood for Tobit model.

    Calculates partial derivatives with respect to coefficients and sigma
    for use in gradient-based optimization.

    Args:
        xs: List of [left, middle, right] predictor arrays from split_left_right_censored.
        ys: List of [left, middle, right] target arrays from split_left_right_censored.
        params: Array of model parameters [coefficients..., sigma].

    Returns:
        Array of gradient values [d/dcoef1, ..., d/dsigma].
    """
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = scipy.stats.norm.logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        # Inverse Mills ratio for left-censored observations
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        # Inverse Mills ratio for right-censored observations
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma

    combo_jac = np.append(beta_jac, sigma_jac / s)

    return -combo_jac


class TobitModel:
    """Maximum-likelihood estimator for Tobit (censored) regression.

    Fits a regression model for data with left-censored, right-censored, or
    uncensored observations using BFGS optimization of the log-likelihood.
    Initializes parameters using OLS on all observations.

    Attributes:
        fit_intercept: Whether to fit an intercept term.
        ols_coef_: OLS coefficients used for initialization.
        ols_intercept: OLS intercept used for initialization.
        coef_: Fitted Tobit coefficients (excluding intercept).
        intercept_: Fitted Tobit intercept.
        sigma_: Fitted standard deviation of residuals.
    """

    def __init__(self, fit_intercept=True):
        """Initializes TobitModel with intercept configuration.

        Args:
            fit_intercept: If True, fits intercept term. If False, assumes
                data is centered.
        """
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None

    def fit(self, x, y, cens, verbose=False):
        """Fits maximum-likelihood Tobit regression model.

        Uses OLS for initialization, then optimizes log-likelihood with BFGS.
        Handles left-censored (-1), uncensored (0), and right-censored (1) data.

        Args:
            x: DataFrame (n_samples, n_features) of predictor variables.
            y: Series (n_samples,) of target values.
            cens: Series (n_samples,) of censoring indicators (-1, 0, or 1).
            verbose: If True, displays optimization details.

        Returns:
            Self with fitted parameters (coef_, intercept_, sigma_).
        """
        x_copy = x.copy()
        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
        else:
            x_copy.scale(with_mean=True, with_std=False, copy=False)
        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        xs, ys = split_left_right_censored(x_copy, y, cens)

        result = minimize(lambda params: tobit_neg_log_likelihood(xs, ys, params), params0, method='BFGS',
                          jac=lambda params: tobit_neg_log_likelihood_der(xs, ys, params), options={'disp': verbose})
        if verbose:
            print(result)
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:-1]
        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = result.x[-1]
        return self

    def predict(self, x):
        """Predicts target values using fitted coefficients.

        Args:
            x: DataFrame (n_samples, n_features) of predictor variables.

        Returns:
            Array of predicted values.
        """
        return self.intercept_ + np.dot(x, self.coef_)

    def score(self, x, y, scoring_function=mean_absolute_error):
        """Evaluates model predictions using specified metric.

        Args:
            x: DataFrame (n_samples, n_features) of predictor variables.
            y: Series (n_samples,) of true target values.
            scoring_function: Metric function (default: mean_absolute_error).

        Returns:
            Score computed by scoring_function(y_true, y_pred).
        """
        y_pred = np.dot(x, self.coef_)
        return scoring_function(y, y_pred)
