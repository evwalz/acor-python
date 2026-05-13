from __future__ import annotations

# ============================================================================
# acor
# ============================================================================

import numpy as np
from scipy.stats import norm, chi2, rankdata

from acor_internals import (
    validate_alternative,
    validate_conf_level,
    validate_inputs,
    validate_method,
    validate_variance_method,
)
from akc_functions import (
    compute_akc,
    compute_akc_multivariate,
    compute_akc_multivariate_variance_auto,
    compute_akc_multivariate_variance_ij,
    compute_akc_variance_auto,
    compute_akc_variance_ij,
)
from agc_functions import (
    compute_agc,
    compute_agc_multivariate,
    compute_agc_multivariate_variance_auto,
    compute_agc_multivariate_variance_ij,
    compute_agc_variance_auto,
    compute_agc_variance_ij,
)

class AcorResult:
    """Result of the acor function."""

    def __init__(self, statistic: np.ndarray, method: str):
        self.statistic = statistic
        self.method = method

class AcorTestResult:
    """Result of the acor.test function."""

    def __init__(
        self,
        statistic: np.ndarray | float,
        pvalue: np.ndarray | float,
        method: str,
        alternative: str,
        conf_level: float,
        **kwargs,
    ):
        self.statistic = statistic
        self.pvalue = pvalue
        self.method = method
        self.alternative = alternative
        self.conf_level = conf_level
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return (
            f"AcorTestResult(statistic={self.statistic}, pvalue={self.pvalue}, "
            f"method={self.method}, alternative={self.alternative}, "
            f"conf_level={self.conf_level})"
        )

def _normal_pvalue(z_values: np.ndarray, alternative: str) -> np.ndarray:
    if alternative == "two.sided":
        return 2.0 * (1.0 - norm.cdf(np.abs(z_values)))
    if alternative == "greater":
        return 1.0 - norm.cdf(z_values)
    return norm.cdf(z_values)


def _build_contrast_matrix(m: int) -> np.ndarray:
    n_pairs = m * (m - 1) // 2
    l_matrix = np.zeros((n_pairs, m), dtype=float)
    row_idx = 0
    for i in range(m - 1):
        for j in range(i + 1, m):
            l_matrix[row_idx, i] = 1.0
            l_matrix[row_idx, j] = -1.0
            row_idx += 1
    return l_matrix


def _fisher_confidence_interval(
    estimate: float,
    se: float,
    z_alpha: float,
    method: str,
) -> np.ndarray:
    """Compute Fisher-transformed CI, respecting method scale."""
    if method in {"cid", "cma"}:
        est_transformed = 2.0 * estimate - 1.0
        se_transformed = 2.0 * se
    else:
        est_transformed = estimate
        se_transformed = se

    est_transformed = np.clip(est_transformed, -1.0 + 1e-10, 1.0 - 1e-10)
    z_est = np.arctanh(est_transformed)
    deriv = 1.0 / (1.0 - est_transformed**2)
    se_z = se_transformed * abs(deriv)
    z_lower = z_est - z_alpha * se_z
    z_upper = z_est + z_alpha * se_z
    ci_transformed = np.array([np.tanh(z_lower), np.tanh(z_upper)], dtype=float)

    if method in {"cid", "cma"}:
        return (ci_transformed + 1.0) / 2.0
    return ci_transformed


def acor(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "akc",
) -> AcorResult:
    """Compute an asymmetric correlation measure.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, m)
        Predictor variable(s).
    y : array_like, shape (n,)
        Outcome variable.
    method : {"akc", "agc", "cid", "cma"}
        Which measure to compute.

    Returns
    -------
    AcorResult
        ``.statistic`` holds the correlation estimate(s),
        ``.method`` holds the method name.
    """
    method = validate_method(method)
    x, y, n, m = validate_inputs(x, y)

    if method in ("akc", "cid"):
        if m == 1:
            akc_val = compute_akc(x[:, 0], y)
            estimates = (akc_val + 1) / 2 if method == "cid" else akc_val
        else:
            akcs = compute_akc_multivariate(x, y)
            estimates = (akcs + 1) / 2 if method == "cid" else akcs
    else:  # agc / cma
        y_ranks = rankdata(y, method="average")
        if m == 1:
            x_ranks = rankdata(x[:, 0], method="average")
            agc_val = compute_agc(y_ranks, x_ranks)
            estimates = (agc_val + 1) / 2 if method == "cma" else agc_val
        else:
            xarray_ranks = np.empty((n, m))
            for j in range(m):
                xarray_ranks[:, j] = rankdata(x[:, j], method="average")
            agcs = compute_agc_multivariate(y_ranks, xarray_ranks)
            estimates = (agcs + 1) / 2 if method == "cma" else agcs

    return AcorResult(statistic=np.asarray(estimates), method=method)


def acor_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "akc",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
    variance: str = "ij",
) -> AcorTestResult:
    """Statistical test for currently supported methods.

    Backend variance/covariance routines are placeholders and are expected
    to be implemented later.
    """
    method = validate_method(method)
    alternative = validate_alternative(alternative)
    conf_level = validate_conf_level(conf_level)
    variance_method = validate_variance_method(variance)
    if not isinstance(iid, bool):
        raise ValueError("iid must be a boolean value.")
    if not isinstance(fisher, bool):
        raise ValueError("fisher must be a boolean value.")
    x, y, n, m = validate_inputs(x, y)
    if n < 3:
        raise ValueError("At least 3 observations are required.")

    if method in {"akc", "cid"}:
        if m == 1:
            if variance_method == "ij":
                result = compute_akc_variance_ij(x[:, 0], y, iid=iid)
            else:
                result = compute_akc_variance_auto(x[:, 0], y, iid=iid)
            base_estimates = result["akc"]
            variance = result["var"]
            variance_ind = result["var_ind"]
        else:
            if variance_method == "ij":
                result = compute_akc_multivariate_variance_ij(x, y, iid=iid)
            else:
                result = compute_akc_multivariate_variance_auto(x, y, iid=iid)
            base_estimates = result["akc_vector"]
            variance = result["Sigma"]
            variance_ind = result["Sigma_ind"]
        if method == "cid":
            estimates = (np.asarray(base_estimates, dtype=float) + 1.0) / 2.0
            variance = np.asarray(variance, dtype=float) / 4.0
            variance_ind = (
                None if variance_ind is None else np.asarray(variance_ind, dtype=float) / 4.0
            )
        else:
            estimates = np.asarray(base_estimates, dtype=float)
            variance = np.asarray(variance, dtype=float)
            variance_ind = (
                None if variance_ind is None else np.asarray(variance_ind, dtype=float)
            )
    else:  # agc / cma
        y_ranks = rankdata(y, method="average")
        if m == 1:
            x_ranks = rankdata(x[:, 0], method="average")
            if variance_method == "ij":
                result = compute_agc_variance_ij(y_ranks, x_ranks, iid=iid)
            else:
                result = compute_agc_variance_auto(y_ranks, x_ranks, iid=iid)
            base_estimates = result["agc"]
            variance = result["var"]
            variance_ind = result["var_ind"]
        else:
            xarray_ranks = np.empty((n, m))
            for j in range(m):
                xarray_ranks[:, j] = rankdata(x[:, j], method="average")
            if variance_method == "ij":
                result = compute_agc_multivariate_variance_ij(y_ranks, xarray_ranks, iid=iid)
            else:
                result = compute_agc_multivariate_variance_auto(y_ranks, xarray_ranks, iid=iid)
            base_estimates = result["agc_vector"]
            variance = result["Sigma"]
            variance_ind = result["Sigma_ind"]
        if method == "cma":
            estimates = (np.asarray(base_estimates, dtype=float) + 1.0) / 2.0
            variance = np.asarray(variance, dtype=float) / 4.0
            variance_ind = (
                None if variance_ind is None else np.asarray(variance_ind, dtype=float) / 4.0
            )
        else:
            estimates = np.asarray(base_estimates, dtype=float)
            variance = np.asarray(variance, dtype=float)
            variance_ind = (
                None if variance_ind is None else np.asarray(variance_ind, dtype=float)
            )

    estimates = np.atleast_1d(estimates)
    variance = np.atleast_2d(variance)
    variance_ind = None if variance_ind is None else np.atleast_2d(variance_ind)
    null_value = 0.0 if method in {"akc", "agc"} else 0.5

    alpha = 1.0 - conf_level
    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    if m == 1:
        var = float(variance[0, 0])
        se = np.sqrt(max(var, np.finfo(float).eps) / n)
        estimate = float(estimates[0])
        z_value = (estimate - null_value) / se
        p_value = float(_normal_pvalue(np.array([z_value]), alternative)[0])
        if fisher:
            ci = _fisher_confidence_interval(estimate, se, z_alpha, method)
        else:
            ci = np.array([estimate - z_alpha * se, estimate + z_alpha * se], dtype=float)

        z_value_ind = None
        p_value_ind = None
        var_ind = None
        if variance_ind is not None:
            var_ind = float(variance_ind[0, 0])
            se_ind = np.sqrt(max(var_ind, np.finfo(float).eps) / n)
            z_value_ind = (estimate - null_value) / se_ind
            p_value_ind = float(_normal_pvalue(np.array([z_value_ind]), alternative)[0])

        return AcorTestResult(
            statistic=float(z_value),
            pvalue=p_value,
            method=method,
            alternative=alternative,
            conf_level=conf_level,
            statistic_ind=None if z_value_ind is None else float(z_value_ind),
            pvalue_ind=p_value_ind,
            estimate=estimate,
            variance=var,
            variance_ind=var_ind,
            null_value=null_value,
            conf_int=ci,
            iid=iid,
            fisher=fisher,
            variance_method=variance_method,
        )

    l_matrix = _build_contrast_matrix(m)
    est_diff = l_matrix @ estimates

    l_s_lt = l_matrix @ (variance / n) @ l_matrix.T
    l_s_lt_inv = np.linalg.pinv(l_s_lt)
    chi_sq_stat = float(est_diff.T @ l_s_lt_inv @ est_diff)
    df = int(np.linalg.matrix_rank(l_s_lt))
    p_value = float(chi2.sf(chi_sq_stat, df=df))
    se_individual = np.sqrt(np.maximum(np.diag(variance), np.finfo(float).eps) / n)
    z_individual = (estimates - null_value) / se_individual
    p_individual = _normal_pvalue(z_individual, alternative)
    ci_lower = estimates - z_alpha * se_individual
    ci_upper = estimates + z_alpha * se_individual
    if fisher:
        ci_pairs = np.array(
            [
                _fisher_confidence_interval(
                    float(estimates[i]),
                    float(se_individual[i]),
                    z_alpha,
                    method,
                )
                for i in range(m)
            ],
            dtype=float,
        )
        ci_lower = ci_pairs[:, 0]
        ci_upper = ci_pairs[:, 1]

    chi_sq_stat_ind = None
    df_ind = None
    p_value_ind = None
    z_individual_ind = [None] * m
    p_individual_ind = [None] * m
    if variance_ind is not None:
        l_s_lt_ind = l_matrix @ (variance_ind / n) @ l_matrix.T
        l_s_lt_inv_ind = np.linalg.pinv(l_s_lt_ind)
        chi_sq_stat_ind = float(est_diff.T @ l_s_lt_inv_ind @ est_diff)
        df_ind = int(np.linalg.matrix_rank(l_s_lt_ind))
        p_value_ind = float(chi2.sf(chi_sq_stat_ind, df=df_ind))
        se_individual_ind = np.sqrt(
            np.maximum(np.diag(variance_ind), np.finfo(float).eps) / n
        )
        z_individual_ind = (estimates - null_value) / se_individual_ind
        p_individual_ind = _normal_pvalue(z_individual_ind, alternative)

    pair_labels = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            pair_labels.append(f"X{i + 1} - X{j + 1}")
    se_diff = np.sqrt(np.maximum(np.diag(l_s_lt), np.finfo(float).eps))
    z_diff = est_diff / se_diff
    p_diff = _normal_pvalue(z_diff, alternative)

    results = [
        {
            "predictor": f"X{i + 1}",
            "estimate": float(estimates[i]),
            "statistic": float(z_individual[i]),
            "statistic_ind": None if z_individual_ind[i] is None else float(z_individual_ind[i]),
            "pvalue": float(p_individual[i]),
            "pvalue_ind": None if p_individual_ind[i] is None else float(p_individual_ind[i]),
            "ci_lower": float(ci_lower[i]),
            "ci_upper": float(ci_upper[i]),
        }
        for i in range(m)
    ]
    pairwise_results = [
        {
            "pair": pair_labels[i],
            "difference": float(est_diff[i]),
            "statistic": float(z_diff[i]),
            "pvalue": float(p_diff[i]),
            "ci_lower": float(est_diff[i] - z_alpha * se_diff[i]),
            "ci_upper": float(est_diff[i] + z_alpha * se_diff[i]),
        }
        for i in range(len(pair_labels))
    ]

    return AcorTestResult(
        statistic=chi_sq_stat,
        pvalue=p_value,
        method=method,
        alternative=alternative,
        conf_level=conf_level,
        statistic_ind=chi_sq_stat_ind,
        pvalue_ind=p_value_ind,
        df=df,
        df_ind=df_ind,
        estimate=estimates,
        variance=variance,
        variance_ind=variance_ind,
        null_value=null_value,
        pairwise_differences=est_diff,
        contrast_matrix=l_matrix,
        results=results,
        pairwise_results=pairwise_results,
        iid=iid,
        fisher=fisher,
        variance_method=variance_method,
    )
