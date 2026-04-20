from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from acor_cpp_bindings import (
    akc_ij_cpp as native_akc_ij_cpp,
    h_bar_vec_v2_cpp as native_h_bar_vec_v2_cpp,
    has_native_extension,
    kendall_tau_sign_cpp as native_kendall_tau_sign_cpp,
)


def _is_binary(y: np.ndarray) -> bool:
    return np.unique(y).size == 2


def _tau_y_stats_binary(y: np.ndarray) -> tuple[float, float]:
    """Binary-optimized E[sgn(Y'-Y'')^2] and tie probability in Y."""
    n = len(y)
    unique_vals = np.unique(y)
    n0 = np.sum(y == unique_vals[0])
    n1 = np.sum(y == unique_vals[1])
    num_pairs = n * (n - 1) / 2
    n_ties_y = n0 * (n0 - 1) / 2 + n1 * (n1 - 1) / 2
    p_tie_y = n_ties_y / num_pairs
    expectation = 1.0 - p_tie_y
    return float(expectation), float(p_tie_y)


def _kendall_tau_sign_binary(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Binary-optimized Kendall sign representation.

    Returns
    -------
    tau : float
        AKC estimate on [-1, 1] scale.
    expectation : float
        Raw pair expectation E[sgn(X'-X'')*sgn(Y'-Y'')].
    """
    n = len(x)
    unique_vals = np.sort(np.unique(y))
    mask0 = y == unique_vals[0]
    mask1 = y == unique_vals[1]
    x0 = x[mask0]
    x1_sorted = np.sort(x[mask1])
    n0 = x0.size
    n1 = x1_sorted.size

    # Counts among X1 for each x0.
    count_less = np.searchsorted(x1_sorted, x0, side="left")
    count_leq = np.searchsorted(x1_sorted, x0, side="right")
    count_greater = n1 - count_leq

    concordant = np.sum(count_greater)
    discordant = np.sum(count_less)

    num_pairs = n * (n - 1) / 2
    n_ties_y = n0 * (n0 - 1) / 2 + n1 * (n1 - 1) / 2
    expectation = (concordant - discordant) / num_pairs
    p_tie_y = n_ties_y / num_pairs
    scale_factor = 1.0 - p_tie_y
    tau = expectation / scale_factor if scale_factor > 1e-10 else 0.0
    return float(tau), float(expectation)


def _compute_kendall_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Dispatch Kendall computation to binary or generic path."""
    if _is_binary(y):
        return _kendall_tau_sign_binary(x, y)
    if has_native_extension():
        result = native_kendall_tau_sign_cpp(x, y)
        return float(result["tau"]), float(result["expectation"])

    n = len(x)
    sign_products = []
    n_ties_y = 0
    num_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            sgn_x = np.sign(x[j] - x[i])
            sgn_y = np.sign(y[j] - y[i])
            sign_products.append(sgn_x * sgn_y)
            if y[j] == y[i]:
                n_ties_y += 1
            num_pairs += 1

    expectation = np.mean(sign_products)
    p_tie_y = n_ties_y / num_pairs
    scale_factor = 1.0 - p_tie_y
    tau = expectation / scale_factor if scale_factor > 1e-10 else 0.0
    return float(tau), float(expectation)


def _k_tau_vector(x: np.ndarray, y: np.ndarray, tau_xy: float) -> np.ndarray:
    if (not _is_binary(y)) and has_native_extension():
        h_bar = np.asarray(native_h_bar_vec_v2_cpp(x, y), dtype=float)
        f_bar = (rankdata(x, method="average") - 0.5) / len(x)
        g_bar = (rankdata(y, method="average") - 0.5) / len(y)
        return 4.0 * h_bar - 2.0 * (f_bar + g_bar) + 1.0 - tau_xy
    return np.array([_k_tau(x[i], y[i], x, y, tau_xy) for i in range(len(x))], dtype=float)


# Python AKC functions
def compute_akc(X, Y):
    """
    Kendall's tau using the sign function representation with scaling:
    tau = E(sgn(X' - X'') * sgn(Y' - Y'')) / (1 - P(Y = Y'))
    """
    x = np.asarray(X)
    y = np.asarray(Y)
    tau, _ = _compute_kendall_stats(x, y)
    return tau


def compute_akc_multivariate(X, Y) -> np.ndarray:
    """AKC for a multiple predictors"""
    m = X.shape[1]
    return np.array([compute_akc(X[:,j], Y) for j in range(m)])


def _tau_y_stats(y: np.ndarray) -> tuple[float, float]:
    """Return E[sgn(Y'-Y'')^2] and tie probability in Y."""
    if _is_binary(y):
        return _tau_y_stats_binary(y)
    n = len(y)
    num_pairs = n * (n - 1) / 2
    _, counts = np.unique(y, return_counts=True)
    n_ties_y = np.sum(counts * (counts - 1) / 2)
    p_tie_y = n_ties_y / num_pairs
    expectation = 1.0 - p_tie_y
    return float(expectation), float(p_tie_y)


def _f_bar(x_value: float, x: np.ndarray) -> float:
    return float(np.mean(x < x_value) + 0.5 * np.mean(x == x_value))


def _g_bar(y_value: float, y: np.ndarray) -> float:
    return float(np.mean(y < y_value) + 0.5 * np.mean(y == y_value))


def _h_bar(x_value: float, y_value: float, x: np.ndarray, y: np.ndarray) -> float:
    p_both_less = np.mean((x < x_value) & (y < y_value))
    p_x_equal_y_less = np.mean((x == x_value) & (y < y_value))
    p_x_less_y_equal = np.mean((x < x_value) & (y == y_value))
    p_both_equal = np.mean((x == x_value) & (y == y_value))
    return float(
        p_both_less
        + 0.5 * p_x_equal_y_less
        + 0.5 * p_x_less_y_equal
        + 0.25 * p_both_equal
    )


def _k_tau(x_value: float, y_value: float, x: np.ndarray, y: np.ndarray, tau_xy: float) -> float:
    return 4.0 * _h_bar(x_value, y_value, x, y) - 2.0 * (
        _f_bar(x_value, x) + _g_bar(y_value, y)
    ) + 1.0 - tau_xy


def _k_p(y_value: float, y: np.ndarray, tau_y: float) -> float:
    return float(tau_y - np.mean(y != y_value))


def _k_p_vector(y: np.ndarray, tau_y: float) -> np.ndarray:
    """Vectorized K_p values for all observations."""
    n = len(y)
    unique, counts = np.unique(y, return_counts=True)
    prob_map = {u: c / n for u, c in zip(unique.tolist(), counts.tolist())}
    p_obs = np.array([prob_map[val] for val in y], dtype=float)
    return tau_y - (1.0 - p_obs)


def _hac_bandwidth(n: int) -> int:
    return int(np.floor(2.0 * (n ** (1.0 / 3.0))))


def _hac_correction_univariate(adjusted_k: np.ndarray) -> float:
    n = len(adjusted_k)
    b = _hac_bandwidth(n)
    corr = 0.0
    max_lag = min(b, n - 1)
    for h in range(1, max_lag + 1):
        omega = 1.0 - h / (b + 1.0)
        autocov_h = (2.0 / n) * np.sum(adjusted_k[: n - h] * adjusted_k[h:])
        corr += omega * autocov_h
    return float(corr)


def _hac_correction_multivariate(adjusted_k_matrix: np.ndarray) -> np.ndarray:
    n, m = adjusted_k_matrix.shape
    b = _hac_bandwidth(n)
    sigma_hac = np.zeros((m, m), dtype=float)
    max_lag = min(b, n - 1)
    for h in range(1, max_lag + 1):
        omega = 1.0 - h / (b + 1.0)
        k_lag = adjusted_k_matrix[: n - h, :]
        k_lead = adjusted_k_matrix[h:, :]
        autocov_h = (k_lag.T @ k_lead + k_lead.T @ k_lag) / n
        sigma_hac += omega * autocov_h
    return sigma_hac


def _hac_variance_univariate(adjusted_k: np.ndarray, scale_factor: float) -> float:
    iid_var = scale_factor * np.mean(adjusted_k**2)
    return float(iid_var + scale_factor * _hac_correction_univariate(adjusted_k))


def _hac_covariance_multivariate(adjusted_k_matrix: np.ndarray, scale_factor: float) -> np.ndarray:
    n = adjusted_k_matrix.shape[0]
    sigma_iid = (adjusted_k_matrix.T @ adjusted_k_matrix) / n
    sigma_hac = _hac_correction_multivariate(adjusted_k_matrix)
    return scale_factor * (sigma_iid + sigma_hac)


def _autocov_lags(series: np.ndarray, lag_max: int) -> np.ndarray:
    n = len(series)
    out = np.empty(lag_max + 1, dtype=float)
    for h in range(lag_max + 1):
        # Match R stats::acf(..., type="covariance", demean=FALSE):
        # denominator is n (not n-h).
        out[h] = np.sum(series[: n - h] * series[h:]) / n
    return out


def _ind_lrv_univariate(x_grade_centered: np.ndarray, y_grade_centered: np.ndarray, n: int, b: int) -> float:
    x_autoc = _autocov_lags(x_grade_centered, n - 1)
    y_autoc = _autocov_lags(y_grade_centered, n - 1)
    out = x_autoc[0] * y_autoc[0]
    max_lag = min(b, n - 1)
    for h in range(1, max_lag + 1):
        w = max(1.0 - abs(h) / (b + 1.0), 0.0)
        out += 2.0 * w * x_autoc[h] * y_autoc[h]
    return float(out)


def _ind_lrv_multivariate(
    x_grades_centered: np.ndarray,
    y_grade_centered: np.ndarray,
    n: int,
    b: int,
    x_by_row: bool = False,
) -> np.ndarray:
    # Normalize to k x N layout.
    if not x_by_row:
        x_grades_centered = x_grades_centered.T
    k = x_grades_centered.shape[0]
    y_autoc = _autocov_lags(y_grade_centered, n - 1)
    sigma = np.zeros((k, k), dtype=float)

    for j in range(k):
        for l in range(j, k):
            xj = x_grades_centered[j, :]
            xl = x_grades_centered[l, :]
            hac_sum = np.mean(xj * xl) * y_autoc[0]
            for h in range(1, min(b, n - 1) + 1):
                w = max(1.0 - abs(h) / (b + 1.0), 0.0)
                xcov_h = np.mean(xj[: n - h] * xl[h:] + xj[h:] * xl[: n - h]) / 2.0
                hac_sum += 2.0 * w * xcov_h * y_autoc[h]
            sigma[j, l] = hac_sum
            if j != l:
                sigma[l, j] = hac_sum
    return sigma


def _akc_asymptotic_variance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (akc estimate, asymptotic variance) using kernel-based formula."""
    tau_y, p_tie_y = _tau_y_stats(y)
    if (1.0 - p_tie_y) <= 1e-12:
        raise ValueError("AKC variance undefined because Y is almost fully tied.")

    akc, tau_xy = _compute_kendall_stats(x, y)
    n = len(x)
    k_tau_values = _k_tau_vector(x, y, tau_xy)
    k_p_values = _k_p_vector(y, tau_y)

    squared_diffs = np.empty(n, dtype=float)
    for i in range(n):
        k_tau_i = k_tau_values[i]
        k_p_i = k_p_values[i]
        diff = k_tau_i - (tau_xy / (1.0 - p_tie_y)) * k_p_i
        squared_diffs[i] = diff * diff

    expectation = np.mean(squared_diffs)
    variance = (4.0 / (1.0 - p_tie_y) ** 2) * expectation
    return float(akc), float(variance)


def _ind_variance_akc_iid(x: np.ndarray, y: np.ndarray, p_tie_y: float) -> float:
    """IID independence variance for univariate AKC."""
    n = len(y)
    y_rank = rankdata(y, method="average")
    x_rank = rankdata(x, method="average")

    var_y_rank = np.sum((y_rank - np.mean(y_rank)) ** 2) / n
    zeta_3y = 1.0 - (12.0 / n**2) * var_y_rank

    var_x_rank = np.sum((x_rank - np.mean(x_rank)) ** 2) / n
    zeta_3x = 1.0 - (12.0 / n**2) * var_x_rank

    return float((4.0 / 9.0) * ((1.0 - zeta_3x) * (1.0 - zeta_3y)) / (1.0 - p_tie_y) ** 2)


def _ind_covariance_akc_iid(x: np.ndarray, y: np.ndarray, p_tie_y: float) -> np.ndarray:
    """IID independence covariance for multivariate AKC."""
    n, m = x.shape
    y_rank = rankdata(y, method="average")
    var_y_rank = np.sum((y_rank - np.mean(y_rank)) ** 2) / n
    zeta_3y = 1.0 - (12.0 / n**2) * var_y_rank

    scale_factor = 4.0 / (1.0 - p_tie_y) ** 2
    x_ranks = np.empty((n, m), dtype=float)
    zeta_3x = np.empty(m, dtype=float)
    for k in range(m):
        x_ranks[:, k] = rankdata(x[:, k], method="average")
        var_x_rank = np.sum((x_ranks[:, k] - np.mean(x_ranks[:, k])) ** 2) / n
        zeta_3x[k] = 1.0 - (12.0 / n**2) * var_x_rank

    sigma_ind = np.zeros((m, m), dtype=float)
    for k in range(m):
        sigma_ind[k, k] = (4.0 / 9.0) * (1.0 - zeta_3x[k]) * (1.0 - zeta_3y) / (1.0 - p_tie_y) ** 2
        for l in range(k + 1, m):
            x_grade_k = (x_ranks[:, k] - 0.5) / n - 0.5
            x_grade_l = (x_ranks[:, l] - 0.5) / n - 0.5
            rho_kl = 12.0 * np.mean(x_grade_k * x_grade_l)
            sigma_ind[k, l] = sigma_ind[l, k] = scale_factor * rho_kl * (1.0 - zeta_3y) / 9.0

    return sigma_ind


def _ind_variance_akc_hac(x: np.ndarray, y: np.ndarray, p_tie_y: float) -> float:
    n = len(y)
    b = _hac_bandwidth(n)
    x_grade = (rankdata(x, method="average") - 0.5) / n - 0.5
    y_grade = (rankdata(y, method="average") - 0.5) / n - 0.5
    return float(64.0 * _ind_lrv_univariate(x_grade, y_grade, n, b) / (1.0 - p_tie_y) ** 2)


def _ind_covariance_akc_hac(x: np.ndarray, y: np.ndarray, p_tie_y: float) -> np.ndarray:
    n, m = x.shape
    b = _hac_bandwidth(n)
    x_grades = np.empty((n, m), dtype=float)
    for k in range(m):
        x_grades[:, k] = (rankdata(x[:, k], method="average") - 0.5) / n - 0.5
    y_grade = (rankdata(y, method="average") - 0.5) / n - 0.5
    return 64.0 * _ind_lrv_multivariate(
        x_grades, y_grade, n, b, x_by_row=False
    ) / (1.0 - p_tie_y) ** 2


def compute_akc_variance_auto(x, y, iid: bool = True) -> dict:
    """AKC variance backend for acor_test."""
    x = np.asarray(x)
    y = np.asarray(y)
    if not iid:
        tau_y, p_tie_y = _tau_y_stats(y)
        if (1.0 - p_tie_y) <= 1e-12:
            raise ValueError("AKC variance undefined because Y is almost fully tied.")
        akc, tau_xy = _compute_kendall_stats(x, y)
        n = len(x)
        k_tau = _k_tau_vector(x, y, tau_xy)
        k_p = _k_p_vector(y, tau_y)
        adjusted_k = k_tau - (tau_xy / (1.0 - p_tie_y)) * k_p
        scale_factor = 4.0 / (1.0 - p_tie_y) ** 2
        var = _hac_variance_univariate(adjusted_k, scale_factor)
        var_ind = _ind_variance_akc_hac(x, y, p_tie_y)
        return {"akc": akc, "var": var, "var_ind": var_ind}

    _, p_tie_y = _tau_y_stats(y)
    akc, var = _akc_asymptotic_variance(x, y)
    var_ind = _ind_variance_akc_iid(x, y, p_tie_y)
    return {"akc": akc, "var": var, "var_ind": var_ind}


def compute_akc_multivariate_variance_auto(x, y, iid: bool = True) -> dict:
    """Multivariate AKC covariance backend for acor_test."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, m = x.shape
    akc_vector = np.zeros(m, dtype=float)
    tau_vector = np.zeros(m, dtype=float)
    k_tau_values = np.zeros((n, m), dtype=float)

    tau_y, p_tie_y = _tau_y_stats(y)
    if (1.0 - p_tie_y) <= 1e-12:
        raise ValueError("AKC covariance undefined because Y is almost fully tied.")

    k_p_values = _k_p_vector(y, tau_y)

    for k in range(m):
        x_k = x[:, k]
        akc_k, tau_k = _compute_kendall_stats(x_k, y)
        akc_vector[k] = akc_k
        tau_vector[k] = tau_k

        k_tau_values[:, k] = _k_tau_vector(x_k, y, tau_k)

    scale_factor = 4.0 / (1.0 - p_tie_y) ** 2
    adjusted_k = np.empty((n, m), dtype=float)
    for k in range(m):
        adjusted_k[:, k] = k_tau_values[:, k] - (tau_vector[k] / (1.0 - p_tie_y)) * k_p_values

    if iid:
        sigma = scale_factor * (adjusted_k.T @ adjusted_k) / n
        sigma_ind = _ind_covariance_akc_iid(x, y, p_tie_y)
    else:
        sigma = _hac_covariance_multivariate(adjusted_k, scale_factor)
        sigma_ind = _ind_covariance_akc_hac(x, y, p_tie_y)

    return {"akc_vector": akc_vector, "Sigma": sigma, "Sigma_ind": sigma_ind}


def compute_akc_variance_ij(x, y, iid: bool = True) -> dict:
    """AKC IJ variance backend via native extension."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = native_akc_ij_cpp(x, y)
    _, p_tie_y = _tau_y_stats(y)
    if iid:
        var = float(result["var_ij"])
        var_ind = _ind_variance_akc_iid(x, y, p_tie_y)
    else:
        ic = np.asarray(result["ic"], dtype=float)
        var = _hac_variance_univariate(ic, scale_factor=1.0)
        var_ind = _ind_variance_akc_hac(x, y, p_tie_y)
    return {
        "akc": float(result["akc"]),
        "var": var,
        "var_ind": var_ind,
        "ic": np.asarray(result["ic"], dtype=float),
    }


def compute_akc_multivariate_variance_ij(x, y, iid: bool = True) -> dict:
    """Multivariate AKC IJ covariance backend via native extension."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    n, m = x.shape
    ic_matrix = np.zeros((n, m), dtype=float)
    akc_vector = np.zeros(m, dtype=float)

    for k in range(m):
        result_k = native_akc_ij_cpp(x[:, k], y)
        akc_vector[k] = float(result_k["akc"])
        ic_matrix[:, k] = np.asarray(result_k["ic"], dtype=float)

    _, p_tie_y = _tau_y_stats(y)
    if iid:
        sigma = (ic_matrix.T @ ic_matrix) / n
        sigma_ind = _ind_covariance_akc_iid(x, y, p_tie_y)
    else:
        sigma = _hac_covariance_multivariate(ic_matrix, scale_factor=1.0)
        sigma_ind = _ind_covariance_akc_hac(x, y, p_tie_y)
    return {
        "akc_vector": akc_vector,
        "Sigma": sigma,
        "Sigma_ind": sigma_ind,
        "ic_matrix": ic_matrix,
    }