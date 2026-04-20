from __future__ import annotations

import numpy as np

from acor_cpp_bindings import (
    agc_ij_cpp as native_agc_ij_cpp,
    has_native_extension,
    kernel_agc_v2_cpp as native_kernel_agc_v2_cpp,
)


def compute_agc(y_rank: np.ndarray, x_rank: np.ndarray) -> float:
    """AGC for a single predictor."""
    y_rank = np.asarray(y_rank, dtype=float)
    x_rank = np.asarray(x_rank, dtype=float)
    n = len(y_rank)
    var_y = np.sum((y_rank - np.mean(y_rank)) ** 2) * (1.0 / (n - 1))
    return float(np.cov(y_rank, x_rank)[0, 1] / var_y)


def compute_agc_multivariate(y_rank: np.ndarray, xarray_ranks: np.ndarray) -> np.ndarray:
    """AGC for multiple predictors where xarray_ranks has shape (n, m)."""
    xarray_ranks = np.asarray(xarray_ranks, dtype=float)
    m = xarray_ranks.shape[1]
    return np.array([compute_agc(y_rank, xarray_ranks[:, j]) for j in range(m)])


def _prob_by_observation(values: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(values, return_counts=True)
    probs = counts.astype(float) / len(values)
    lookup = dict(zip(unique.tolist(), probs.tolist()))
    return np.array([lookup[v] for v in values], dtype=float)


def _comp_rho_agc(y_rank: np.ndarray, x_rank: np.ndarray) -> tuple[float, float]:
    n = len(y_rank)
    mean_rank = (n + 1.0) / 2.0
    var_y = np.sum((y_rank - np.mean(y_rank)) ** 2) * (1.0 / (n - 1))
    rho = (12.0 / (n**3)) * np.sum((x_rank - mean_rank) * (y_rank - mean_rank))
    agc = np.cov(y_rank, x_rank)[0, 1] / var_y
    return float(rho), float(agc)


def _agc_y_preamble(y_rank: np.ndarray) -> tuple[int, float, np.ndarray, float]:
    n = len(y_rank)
    var_y_rank_biased = np.sum((y_rank - np.mean(y_rank)) ** 2) / n
    zeta_3y = 1.0 - (12.0 / n**2) * var_y_rank_biased
    if (1.0 - zeta_3y) < 1e-10:
        raise ValueError("Y has near-zero rank variance; AGC variance is undefined.")
    k_zeta = _prob_by_observation(y_rank) ** 2 - zeta_3y
    sigma_zeta = 9.0 * np.mean(k_zeta**2)
    return n, float(zeta_3y), k_zeta, float(sigma_zeta)


def _kernel_agc_iid(y_rank: np.ndarray, x_rank: np.ndarray, rho: float) -> np.ndarray:
    """Direct kernel implementation matching the R decomposition (IID path)."""
    n = len(y_rank)
    y_unique, y_counts = np.unique(y_rank, return_counts=True)
    m_unique = len(y_unique)
    g_y_unique = (y_unique - 0.5) / n
    g_x = (x_rank - 0.5) / n

    all_exp = np.zeros((n, m_unique), dtype=float)
    for i in range(n):
        sgn_x = np.sign(x_rank - x_rank[i])
        for j, y_val in enumerate(y_unique):
            sgn_y = np.sign(y_rank - y_val)
            mean_sign = np.mean(sgn_x * sgn_y)
            all_exp[i, j] = mean_sign + 2.0 * g_x[i] + 2.0 * g_y_unique[j] - 1.0

    # g1 depends only on Y group (column average mapped back to each observation)
    col_means = np.mean(all_exp, axis=0)
    y_to_index = {val: idx for idx, val in enumerate(y_unique.tolist())}
    g1 = np.array([col_means[y_to_index[v]] for v in y_rank], dtype=float) / 4.0

    # g2 is weighted row sum by Y frequencies
    g2 = (all_exp @ y_counts.astype(float)) / (4.0 * n)
    g_y_full = (y_rank - 0.5) / n
    return 4.0 * (g1 + g2 + g_x * g_y_full - g_y_full - g_x) + 1.0 - rho


def _kernel_agc_binary(y_rank: np.ndarray, x_rank: np.ndarray, rho: float) -> np.ndarray:
    """Binary-optimized AGC kernel matching the R binary specialization."""
    n = len(x_rank)
    y_unique = np.sort(np.unique(y_rank))
    if y_unique.size != 2:
        raise ValueError("_kernel_agc_binary requires exactly two unique y ranks.")

    y0, y1 = y_unique[0], y_unique[1]
    mask0 = y_rank == y0
    mask1 = y_rank == y1
    n0 = int(np.sum(mask0))
    n1 = int(np.sum(mask1))

    g_x = (x_rank - 0.5) / n
    g_y0 = (y0 - 0.5) / n
    g_y1 = (y1 - 0.5) / n
    g_y = np.where(mask0, g_y0, g_y1)

    x_rank_g0 = np.sort(x_rank[mask0])
    x_rank_g1 = np.sort(x_rank[mask1])

    # Equivalent to R findInterval logic used in kernel_agc_binary().
    count_less_g1 = np.searchsorted(x_rank_g1, x_rank, side="left")
    count_leq_g1 = np.searchsorted(x_rank_g1, x_rank, side="right")
    count_greater_g1 = n1 - count_leq_g1
    sum_sign_g1 = count_greater_g1 - count_less_g1

    count_less_g0 = np.searchsorted(x_rank_g0, x_rank, side="left")
    count_leq_g0 = np.searchsorted(x_rank_g0, x_rank, side="right")
    count_greater_g0 = n0 - count_leq_g0
    sum_sign_g0 = count_greater_g0 - count_less_g0

    mean_sign_x_0 = sum_sign_g1 / n
    mean_sign_x_1 = -sum_sign_g0 / n

    all_exp_0 = mean_sign_x_0 + 2.0 * g_x + 2.0 * g_y0 - 1.0
    all_exp_1 = mean_sign_x_1 + 2.0 * g_x + 2.0 * g_y1 - 1.0

    col_sum_0 = np.sum(all_exp_0)
    col_sum_1 = np.sum(all_exp_1)
    g1 = np.where(mask0, col_sum_0 / (4.0 * n), col_sum_1 / (4.0 * n))
    g2 = (all_exp_0 * n0 + all_exp_1 * n1) / (4.0 * n)

    return 4.0 * (g1 + g2 + g_x * g_y - g_y - g_x) + 1.0 - rho


def _kernel_agc_v2_native(y_rank: np.ndarray, x_rank: np.ndarray, rho: float) -> np.ndarray:
    return np.asarray(native_kernel_agc_v2_cpp(x_rank, y_rank, float(rho)), dtype=float)


def _agc_iid_variance(
    k_p: np.ndarray, k_zeta: np.ndarray, rho: float, zeta_3y: float, sigma_zeta: float
) -> float:
    factor = 1.0 / ((1.0 - zeta_3y) ** 2)
    sigma_rho = 9.0 * np.mean(k_p**2)
    sigma_pz = 9.0 * np.mean(k_p * k_zeta)
    return float(
        factor
        * (
            sigma_rho
            + (2.0 * rho * sigma_pz) / (1.0 - zeta_3y)
            + (rho**2 * sigma_zeta) / ((1.0 - zeta_3y) ** 2)
        )
    )


def _hac_bandwidth(n: int) -> int:
    return int(np.floor(2.0 * (n ** (1.0 / 3.0))))


def _hac_correction_univariate(series: np.ndarray) -> float:
    n = len(series)
    b = _hac_bandwidth(n)
    corr = 0.0
    for h in range(1, min(b, n - 1) + 1):
        omega = 1.0 - h / (b + 1.0)
        corr += omega * (2.0 / n) * np.sum(series[: n - h] * series[h:])
    return float(corr)


def _hac_correction_multivariate(series_mat: np.ndarray) -> np.ndarray:
    n, m = series_mat.shape
    b = _hac_bandwidth(n)
    sigma_hac = np.zeros((m, m), dtype=float)
    for h in range(1, min(b, n - 1) + 1):
        omega = 1.0 - h / (b + 1.0)
        lag = series_mat[: n - h, :]
        lead = series_mat[h:, :]
        sigma_hac += omega * (lag.T @ lead + lead.T @ lag) / n
    return sigma_hac


def _hac_variance_univariate(series: np.ndarray, scale_factor: float) -> float:
    iid_var = scale_factor * np.mean(series**2)
    return float(iid_var + scale_factor * _hac_correction_univariate(series))


def _hac_covariance_multivariate(series_mat: np.ndarray, scale_factor: float) -> np.ndarray:
    n = series_mat.shape[0]
    sigma_iid = (series_mat.T @ series_mat) / n
    sigma_hac = _hac_correction_multivariate(series_mat)
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
    for h in range(1, min(b, n - 1) + 1):
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


def _agc_hac_c_zeta(k_zeta: np.ndarray, b: int, n: int) -> float:
    c_zeta = 0.0
    for h in range(1, min(b, n - 1) + 1):
        omega = 1.0 - h / (b + 1.0)
        c_k = np.sum(k_zeta[: n - h] * k_zeta[h:])
        c_zeta += omega * (2.0 / n) * c_k
    return float(c_zeta)


def _agc_hac_univariate_terms(k_p: np.ndarray, k_zeta: np.ndarray, b: int, n: int) -> tuple[float, float]:
    a_spear = 0.0
    b_both = 0.0
    for h in range(1, min(b, n - 1) + 1):
        omega = 1.0 - h / (b + 1.0)
        k1_lag = k_p[: n - h]
        k1_lead = k_p[h:]
        k0_lag = k_zeta[: n - h]
        k0_lead = k_zeta[h:]
        a_k = np.sum(k1_lag * k1_lead)
        b_k = np.sum(k1_lag * k0_lead + k0_lag * k1_lead)
        a_spear += omega * (2.0 / n) * a_k
        b_both += omega * (1.0 / n) * b_k
    return float(a_spear), float(b_both)


def _agc_hac_offdiag_terms(
    k_p1: np.ndarray, k_p2: np.ndarray, k_zeta: np.ndarray, b: int, n: int
) -> tuple[float, float, float]:
    a2_spear = 0.0
    b_one = 0.0
    b_two = 0.0
    for h in range(1, min(b, n - 1) + 1):
        omega = 1.0 - h / (b + 1.0)
        kp1_lag = k_p1[: n - h]
        kp1_lead = k_p1[h:]
        kp2_lag = k_p2[: n - h]
        kp2_lead = k_p2[h:]
        k0_lag = k_zeta[: n - h]
        k0_lead = k_zeta[h:]
        a_k = np.sum(kp1_lag * kp2_lead + kp1_lead * kp2_lag)
        b_k1 = np.sum(kp1_lag * k0_lead + kp1_lead * k0_lag)
        b_k2 = np.sum(kp2_lag * k0_lead + kp2_lead * k0_lag)
        a2_spear += omega * (1.0 / n) * a_k
        b_one += omega * (1.0 / n) * b_k1
        b_two += omega * (1.0 / n) * b_k2
    return float(a2_spear), float(b_one), float(b_two)


def _agc_assemble_hac_variance(
    rho: float, zeta_3y: float, a_spear: float, b_both: float, c_zeta_hac: float
) -> float:
    factor = 1.0 / ((1.0 - zeta_3y) ** 2)
    return float(
        factor
        * 9.0
        * (
            a_spear
            + (2.0 * rho * b_both) / (1.0 - zeta_3y)
            + (rho**2 * c_zeta_hac) / ((1.0 - zeta_3y) ** 2)
        )
    )


def _agc_assemble_hac_covariance(
    rho_j: float,
    rho_i: float,
    zeta_3y: float,
    a2_spear: float,
    b_one: float,
    b_two: float,
    c_zeta_hac: float,
) -> float:
    factor = 1.0 / ((1.0 - zeta_3y) ** 2)
    return float(
        factor
        * 9.0
        * (
            a2_spear
            + (rho_j * b_two) / (1.0 - zeta_3y)
            + (rho_i * b_one) / (1.0 - zeta_3y)
            + (rho_j * rho_i * c_zeta_hac) / ((1.0 - zeta_3y) ** 2)
        )
    )


def _ind_variance_agc_iid(x_rank: np.ndarray, n: int, zeta_3y: float) -> float:
    var_x_rank_biased = np.sum((x_rank - np.mean(x_rank)) ** 2) / n
    zeta_3x = 1.0 - (12.0 / n**2) * var_x_rank_biased
    return float((1.0 - zeta_3x) / (1.0 - zeta_3y))


def _ind_covariance_agc_iid(xarray_ranks: np.ndarray, n: int, zeta_3y: float) -> np.ndarray:
    m = xarray_ranks.shape[1]
    sigma_ind = np.zeros((m, m), dtype=float)
    zeta_3x = np.zeros(m, dtype=float)

    for j in range(m):
        var_x_rank_biased = np.sum((xarray_ranks[:, j] - np.mean(xarray_ranks[:, j])) ** 2) / n
        zeta_3x[j] = 1.0 - (12.0 / n**2) * var_x_rank_biased
        sigma_ind[j, j] = (1.0 - zeta_3x[j]) / (1.0 - zeta_3y)

    for j in range(m):
        for i in range(j + 1, m):
            x_grade_j = (xarray_ranks[:, j] - 0.5) / n - 0.5
            x_grade_i = (xarray_ranks[:, i] - 0.5) / n - 0.5
            rho_ji = 12.0 * np.mean(x_grade_j * x_grade_i)
            sigma_ind[j, i] = sigma_ind[i, j] = rho_ji / (1.0 - zeta_3y)

    return sigma_ind


def _ind_variance_agc_hac(
    x_rank: np.ndarray, y_rank: np.ndarray, n: int, zeta_3y: float, b: int
) -> float:
    x_grade = (x_rank - 0.5) / n - 0.5
    y_grade = (y_rank - 0.5) / n - 0.5
    return float(144.0 * _ind_lrv_univariate(x_grade, y_grade, n, b) / (1.0 - zeta_3y) ** 2)


def _ind_covariance_agc_hac(
    xarray_ranks: np.ndarray, y_rank: np.ndarray, n: int, zeta_3y: float, b: int
) -> np.ndarray:
    x_grades = ((xarray_ranks.T - 0.5) / n - 0.5)  # m x n
    y_grade = (y_rank - 0.5) / n - 0.5
    return 144.0 * _ind_lrv_multivariate(
        x_grades, y_grade, n, b, x_by_row=True
    ) / (1.0 - zeta_3y) ** 2


def compute_agc_variance_auto(y_ranks, x_ranks, iid: bool = True) -> dict:
    """AGC variance backend for acor_test.

    The IID path follows the same decomposition as the R implementation.
    """
    y_ranks = np.asarray(y_ranks, dtype=float)
    x_ranks = np.asarray(x_ranks, dtype=float)
    n, zeta_3y, k_zeta, sigma_zeta = _agc_y_preamble(y_ranks)
    rho, agc = _comp_rho_agc(y_ranks, x_ranks)
    if np.unique(y_ranks).size == 2:
        kernel_fn = _kernel_agc_binary
    else:
        kernel_fn = _kernel_agc_v2_native if has_native_extension() else _kernel_agc_iid
    k_p = kernel_fn(y_ranks, x_ranks, rho)

    var_iid = _agc_iid_variance(k_p, k_zeta, rho, zeta_3y, sigma_zeta)
    if iid:
        var = var_iid
        var_ind = _ind_variance_agc_iid(x_ranks, n, zeta_3y)
    else:
        b = _hac_bandwidth(n)
        c_zeta_hac = _agc_hac_c_zeta(k_zeta, b, n)
        a_spear, b_both = _agc_hac_univariate_terms(k_p, k_zeta, b, n)
        var_hac = _agc_assemble_hac_variance(rho, zeta_3y, a_spear, b_both, c_zeta_hac)
        var = var_iid + var_hac
        var_ind = _ind_variance_agc_hac(x_ranks, y_ranks, n, zeta_3y, b)
    return {"agc": agc, "var": var, "var_ind": var_ind}


def compute_agc_multivariate_variance_auto(y_ranks, xarray_ranks, iid: bool = True) -> dict:
    """Multivariate AGC covariance backend for acor_test.

    Expects xarray_ranks with shape (n, m).
    """
    y_ranks = np.asarray(y_ranks, dtype=float)
    xarray_ranks = np.asarray(xarray_ranks, dtype=float)
    if xarray_ranks.ndim == 1:
        xarray_ranks = xarray_ranks[:, np.newaxis]

    n, zeta_3y, k_zeta, sigma_zeta = _agc_y_preamble(y_ranks)
    m = xarray_ranks.shape[1]
    if np.unique(y_ranks).size == 2:
        kernel_fn = _kernel_agc_binary
    else:
        kernel_fn = _kernel_agc_v2_native if has_native_extension() else _kernel_agc_iid

    rhos = np.zeros(m, dtype=float)
    agc_vector = np.zeros(m, dtype=float)
    kps = np.zeros((m, n), dtype=float)

    for j in range(m):
        rho_j, agc_j = _comp_rho_agc(y_ranks, xarray_ranks[:, j])
        rhos[j] = rho_j
        agc_vector[j] = agc_j
        kps[j, :] = kernel_fn(y_ranks, xarray_ranks[:, j], rho_j)

    factor = 1.0 / ((1.0 - zeta_3y) ** 2)
    sigma_pz_vec = np.array([9.0 * np.mean(kps[j, :] * k_zeta) for j in range(m)], dtype=float)
    sigma = np.zeros((m, m), dtype=float)

    for j in range(m):
        sigma_rho = 9.0 * np.mean(kps[j, :] ** 2)
        var_iid = factor * (
            sigma_rho
            + (2.0 * rhos[j] * sigma_pz_vec[j]) / (1.0 - zeta_3y)
            + (rhos[j] ** 2 * sigma_zeta) / ((1.0 - zeta_3y) ** 2)
        )
        sigma[j, j] = var_iid
        for i in range(j + 1, m):
            sigma_rho2 = 9.0 * np.mean(kps[j, :] * kps[i, :])
            cov_agc = factor * (
                sigma_rho2
                + (rhos[j] * sigma_pz_vec[i]) / (1.0 - zeta_3y)
                + (rhos[i] * sigma_pz_vec[j]) / (1.0 - zeta_3y)
                + (rhos[j] * rhos[i] * sigma_zeta) / ((1.0 - zeta_3y) ** 2)
            )
            sigma[j, i] = sigma[i, j] = cov_agc

    if iid:
        sigma_ind = _ind_covariance_agc_iid(xarray_ranks, n, zeta_3y)
    else:
        b = _hac_bandwidth(n)
        c_zeta_hac = _agc_hac_c_zeta(k_zeta, b, n)
        for j in range(m):
            a_spear, b_both = _agc_hac_univariate_terms(kps[j, :], k_zeta, b, n)
            sigma[j, j] += _agc_assemble_hac_variance(
                rhos[j], zeta_3y, a_spear, b_both, c_zeta_hac
            )
            for i in range(j + 1, m):
                a2_spear, b_one, b_two = _agc_hac_offdiag_terms(
                    kps[j, :], kps[i, :], k_zeta, b, n
                )
                sigma[j, i] += _agc_assemble_hac_covariance(
                    rhos[j], rhos[i], zeta_3y, a2_spear, b_one, b_two, c_zeta_hac
                )
                sigma[i, j] = sigma[j, i]
        sigma_ind = _ind_covariance_agc_hac(xarray_ranks, y_ranks, n, zeta_3y, b)
    return {"agc_vector": agc_vector, "Sigma": sigma, "Sigma_ind": sigma_ind}


def compute_agc_variance_ij(y_ranks, x_ranks, iid: bool = True) -> dict:
    """AGC IJ variance backend via native extension."""
    y_ranks = np.asarray(y_ranks, dtype=float)
    x_ranks = np.asarray(x_ranks, dtype=float)
    result = native_agc_ij_cpp(x_ranks, y_ranks)
    n, zeta_3y, _, _ = _agc_y_preamble(y_ranks)
    if iid:
        var = float(result["var_ij"])
        var_ind = _ind_variance_agc_iid(x_ranks, n, zeta_3y)
    else:
        b = _hac_bandwidth(n)
        ic = np.asarray(result["ic"], dtype=float)
        var = _hac_variance_univariate(ic, scale_factor=1.0)
        var_ind = _ind_variance_agc_hac(x_ranks, y_ranks, n, zeta_3y, b)
    return {
        "agc": float(result["agc"]),
        "var": var,
        "var_ind": var_ind,
        "ic": np.asarray(result["ic"], dtype=float),
    }


def compute_agc_multivariate_variance_ij(y_ranks, xarray_ranks, iid: bool = True) -> dict:
    """Multivariate AGC IJ covariance backend via native extension."""
    y_ranks = np.asarray(y_ranks, dtype=float)
    xarray_ranks = np.asarray(xarray_ranks, dtype=float)
    if xarray_ranks.ndim == 1:
        xarray_ranks = xarray_ranks[:, np.newaxis]

    n, m = xarray_ranks.shape
    ic_matrix = np.zeros((n, m), dtype=float)
    agc_vector = np.zeros(m, dtype=float)

    for k in range(m):
        result_k = native_agc_ij_cpp(xarray_ranks[:, k], y_ranks)
        agc_vector[k] = float(result_k["agc"])
        ic_matrix[:, k] = np.asarray(result_k["ic"], dtype=float)

    _, zeta_3y, _, _ = _agc_y_preamble(y_ranks)
    if iid:
        sigma = (ic_matrix.T @ ic_matrix) / n
        sigma_ind = _ind_covariance_agc_iid(xarray_ranks, n, zeta_3y)
    else:
        b = _hac_bandwidth(n)
        sigma = _hac_covariance_multivariate(ic_matrix, scale_factor=1.0)
        sigma_ind = _ind_covariance_agc_hac(xarray_ranks, y_ranks, n, zeta_3y, b)
    return {
        "agc_vector": agc_vector,
        "Sigma": sigma,
        "Sigma_ind": sigma_ind,
        "ic_matrix": ic_matrix,
    }
