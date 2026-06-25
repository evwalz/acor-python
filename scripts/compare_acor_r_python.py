#!/usr/bin/env python3
"""Compare acor (AKC / AGC) results between R and Python via rpy2.

Loads the local R package from ``project_R`` with ``devtools::load_all`` and
compares point estimates and asymptotic variances against the Python
implementation in this repo.

Requirements
------------
- R with the ``acor`` package source (sibling ``project_R`` directory)
- ``devtools`` in R: ``install.packages("devtools")``
- Python: ``pip install rpy2 numpy scipy`` plus a built ``acor`` extension

Example
-------
    python scripts/compare_acor_r_python.py
    python scripts/compare_acor_r_python.py --n 200 --seed 1 --tol 1e-10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Local Python package (src layout).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acor_functions import acor, acor_test  # noqa: E402


def _default_r_pkg_dir() -> Path:
    return _REPO_ROOT.parent / "project_R"


@dataclass
class CaseResult:
    label: str
    estimate_diff: float
    variance_diff: float
    variance_ind_diff: float | None
    py_estimate: float
    r_estimate: float
    py_variance: float
    r_variance: float


def _load_r_acor(r_pkg_dir: Path):
    try:
        import rpy2.robjects as ro
        from rpy2.rinterface import NULL
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise SystemExit(
            "rpy2 is required. Install with: pip install rpy2\n"
            "Also ensure R is installed and R_HOME is set if rpy2 cannot find libR."
        ) from exc

    numpy2ri.activate()

    r_pkg_dir = r_pkg_dir.resolve()
    if not (r_pkg_dir / "DESCRIPTION").is_file():
        raise SystemExit(f"R package not found at {r_pkg_dir}")

    ro.r(
        f"""
        if (!requireNamespace("devtools", quietly = TRUE)) {{
          stop("Install R devtools: install.packages('devtools')")
        }}
        devtools::load_all("{r_pkg_dir.as_posix()}", quiet = TRUE, reset = TRUE)
        """
    )
    return ro, NULL, importr("acor")


def _as_float(x: Any) -> float:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size != 1:
        raise ValueError(f"expected scalar, got shape {arr.shape}")
    return float(arr[0])


def _r_y(ro, y: np.ndarray):
    """R acor expects Y as a numeric vector (not a 1-column matrix)."""
    return ro.FloatVector(np.asarray(y, dtype=float).ravel())


def _r_x(ro, x: np.ndarray):
    """R acor expects X as a numeric vector or n x m matrix (column-major)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return ro.FloatVector(x.ravel())
    if x.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {x.shape}")
    n, m = x.shape
    # R stores matrices in column-major order.
    return ro.r.matrix(ro.FloatVector(x.reshape(-1, order="F")), nrow=n, ncol=m)


def _r_test(
    ro,
    NULL,
    acor_r,
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    variance: str,
    iid: bool,
) -> tuple[float, float, float | None]:
    res = acor_r.acor_test(
        _r_x(ro, x),
        _r_y(ro, y),
        method=method,
        variance=variance,
        IID=iid,
        fisher=False,
        conf_level=0.95,
        alternative="two.sided",
        aeq_y=False,
    )
    estimate = _as_float(res.rx2("estimate"))
    variance = _as_float(res.rx2("variance"))
    var_ind = res.rx2("variance_ind")
    variance_ind = None if var_ind is NULL else _as_float(var_ind)
    return estimate, variance, variance_ind


def _py_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    variance: str,
    iid: bool,
) -> tuple[float, float, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x[:, np.newaxis]
    res = acor_test(
        x,
        y,
        method=method,
        variance=variance,
        iid=iid,
        fisher=False,
        conf_level=0.95,
        alternative="two.sided",
    )
    return float(res.estimate), float(res.variance), res.variance_ind


def _compare_univariate(
    ro,
    NULL,
    acor_r,
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    variance: str,
    iid: bool,
    label: str,
    tol: float,
) -> CaseResult:
    py_est, py_var, py_var_ind = _py_test(x, y, method=method, variance=variance, iid=iid)
    r_est, r_var, r_var_ind = _r_test(
        ro, NULL, acor_r, x, y, method=method, variance=variance, iid=iid
    )

    est_diff = abs(py_est - r_est)
    var_diff = abs(py_var - r_var)
    if py_var_ind is None and r_var_ind is None:
        var_ind_diff = None
    elif py_var_ind is None or r_var_ind is None:
        var_ind_diff = float("inf")
    else:
        var_ind_diff = abs(py_var_ind - r_var_ind)

    result = CaseResult(
        label=label,
        estimate_diff=est_diff,
        variance_diff=var_diff,
        variance_ind_diff=var_ind_diff,
        py_estimate=py_est,
        r_estimate=r_est,
        py_variance=py_var,
        r_variance=r_var,
    )

    ok = est_diff <= tol and var_diff <= tol
    if var_ind_diff is not None:
        ok = ok and var_ind_diff <= tol
    status = "OK" if ok else "FAIL"
    ind_msg = "n/a" if var_ind_diff is None else f"{var_ind_diff:.3e}"
    print(
        f"[{status}] {label}\n"
        f"       estimate  py={py_est:.12g}  r={r_est:.12g}  |diff|={est_diff:.3e}\n"
        f"       variance  py={py_var:.12g}  r={r_var:.12g}  |diff|={var_diff:.3e}\n"
        f"       var_ind   |diff|={ind_msg}"
    )
    return result


def _compare_acor_point(
    ro,
    acor_r,
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    label: str,
    tol: float,
) -> CaseResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x[:, np.newaxis]

    py_est = np.asarray(acor(x, y, method=method).statistic, dtype=float).ravel()
    r_est = np.asarray(
        acor_r.acor(_r_x(ro, x), _r_y(ro, y), method=method).rx2("estimate"),
        dtype=float,
    ).ravel()

    est_diff = float(np.max(np.abs(py_est - r_est)))
    status = "OK" if est_diff <= tol else "FAIL"
    print(f"[{status}] {label}  acor point estimate  max |diff|={est_diff:.3e}")
    return CaseResult(
        label=label,
        estimate_diff=est_diff,
        variance_diff=0.0,
        variance_ind_diff=None,
        py_estimate=float(py_est[0]),
        r_estimate=float(r_est[0]),
        py_variance=0.0,
        r_variance=0.0,
    )


def _datasets(seed: int, n: int) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    x_cont = rng.standard_normal(n)
    y_cont = rng.standard_normal(n)
    yield "continuous", x_cont, y_cont

    x_tie = rng.standard_normal(n)
    y_tie = np.round(rng.standard_normal(n), 1)
    yield "y_ties", x_tie, y_tie

    x_bin = rng.standard_normal(n)
    y_bin = rng.integers(0, 2, size=n).astype(float)
    yield "binary_y", x_bin, y_bin

    x_mv = rng.standard_normal((n, 2))
    y_mv = rng.standard_normal(n)
    yield "multivariate_x", x_mv, y_mv


def _run_all(r_pkg_dir: Path, *, n: int, seed: int, tol: float) -> int:
    ro, NULL, acor_r = _load_r_acor(r_pkg_dir)
    results: list[CaseResult] = []

    for data_label, x, y in _datasets(seed, n):
        x_arr = np.asarray(x, dtype=float)
        x_uni = x_arr if x_arr.ndim == 1 else x_arr[:, 0]

        for method in ("akc", "agc"):
            results.append(
                _compare_acor_point(
                    ro,
                    acor_r,
                    x_arr,
                    y,
                    method=method,
                    label=f"{data_label} / acor / {method}",
                    tol=tol,
                )
            )

            for variance in ("plugin", "ij"):
                for iid in (True, False):
                    iid_label = "IID" if iid else "HAC"
                    results.append(
                        _compare_univariate(
                            ro,
                            NULL,
                            acor_r,
                            x_uni,
                            y,
                            method=method,
                            variance=variance,
                            iid=iid,
                            label=f"{data_label} / {method} / {variance} / {iid_label}",
                            tol=tol,
                        )
                    )

        if x.ndim == 2:
            for method in ("akc", "agc"):
                for variance in ("plugin", "ij"):
                    for iid in (True, False):
                        iid_label = "IID" if iid else "HAC"
                        # Multivariate: compare first column estimate + full variance diag.
                        py_x = np.asarray(x, dtype=float)
                        py_res = acor_test(
                            py_x,
                            y,
                            method=method,
                            variance=variance,
                            iid=iid,
                        )
                        r_res = acor_r.acor_test(
                            _r_x(ro, py_x),
                            _r_y(ro, y),
                            method=method,
                            variance=variance,
                            IID=iid,
                            fisher=False,
                            conf_level=0.95,
                            alternative="two.sided",
                            aeq_y=False,
                        )
                        py_est = np.asarray(py_res.estimate, dtype=float)
                        r_est = np.asarray(r_res.rx2("estimate"), dtype=float).ravel()
                        py_var = np.asarray(py_res.variance, dtype=float)
                        r_var = np.asarray(r_res.rx2("variance"), dtype=float)

                        est_diff = float(np.max(np.abs(py_est - r_est)))
                        var_diff = float(np.max(np.abs(py_var - r_var)))
                        ok = est_diff <= tol and var_diff <= tol
                        status = "OK" if ok else "FAIL"
                        label = f"{data_label} / multivariate / {method} / {variance} / {iid_label}"
                        print(
                            f"[{status}] {label}\n"
                            f"       max |estimate diff|={est_diff:.3e}\n"
                            f"       max |variance diff|={var_diff:.3e}"
                        )
                        results.append(
                            CaseResult(
                                label=label,
                                estimate_diff=est_diff,
                                variance_diff=var_diff,
                                variance_ind_diff=None,
                                py_estimate=float(py_est[0]),
                                r_estimate=float(r_est[0]),
                                py_variance=float(py_var[0, 0]),
                                r_variance=float(r_var[0, 0]),
                            )
                        )

    n_fail = sum(
        1
        for r in results
        if r.estimate_diff > tol
        or r.variance_diff > tol
        or (r.variance_ind_diff is not None and r.variance_ind_diff > tol)
    )
    print("\n--- summary ---")
    print(f"cases: {len(results)}, failures: {n_fail}, tolerance: {tol}")
    if results:
        print(f"max |estimate diff|:  {max(r.estimate_diff for r in results):.3e}")
        print(f"max |variance diff|:  {max(r.variance_diff for r in results):.3e}")
        ind_diffs = [r.variance_ind_diff for r in results if r.variance_ind_diff is not None]
        if ind_diffs:
            print(f"max |var_ind diff|:   {max(ind_diffs):.3e}")
    return 1 if n_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-pkg-dir",
        type=Path,
        default=_default_r_pkg_dir(),
        help="Path to the R acor package source (project_R)",
    )
    parser.add_argument("--n", type=int, default=120, help="Sample size per synthetic dataset")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Pass tolerance for |py - r| (use ~1e-6 if extensions differ)",
    )
    args = parser.parse_args()
    sys.exit(_run_all(args.r_pkg_dir, n=args.n, seed=args.seed, tol=args.tol))


if __name__ == "__main__":
    main()
