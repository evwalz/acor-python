from __future__ import annotations

from importlib import import_module
from typing import Any


def _load_extension() -> Any | None:
    try:
        return import_module("_acor_cpp")
    except Exception:
        return None


_EXT = _load_extension()


def has_native_extension() -> bool:
    return _EXT is not None


def native_extension_path() -> str | None:
    if _EXT is None:
        return None
    return getattr(_EXT, "__file__", None)


def akc_ij_cpp(x, y):
    if _EXT is None:
        raise NotImplementedError(
            "Native extension `_acor_cpp` is not available. Build it first "
            "to use IJ variance functions."
        )
    return _EXT.akc_ij_cpp(x, y)


def agc_ij_cpp(x, y):
    if _EXT is None:
        raise NotImplementedError(
            "Native extension `_acor_cpp` is not available. Build it first "
            "to use IJ variance functions."
        )
    return _EXT.agc_ij_cpp(x, y)


def kendall_tau_sign_cpp(x, y):
    if _EXT is None:
        raise NotImplementedError(
            "Native extension `_acor_cpp` is not available. Build it first "
            "to use Fenwick delta kernels."
        )
    return _EXT.kendall_tau_sign_cpp(x, y)


def h_bar_vec_v2_cpp(x, y):
    if _EXT is None:
        raise NotImplementedError(
            "Native extension `_acor_cpp` is not available. Build it first "
            "to use Fenwick delta kernels."
        )
    return _EXT.h_bar_vec_v2_cpp(x, y)


def kernel_agc_v2_cpp(x_rank, y_rank, rho):
    if _EXT is None:
        raise NotImplementedError(
            "Native extension `_acor_cpp` is not available. Build it first "
            "to use Fenwick delta kernels."
        )
    return _EXT.kernel_agc_v2_cpp(x_rank, y_rank, rho)
