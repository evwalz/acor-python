# asym-correlation (Python)

Python implementation of **Asymmetric Correlation Measures** (acor): AKC, AGC, CID, and CMA.

## Install

**From PyPI** (after the first release):

```bash
pip install asym-correlation
```

**From GitHub** (builds the C++ extension on your machine):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

GitHub installs require a C++17 compiler and Python headers. PyPI wheels do not.

The distribution is named `asym-correlation`, but you import it as `acor`:

```python
from acor import acor, acor_test
```

Verify the install:

```bash
python -c "import acor, _acor_cpp; print('ok', acor.__name__, _acor_cpp.__file__)"
```

## Usage

```python
import numpy as np
from acor import acor, acor_test

rng = np.random.default_rng(123)
x = rng.normal(size=200)
y = rng.normal(size=200)

# Point estimate only
print(acor(x, y, method="akc"))

# Test with default inference (IJ variance, IID)
result = acor_test(x, y, method="akc")
print(result.estimate, result.pvalue, result.conf_int)

# Plug-in asymptotic variance instead of IJ
result_plugin = acor_test(x, y, method="akc", variance="plugin")

# Multiple predictors: x shape (n, m)
X = rng.normal(size=(200, 3))
print(acor(X, y, method="agc").statistic)
```

### `acor_test` options

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `method` | `"akc"` | `"akc"`, `"agc"`, `"cid"`, or `"cma"` |
| `variance` | `"ij"` | `"ij"` (infinitesimal jackknife) or `"plugin"` (closed-form kernel variance). `"delta"` is accepted as an alias for `"plugin"`. |
| `iid` | `True` | If `False`, use a Bartlett-kernel HAC correction (time-series inference). |
| `alternative` | `"two.sided"` | `"two.sided"`, `"less"`, or `"greater"` |
| `conf_level` | `0.95` | Confidence level for `result.conf_int` |
| `fisher` | `False` | If `True`, build the CI via Fisher transformation |

`result.variance` is the asymptotic variance of the estimator; `result.variance_ind` is the
independence benchmark variance (same closed form regardless of `variance`).

## Methods

| Method | Scale | Independence value |
|--------|-------|---------------------|
| AKC, AGC | [-1, 1] | 0 |
| CID, CMA | [0, 1] | 0.5 |

- **AKC** — Asymmetric Kendall Correlation (Kendall / U-statistic framework)
- **AGC** — Asymmetric Grade Correlation (mid-rank / Spearman framework)
- **CID** — (AKC + 1) / 2
- **CMA** — (AGC + 1) / 2

## Native extension

IJ AKC/AGC and Fenwick-based kernels live in `_acor_cpp`. The install paths above build
this extension automatically. Developers can also build manually — see `cpp/CMakeLists.txt`.
