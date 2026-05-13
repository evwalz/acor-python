# acor (Python)

Python package for **Asymmetric Correlation Measures**: AKC, AGC, CID, and CMA.

## Install

From PyPI (after release):

```bash
pip install acor
```

From GitHub:

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

Verify native extension and package import:

```bash
python -c "import acor, _acor_cpp; print('ok', acor.__name__, _acor_cpp.__file__)"
```

## Python Usage

```python
import numpy as np
from acor import acor, acor_test

x = np.random.default_rng(123).normal(size=200)
y = np.random.default_rng(456).normal(size=200)

print(acor(x, y, method="akc"))
print(acor_test(x, y, method="akc", alternative="two.sided", variance="plugin"))
```

## Methods

| Method | Scale | Independence value |
|--------|-------|---------------------|
| AKC, AGC | [-1, 1] | 0 |
| CID, CMA | [0, 1] | 0.5 |

- **AKC** = Asymmetric Kendall Correlation (Kendall framework)
- **AGC** = Asymmetric Grade Correlation (Spearman framework)
- **CID** = (AKC + 1) / 2
- **CMA** = (AGC + 1) / 2

## Native extension build (developers)

IJ AKC/AGC and Fenwick-based kernels are provided by the native module `_acor_cpp`.
For end users this is bundled in wheels. Manual build is mainly for development.

Build prerequisites:
- `pybind11`
- `cmake`
- a C++17 compiler

Example local build:

```bash
python -m pip install pybind11
cmake -S cpp -B cpp/build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --config Release
```

Then make sure `_acor_cpp` is importable (for local dev usually `PYTHONPATH="src:cpp/build"`).

## Wheel/Release automation

GitHub Actions workflow `.github/workflows/wheels.yml` builds wheels for:
- Linux
- macOS (x86_64 + arm64)
- Windows

On version tags (`v*`), built wheels are published to PyPI.

