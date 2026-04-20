from __future__ import annotations

import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "_acor_cpp",
        ["cpp/acor_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]


setup(ext_modules=ext_modules)
