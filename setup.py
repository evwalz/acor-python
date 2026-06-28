from __future__ import annotations

import sys

import pybind11
from setuptools import Extension, setup

if sys.platform == "win32":
    _extra_compile_args = ["/std:c++17"]
else:
    _extra_compile_args = ["-std=c++17"]

ext_modules = [
    Extension(
        "_acor_cpp",
        ["cpp/acor_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=_extra_compile_args,
    )
]


setup(ext_modules=ext_modules)
