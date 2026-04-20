cmake_minimum_required(VERSION 3.18)
project(acor_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_acor_cpp acor_cpp.cpp)
