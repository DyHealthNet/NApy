cmake_minimum_required(VERSION 3.15)

set(LOCAL_CONDA "option_value" CACHE STRING "User-specified path to local conda installation.")

project(LIBNAPY VERSION 0.1
	DESCRIPTION "A parallel, nan-aware C++ implementation of statistical computations with python bindings."
               LANGUAGES CXX)

if(APPLE)
	find_package(OpenMP COMPONENTS CXX REQUIRED)  
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fPIC -Wall -pedantic")  
elseif(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -O3 -funroll-loops -flto -fPIC -Wall -pedantic")
endif()

include_directories(
	$ENV{CONDA_PREFIX}/include/python3.11
	$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/pybind11/include
	$ENV{CONDA_PREFIX}/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}
)

find_package(pybind11 REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

pybind11_add_module(libnapy napy.cpp)

if(APPLE)
	target_link_libraries(libnapy PRIVATE matrix stats OpenMP::OpenMP_CXX)  
elseif(UNIX)
	target_link_libraries(libnapy PRIVATE matrix stats)
endif()
