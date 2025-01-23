#!/bin/bash
# Check if conda is installed.
if ! command -v conda >/dev/null 2>&1; then
    echo "Install error: 'conda' is not installed or cannot be found. Please make sure that 'conda' is available."
    exit 1
fi
# Add module path to PYTHONPATH variable to make napy globally accessible.
export PYTHONPATH="${PYTHONPATH}:$(pwd)/module"
# Create conda environment and build directory for cmake.
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Installing conda environment for NApy..."
conda env create -f environment_linux.yml
conda activate napy
mkdir -p build
cd build
# Init CMake and compile C++ files.
echo "Initializing make file..."
cmake ..
echo "Building C++ part of NApy..."
make
# Move module file to suitable directory.
if ls libnanpy.cpython* &>/dev/null; then
    cp libnanpy.cpython* ../module
# Check if a file ending with '.so' exists.
elif ls *.so &>/dev/null; then
    cp *.so ../module
# If neither condition is met, compilation has failed.
else
    echo "Compilation error: no matching module file has been found in the build directory."
    exit 1
fi
echo "NApy installation finished successfully!"
