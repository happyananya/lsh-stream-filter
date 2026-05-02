#!/bin/bash
# Build and install Falconn++ from source into the active venv
# Run from the project root: bash scripts/build_falconnpp.sh

set -e

echo "=== Installing build prerequisites ==="
sudo apt-get update
sudo apt-get install -y cmake ninja-build libeigen3-dev libboost-all-dev

echo "=== Installing pybind11 ==="
pip install pybind11

echo "=== Cloning FalconnPP ==="
if [ -d "vendor/FalconnPP" ]; then
    echo "FalconnPP already cloned, pulling latest..."
    cd vendor/FalconnPP && git pull && cd ../..
else
    mkdir -p vendor
    cd vendor
    git clone --recursive https://github.com/NinhPham/FalconnPP.git
    cd ..
fi

echo "=== Building FalconnPP Python extension ==="
cd vendor/FalconnPP
pip install .
cd ../..

echo "=== Verifying installation ==="
python -c "import FalconnPP; print('FalconnPP imported successfully!')"

echo "=== Done ==="
