#!/bin/bash
set -e

cp ever/new_files/*.py .
cp -r ever/new_files/notebooks .
cp ever/new_files/scene/* scene/
cp ever/new_files/gaussian_renderer/* gaussian_renderer/
cp ever/new_files/utils/* utils/

# Build splinetracer
mkdir -p ever/build
cd ever/build
# CXX=/usr/bin/g++-11 CC=/usr/bin/gcc-11 cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -D_GLIBCXX_USE_CXX11_ABI=1 ..
# CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR ..
CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR \
-DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86" \
-D PYTHON_EXECUTABLE=$(which python) \
-D CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
..
make -j$(nproc)
cd ../..

pip install -e submodules/simple-knn

# SIBR Viewer
cd SIBR_viewers

git reset --hard HEAD
git clean -fd

git checkout fossa_compatibility
git submodule update --init --recursive

git merge my_fixes

git apply --ignore-whitespace ../ever/new_files/sibr_patch.patch

cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release \
         -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 
         
cmake --build build -j$(nproc) --target install
cd ..
