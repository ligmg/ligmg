#!/bin/bash -ex

mkdir third_party
cd third_party

echo "Downloading and building CombBLAS"
git clone https://github.com/tkonolige/CombBLAS.git
cd CombBLAS
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
cd ../../

echo "Downloading and building mxx"
git clone https://github.com/tkonolige/mxx.git
cd mxx
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
cd ../../..

echo "Building"
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCombBLAS_DIR=$(pwd)/../third_party/CombBLAS/build/CombBLAS -Dmxx_DIR=$(pwd)/../third_party/mxx/build/mxx
make
