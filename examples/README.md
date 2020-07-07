# Simple example of using LIGMG on a matrix market file

## Building

Assuming that `build.sh` was used to build LIGMG, this example can be built with
```sh
# Starting in the exmaples/ directory

mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIGMG_DIR=../build/LIGMG
make
```
