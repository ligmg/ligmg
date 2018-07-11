# LigMG (Large Irregular Graph MultiGrid)-- A distributed memory graph Laplacian solver for large irregular graphs

From the paper "A Parallel Solver for Graph Laplacians". Tristan Konolige and Jed Brown. 2018. In Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '18). ACM, New York, NY, USA, Article 3, 11 pages. DOI: https://doi.org/10.1145/3218176.3218227 .


## Building

### Dependencies

This library depends on MPI, PETSc, mxx, and CombBLAS. It also requires a relatively recent version of cmake (at least 3.9).

This library uses a fork of CombBLAS available here: [https://github.com/tkonolige/CombBLAS](https://github.com/tkonolige/CombBLAS).

mxx can be found here: [https://github.com/patflick/mxx](https://github.com/patflick/mxx).

### Build script

Running `./build.sh` will download and build dependencies and then build this package. PETSc will not be downloaded, so ensure that it is installed and `PETSC_DIR` and `PETSC_ARCH` are set. `build/bin` will contain the build executables.


## Usage

### Add as a Dependency

In your `CMakeLists.txt`:

```cmake
find_package(ligmg)

target_link_libraries(my_executable ligmg)
```

### In PETSc

```cpp
#include <ligmg/ligmg.hpp>
#include <ligmg/petsc_shim.hpp>
#include <petsc.h>

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, (char*)0, NULL);

  // register this solver
  PCRegister("ligmg", PCCreate_LIGMG);
}
```

You can use this solver with `-ksp_type preonly -pc_type ligmg`.

### In Your Code

In your `.cpp`:
```cpp
#include <ligmg/ligmg.hpp>
#include <CombBLAS/CombBLAS.h>

int main(int argc, char** argv) {
  // Create a matrix somehow
  combblas::SpParMat<..> A;

  // create the solver
  ligmg::ligmg solver(A);

  // solve with a right hand side
  combblas::DenseParVec<..> rhs;
  auto result = solver.solve(rhs);

  // get the solution
  auto solution = result.solution;
}
```

### Graphs with Over 2 Billion Edges

This solver can handle graphs that have more edges than `INT_MAX`. Just make sure that PETSc has been compiled with `--with-64-bit-indices`.
