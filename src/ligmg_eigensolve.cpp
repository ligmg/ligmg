#include <ligmg/ligmg.hpp>
#include <petsc.h>
#include <slepc.h>
#include <iostream>
#include <vector>
#include "petscviewerhdf5.h"
#include <ligmg/petsc_shim.hpp>

using namespace std;
using namespace combblas;

int main(int argc, char** argv) {

  SlepcInitialize(&argc, &argv, (char*)0, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool leader = rank == 0;

  PCRegister("ligmg", PCCreate_LIGMG);

  // read matrix
  SpParMat<int, double, SpDCCols<int, double>> A(MPI_COMM_WORLD);
  A.ParallelReadMM(argv[1]);

  // wrap matrix to pass to petsc
  Mat mat = ligmg::wrap_combblas_mat(&A);

  // set up eigensolver
  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, mat, NULL);

  EPSSetFromOptions(eps);

  PetscInt ierr;
  Vec x;
  ierr = MatCreateVecs(mat,&x,NULL);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = EPSSetDeflationSpace(eps,1,&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  // solve
  EPSSolve(eps);


  // converged pairs
  PetscInt nconv;
  EPSGetConverged(eps, &nconv);
  if(leader) cout << "# converged eigenvalues " << nconv << endl;

  vector<PetscScalar> eigs;
  if (leader)
  for (int i = 0; i < nconv; i++) {
    PetscScalar vr, vi;
    EPSGetEigenvalue(eps, i, &vr, &vi);
    cout << "Eigenvalue " << i << ": " << vr << " + " << vi << "i" << endl;
    eigs.push_back(vr);
  }

  PetscViewer viewer;
  PetscViewerHDF5Open(MPI_COMM_WORLD, argv[2], FILE_MODE_WRITE, &viewer);

  Vec eigs_vec;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, eigs.size(), nconv, eigs.data(), &eigs_vec);
  PetscObjectSetName((PetscObject)eigs_vec, "eigen_values");
  VecView(eigs_vec, viewer);

  // write out eigenvectors
  for (int i = 0; i < nconv; i++) {
    Vec x;
    MatCreateVecs(mat, &x, NULL);
    EPSGetEigenvector(eps, i, x, NULL);
    PetscObjectSetName((PetscObject)x, ("eigen_vector_" + to_string(i)).c_str());
    VecView(x, viewer);
  }
}
