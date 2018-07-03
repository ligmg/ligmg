// Find eigenvectors/values of SᵀA'S where A' is A normalized by columns and rows
#include <petsc.h>
#include <slepc.h>
#include <iostream>
#include <vector>
#include "petscviewerhdf5.h"

using namespace std;

Mat load_from_file(const std::string filename) {
  Mat A;
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ,
                        &viewer);
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetFromOptions(A);
  MatLoad(A, viewer);
  PetscViewerDestroy(&viewer);
  return A;
}

PetscErrorCode ksp_solve(Mat m, Vec x, Vec y) {
  KSP ctx;
  MatShellGetContext(m, &ctx);
  KSPSolve(ctx, x, y);
  return 0;
}

int main(int argv, char** argc) {
  // if (argv != 4) {
  //   std::cout << "Usage: " << argc[0] << " A.petsc S.petsc out.h5" << std::endl;
  //   exit(1);
  // }

  SlepcInitialize(&argv, &argc, (char*)0, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool leader = rank == 0;

  Mat A = load_from_file(argc[1]);
  if(leader) std::cout << "matrix created" << std::endl;

  // set up eigensolver
  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, A, NULL);

  EPSSetFromOptions(eps);

  PetscInt ierr;
  Vec x;
  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = EPSSetDeflationSpace(eps,1,&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  // solve
  EPSSolve(eps);

  // converged pairs
  PetscInt nconv;
  EPSGetConverged(eps, &nconv);
  if(leader) std::cout << "# converged eigenvalues " << nconv << std::endl;

  std::vector<PetscScalar> eigs;
  if (leader)
  for (int i = 0; i < nconv; i++) {
    PetscScalar vr, vi;
    EPSGetEigenvalue(eps, i, &vr, &vi);
    std::cout << "Eigenvalue " << i << ": " << vr << " + " << vi << "i" << std::endl;
    eigs.push_back(vr);
  }

  PetscViewer viewer;
  PetscViewerHDF5Open(MPI_COMM_WORLD, argc[2], FILE_MODE_WRITE, &viewer);

  Vec eigs_vec;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, eigs.size(), nconv, eigs.data(), &eigs_vec);
  PetscObjectSetName((PetscObject)eigs_vec, "eigen_values");
  VecView(eigs_vec, viewer);

  // write out eigenvectors
  for (int i = 0; i < nconv; i++) {
    Vec x;
    MatCreateVecs(A, &x, NULL);
    EPSGetEigenvector(eps, i, x, NULL);
    PetscObjectSetName((PetscObject)x, ("eigen_vector_" + std::to_string(i)).c_str());
    VecView(x, viewer);
  }
}

