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

  std::string outfile = argc[3];

  SlepcInitialize(&argv, &argc, (char*)0, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool leader = rank == 0;

  Mat A = load_from_file(argc[1]);
  Mat S = load_from_file(argc[2]);
  Mat St;
  MatTranspose(S, MAT_INITIAL_MATRIX, &St);

  PetscInt M, N, m, n;
  MatGetSize(A, &M, &N);
  MatGetLocalSize(A, &m, &n);

  Vec diag;
  MatCreateVecs(A, NULL, &diag);
  MatGetDiagonal(A, diag);
  // VecSqrtAbs(diag);

  Mat M_;
  MatCreateAIJ(MPI_COMM_WORLD, m, m, M, M, 1, NULL, 0, NULL, &M_);
  MatDiagonalSet(M_, diag, INSERT_VALUES);

  VecReciprocal(diag);
  Mat Minv;
  MatCreateAIJ(MPI_COMM_WORLD, m, n, M, N, 1, NULL, 0, NULL, &Minv);
  MatDiagonalSet(Minv, diag, INSERT_VALUES);
  // MatDiagonalScale(A, diag, diag);



  MatGetSize(S, &M, &N);
  MatGetLocalSize(S, &m, &n);

  // create (SᵀM⁻¹S)SᵀAS
  // Mat mats[6] = {St, Minv, S, St, A, S};
  Mat C;
  MatCreate(MPI_COMM_WORLD, &C);
  MatSetSizes(C, n, n, N, N);
  MatSetType(C, MATCOMPOSITE);
  MatCompositeAddMat(C, S);
  MatCompositeAddMat(C, A);
  // MatCompositeAddMat(C, St);
  // MatCompositeAddMat(C, S);
  MatCompositeAddMat(C, Minv);
  MatCompositeAddMat(C, St);

  Mat StMS;
  MatPtAP(M_, S, MAT_INITIAL_MATRIX, PETSC_DECIDE, &StMS);

  Mat StMS_inv;
  KSP ksp;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, StMS, StMS);
  MatCreateShell(MPI_COMM_WORLD, n, n, N, N, ksp, &StMS_inv);
  MatShellSetOperation(StMS_inv, MATOP_MULT, (void(*)())ksp_solve);

  // MatCompositeAddMat(C, StMS_inv);

  // MatCompositeAddMat(C, S);
  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);
  MatCompositeSetType(C, MAT_COMPOSITE_MULTIPLICATIVE);
  if(leader) std::cout << "matrix created" << std::endl;

  // set up eigensolver
  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, C, NULL);
  EPSSetFromOptions(eps);

  // solve
  EPSSolve(eps);

  // converged std::pairs
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
  PetscViewerHDF5Open(MPI_COMM_WORLD, outfile.c_str(), FILE_MODE_WRITE, &viewer);

  Vec eigs_vec;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, eigs.size(), nconv, eigs.data(), &eigs_vec);
  PetscObjectSetName((PetscObject)eigs_vec, "eigen_values");
  VecView(eigs_vec, viewer);

  // write out eigenstd::vectors
  for (int i = 0; i < nconv; i++) {
    Vec x;
    MatCreateVecs(C, &x, NULL);
    EPSGetEigenvector(eps, i, x, NULL);
    PetscObjectSetName((PetscObject)x, ("eigen_vector_" + std::to_string(i)).c_str());
    VecView(x, viewer);
  }
}

