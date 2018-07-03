#pragma once

#include "options.hpp"
#include "petscksp.h"
#include "util.hpp"

namespace ligmg {

PetscErrorCode petsc_print(KSP,PetscInt,PetscReal res_norm,void*) {
  std::cout << "* " << res_norm << std::endl;
  return 0;
}

// function wrapping KSPSolve with either KSPPIPEFCG or KSPFCG
// TODO: determine how to handle solution vector. return it or take a
// reference?
// right now we do both
template <class T, class F, class S, class Func>
ConvergenceInfo<T, F>
petsc_fcg(const combblas::SpParMat<T, F, S> &A, combblas::DenseParVec<T, F> &x,
          const combblas::DenseParVec<T, F> &b, Func prec, double prec_work,
          KSPType ksp_type, Options opts) {
  auto start = MPI_Wtime();

  Mat mat = wrap_combblas_mat(&A);

  // TODO: can avoid copying by just passing pointer to stack
  // auto Ap = new combblas::SpParMat<T, F, S>(A);

  // set up ksp object
  KSP ksp;
  KSPCreate(A.getcommgrid()->GetWorld(), &ksp);
  KSPSetOperators(ksp, mat, mat);
  KSPSetType(ksp, ksp_type);
  // KSPSetType(ksp, KSPRICHARDSON);
  KSPSetFromOptions(ksp);

  // KSPMonitorSet(ksp, petsc_print, NULL, NULL);

  // set up the preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);
  // wrap preconditioner to pass PETSc arrays in and out
  auto prec_f = [&](Vec in, Vec out) {
    // copy x and b data to combblas array
    // TODO: can I use VecPlaceArray and friends to avoid making a copy on
    // each combblas::SpMV
    const F *ary;
    VecGetArrayRead(in, &ary);
    combblas::DenseParVec<T, F> rhs(A.getcommgrid(), A.getnrow());
    std::copy_n(ary, rhs.getLocalLength(), rhs.data().begin());
    VecRestoreArrayRead(in, &ary);

    // apply preconditioner
    prec(rhs);

    // copy x back
    F *ary_ret;
    VecGetArray(out, &ary_ret);
    std::copy_n(rhs.data().begin(), rhs.getLocalLength(), ary_ret);
    VecRestoreArray(out, &ary_ret);

    return 0;
  };
  PCShellSetContext(pc, &prec_f); // store wrapper lambda
  using PrecFunc = decltype(prec_f);
  // Wrapping the wrapper lambda. This wrapper is needed because we get a
  // function pointer to a capturing lambda. Instead we pass the lambda in
  // separately and then call it.
  PCShellSetApply(pc, [](PC pc, Vec in, Vec out) -> PetscErrorCode {
    PrecFunc *prec_f;
    PCShellGetContext(pc, (void **)&prec_f);
    (*prec_f)(in, out);

    return 0;
  });
  PetscErrorCode ierr;
  ierr = KSPSetUp(ksp); PETSC_CHECK_ERROR(ierr);

  KSPSetTolerances(ksp, opts.tol, 1e-60, 1e10, opts.iters);
  KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); // use whatever x is passed in

  // pass input vectors to petsc
  Vec xx;
  Vec bb;
  VecCreateMPIWithArray(x.getcommgrid()->GetWorld(), 1, x.getLocalLength(),
                        x.getTotalLength(), x.data().data(), &xx);
  VecCreateMPIWithArray(b.getcommgrid()->GetWorld(), 1, b.getLocalLength(),
                        b.getTotalLength(), b.data().data(), &bb);

  // set up residual history
  std::vector<double> residuals(opts.iters + 1);
  KSPSetResidualHistory(ksp, residuals.data(), residuals.size(), PETSC_TRUE);

  // perform the solve
  ierr = KSPSolve(ksp, bb, xx); PETSC_CHECK_ERROR(ierr);

  // get iteration count
  T iters;
  KSPGetIterationNumber(ksp, &iters);

  // shrink residual vector
  PetscInt c;
  KSPGetResidualHistory(ksp, NULL, &c);
  residuals.resize(c); // initial residual is included?

  VecDestroy(&xx);
  VecDestroy(&bb);
  KSPDestroy(&ksp);
  MatDestroy(&mat);

  auto end = MPI_Wtime();
  return {residuals, iters, end - start, iters * (1 + prec_work), x};
}
}
