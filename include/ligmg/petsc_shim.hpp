#pragma once

#include "ligmg.hpp"
#include <petsc/private/pcimpl.h> // to access members of PC

// TODO:
//   - This applies the full multigrid solve as a preconditioner. Ideally, we
//     want to apply just one v-cycle or kcycle. However, there is no way to
//     handle factor out the work of the first elimination level if we just apply
//     one v-cycle.
//   - Verify matrix type is correct
//   - Verify solver is being used with preonly?
//   - Pass options in
//   - Guess correct relative tolerance

using LIGMGSolver = ligmg::Solver<PetscInt, PetscScalar, combblas::SpDCCols<PetscInt, PetscScalar>>;

static PetscErrorCode PCSetUp_LIGMG(PC pc) {
  PetscFunctionBegin;

  combblas::SpParMat<PetscInt, PetscScalar,
                     combblas::SpDCCols<PetscInt, PetscScalar>>* A;
  MatShellGetContext(pc->pmat, &A);

  auto* solver = new LIGMGSolver(*A);
  pc->data = solver;

  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LIGMG(PC pc) {
  PetscFunctionBegin;

  LIGMGSolver* solver = (LIGMGSolver*)pc->data;
  delete solver;

  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LIGMG(PC pc, Vec x, Vec y) {
  LIGMGSolver* solver = (LIGMGSolver*)pc->data;

  PetscFunctionBegin;

  // convert PETSc Vecs to DenseParVecs
  // TODO: can prevent copy by having custom PEETSc Vec which is backed by a
  // DenseParVec
  const PetscScalar* ary;
  VecGetArrayRead(x, &ary);
  combblas::DenseParVec<PetscInt, PetscScalar> rhs(solver->hierarchy[0]->commgrid(),
                                   solver->hierarchy[0]->size());
  std::copy_n(ary, rhs.getLocalLength(), rhs.data().begin());

  ligmg::ConvergenceInfo<PetscInt, PetscScalar> info = solver->solve(rhs);

  PetscScalar* ary_ret;
  VecGetArray(y, &ary_ret);
  std::copy_n(info.solution.data().begin(), info.solution.getLocalLength(), ary_ret);
  VecRestoreArray(y, &ary_ret);

  PetscFunctionReturn(0);
}

PetscErrorCode PCCreate_LIGMG(PC pc) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  pc->ops->apply = PCApply_LIGMG;
  pc->ops->applytranspose = PCApply_LIGMG;
  pc->ops->setup = PCSetUp_LIGMG;
  pc->ops->destroy = PCDestroy_LIGMG;

  PetscFunctionReturn(0);
}
