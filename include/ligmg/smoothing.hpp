#pragma once

#include "options.hpp"
#include "util.hpp"
#include <CombBLAS/CombBLAS.h>
#include <petscksp.h>
#include <petscmat.h>

namespace ligmg {

// Special Semiring that ignores positive elements in multiplication
template <class T1, class T2> struct IgnoreDiagSR {
  typedef typename combblas::promote_trait<T1, T2>::T_promote T_promote;
  static T_promote id() { return 0; }
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_SUM; };
  static T_promote add(const T_promote &arg1, const T_promote &arg2) {
    return arg1 + arg2;
  }
  static T_promote multiply(const T1 &arg1, const T2 &arg2) {
    if (arg1 > 0) {
      return static_cast<T_promote>(arg2);
    } else {
      return (static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2));
    }
  }
  static void axpy(T1 a, const T2 &x, T_promote &y) {
    if (a < 0) {
      y += a * x;
    }
  }
};

template <class T1, class T2> struct DiagSR {
  typedef typename combblas::promote_trait<T1, T2>::T_promote T_promote;
  static T_promote id() { return 0; }
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_SUM; };
  static T_promote add(const T_promote &arg1, const T_promote &arg2) {
    return arg1 + arg2;
  }
  static T_promote multiply(const T1 &arg1, const T2 &arg2) {
    if (arg1 <= 0) {
      return static_cast<T_promote>(arg2);
    } else {
      return (static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2));
    }
  }
  static void axpy(T1 a, const T2 &x, T_promote &y) {
    if (a > 0) {
      y += a * x;
    }
  }
};

template <class T1, class T2> struct InvDiagSR {
  typedef typename combblas::promote_trait<T1, T2>::T_promote T_promote;
  static T_promote id() { return 0; }
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_SUM; };
  static T_promote add(const T_promote &arg1, const T_promote &arg2) {
    return arg1 + arg2;
  }
  static T_promote multiply(const T1 &arg1, const T2 &arg2) {
    if (arg1 <= 0) {
      return static_cast<T_promote>(arg2);
    } else {
      return (1 / static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2));
    }
  }
  static void axpy(T1 a, const T2 &x, T_promote &y) {
    if (a > 0) {
      y += 1 / a * x;
    }
  }
};

template <class T, class F, template <class, class> class S, class Vec>
void jacobi(const combblas::SpParMat<T, F, S<T, F>> &A, Vec &x, const Vec &b,
            F weight = 2.0 / 3.0) {
  // R*x
  auto Rx = combblas::SpMV<IgnoreDiagSR<F, F>>(A, x);
  if (A.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    auto seq = A.seq();
    for (auto colit = seq.begcol(); colit != seq.endcol();
         ++colit) { // iterate over columns
      auto j = colit.colid();
      for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
        auto i = nzit.rowid();
        auto k = nzit.value();
        if (i == j) {
          x.data()[i] = weight * (b.data()[i] - Rx.data()[i]) / k +
                        (1 - weight) * x.data()[i];
        }
      }
    }
  }
}

// wrapper for KSPChebychev on a combblas::SpParMat
template <class T, class F, class DER> struct KSPChebyWrapper {
  std::shared_ptr<combblas::SpParMat<T, F, DER>> A;
  std::shared_ptr<std::remove_pointer<KSP>::type> ksp;
};

// // wrapper for KSPChebychev on a combblas::SpParMat
// template <class T, class F, class DER>
// struct KSPChebyWrapper {
//   KSP* ksp;
//   std::shared_ptr<combblas::SpParMat<T, F, DER>> mat;
//   bool live;
//   KSPChebyWrapper(KSP* ksp, combblas::SpParMat<T, F, DER>* mat)
//       : ksp(ksp), mat(mat), live(true) {}
//   KSPChebyWrapper(KSPChebyWrapper<T, F, DER>&& rhs)
//       : ksp(rhs.ksp), mat(move(rhs.mat)), live(rhs.live) {
//     rhs.live = false;
//   }
//   KSPChebyWrapper() : live(false) {}
//   ~KSPChebyWrapper() {
//     if (live) {
//       KSPDestroy(ksp);
//       ksp = NULL;
//     }
//     mat.reset();
//   }
// };

template <class T, class F, class S>
KSPChebyWrapper<T, F, S> chebyshev_create(const combblas::SpParMat<T, F, S> &A,
                                          int iters, double lower,
                                          double upper) {
  auto Ap = std::make_shared<combblas::SpParMat<T, F, S>>(A);

  // create matrix wrapper
  T lrows = (Ap->getcommgrid()->GetDiagWorld() == MPI_COMM_NULL)
                ? 0
                : Ap->getlocalrows();
  Mat mat;
  MatCreateShell(Ap->getcommgrid()->GetWorld(), lrows, lrows, Ap->getnrow(),
                 Ap->getncol(), (void *)Ap.get(),
                 &mat); // TODO need to add destructor for this shell
  MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))SpMV_inplace<T, F, S>);
  MatShellSetOperation(mat, MATOP_GET_DIAGONAL,
                       (void (*)(void))getdiag<T, F, S>);

  KSP ksp = KSP();
  KSPCreate(Ap->getcommgrid()->GetWorld(), &ksp);
  KSPSetOperators(ksp, mat, mat);
  KSPSetType(ksp, KSPCHEBYSHEV);
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCJACOBI);
  KSPSetUp(ksp);

  KSPSetTolerances(ksp, 1e-10, 1e-10, 1e10, iters);
  KSPSetNormType(ksp, KSP_NORM_NONE);
  KSPSetConvergenceTest(ksp, KSPConvergedSkip, NULL, NULL);
  KSPChebyshevEstEigSet(ksp, 0, lower, 0, upper);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  // TODO: use CG and 5 iterations
  KSP ksp_cheby;
  KSPChebyshevEstEigGetKSP(ksp, &ksp_cheby);
  KSPSetType(ksp_cheby, KSPCG);
  KSPSetTolerances(ksp_cheby, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 5);

  // TODO: do ksp solve with dummy std::vector and zero iterations to force
  // computation of eigenvalues

  return {Ap, std::shared_ptr<std::remove_pointer<KSP>::type>(
                  ksp, [](KSP ksp) { KSPDestroy(&ksp); })};
}

template <class T, class F, class S>
void chebyshev(const KSPChebyWrapper<T, F, S> &ksp,
               combblas::DenseParVec<T, F> &x,
               const combblas::DenseParVec<T, F> &b) {
  Vec xx;
  Vec bb;
  VecCreateMPIWithArray(x.getcommgrid()->GetWorld(), 1, x.getLocalLength(),
                        x.getTotalLength(), x.data().data(), &xx);
  VecCreateMPIWithArray(b.getcommgrid()->GetWorld(), 1, b.getLocalLength(),
                        b.getTotalLength(), b.data().data(), &bb);
  KSPSolve(ksp.ksp.get(), bb, xx);
  // KSPView(*ksp.ksp,PETSC_VIEWER_STDOUT_WORLD); // For debugging
  VecDestroy(&xx);
  VecDestroy(&bb);
}

template <class T, class F, class DER>
std::function<void(combblas::DenseParVec<T, F> &,
                   const combblas::DenseParVec<T, F> &)>
create_smoother(const combblas::SpParMat<T, F, DER> &mat, int iters,
                SmootherType type, Options opts) {
  switch (type) {
  case SmootherType::Chebyshev: {
    auto cheby =
        chebyshev_create(mat, iters, opts.cheby_lower, opts.cheby_upper);
    return std::function<void(combblas::DenseParVec<T, F> &,
                              const combblas::DenseParVec<T, F> &)>([cheby](
        combblas::DenseParVec<T, F> &x, const combblas::DenseParVec<T, F> &b) {
      chebyshev(cheby, x, b);
    });
    break;
  }
  case SmootherType::Jacobi: {
    return std::function<void(combblas::DenseParVec<T, F> &,
                              const combblas::DenseParVec<T, F> &)>(
        [=](combblas::DenseParVec<T, F> &x,
            const combblas::DenseParVec<T, F> &b) { jacobi(mat, x, b); });
    break;
  }
  default:
    throw "unreachable";
  }
}
}
