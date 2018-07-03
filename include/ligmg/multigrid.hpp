#pragma once

#include <petsc.h>
#include <iostream>
#include <memory>
#include "aggregation.hpp"
#include "cg.hpp"
#include "elimination.hpp"
#include "optional.hpp"
#include "options.hpp"
#include "redistribute.hpp"
#include "smoothing.hpp"

#ifdef DEBUG_INFO
#include "write_residuals.hpp"
#endif

namespace ligmg {

struct KSPDeleter {
  void operator()(KSP x) { KSPDestroy(&x); }
};

// wraps a KSP instance and its associated matrix
struct KSPWrapper {
  std::unique_ptr<std::remove_pointer<KSP>::type, KSPDeleter> ksp;
  std::unique_ptr<std::remove_pointer<Mat>::type, MatDeleter> mat;
  KSPWrapper(KSP ksp, Mat mat) : ksp(ksp), mat(mat) {}
  KSPWrapper(KSPWrapper&& rhs) = default;
  KSPWrapper(std::nullptr_t n) : ksp(n), mat(n) {}
};

// A multigrid hierarchy
template <class IT, class NZ, class DER>
using Hierarchy = std::vector<std::unique_ptr<Level<IT, NZ, DER>>>;

// Multigrid level for coarse level solve using PETSc direct solver
// Direct solver used must be distributed memory solver if coarse level lives
// on more than one process.
// TODO: Check if solver is serial and redistribute to single process
template <class T, class F, class DER>
struct CoarseLevel : Level<T, F, DER> {
  combblas::SpParMat<T, F, DER> A;
  KSPWrapper petsc_ksp;

  CoarseLevel(combblas::SpParMat<T, F, DER> A, KSPWrapper&& petsc_ksp)
      : A(A), petsc_ksp(std::move(petsc_ksp)) {}

  virtual void do_level(combblas::DenseParVec<T, F>& x,
                        const combblas::DenseParVec<T, F>& b,
                        std::function<void(combblas::DenseParVec<T, F>&,
                                           const combblas::DenseParVec<T, F>&)>
                            recurse,
                        Options opts) {
    // create PETSc vectors
    Vec b_petsc;
    Vec x_petsc;
    VecCreateSeqWithArray(MPI_COMM_SELF, PETSC_DECIDE, b.data().size(),
                          b.data().data(), &b_petsc);
    VecCreateSeqWithArray(MPI_COMM_SELF, PETSC_DECIDE, x.data().size(),
                          x.data().data(), &x_petsc);

    // solve
    PetscErrorCode ierr;
    ierr = KSPSolve(petsc_ksp.ksp.get(), b_petsc, x_petsc); PETSC_CHECK_ERROR(ierr);

    // clean up
    VecDestroy(&x_petsc);
    VecDestroy(&b_petsc);
  }
  virtual bool is_exact() { return false; }
  virtual combblas::SpParMat<T, F, DER>& matrix() { return A; }
  virtual T nnz() { return A.getnnz(); }
  virtual T size() { return A.getnrow(); }
  // FIXME: better estimate
  virtual T work(Options opts) { return A.getnnz(); }
  virtual std::string name() { return "coarse"; }
  virtual std::shared_ptr<combblas::CommGrid> commgrid() {
    return A.getcommgrid();
  }
  virtual void dump(std::string basename) {
    write_petsc(A, basename + "A.petsc");
  }
};

// a V-cycle solve
template <class T, class F, class DER>
void vcycle(const Hierarchy<T, F, DER>& h, size_t level,
            combblas::DenseParVec<T, F>& x,
            const combblas::DenseParVec<T, F>& b, Options opts) {
  h[level]->do_level(x, b,
                     std::function<void(combblas::DenseParVec<T, F>&,
                                        const combblas::DenseParVec<T, F>&)>(
                         [&](combblas::DenseParVec<T, F>& x,
                             const combblas::DenseParVec<T, F>& b) {
                           vcycle(h, level + 1, x, b, opts);
                         }),
                     opts);
}

// a K-cycle solve
template <class T, class F, class DER>
void kcycle(const Hierarchy<T, F, DER>& h, size_t level,
            combblas::DenseParVec<T, F>& x,
            const combblas::DenseParVec<T, F>& b, Options opts) {
  h[level]->do_level(x, b,
                     std::function<void(combblas::DenseParVec<T, F>&,
                                        const combblas::DenseParVec<T, F>&)>(
                         [&](combblas::DenseParVec<T, F>& x,
                             const combblas::DenseParVec<T, F>& b) {
                           if (h[level + 1]->is_exact()) {
                             kcycle(h, level + 1, x, b, opts);
                           } else {
                             // TODO: split out cg opts
                             auto opts2 = opts;
                             // opts2.tol = 1e-60;
                             opts2.iters = opts.kcycle_iters;
                             petsc_fcg(h[level + 1]->matrix(), x, b,
                                       [&](combblas::DenseParVec<T, F>& r) {
                                         combblas::DenseParVec<T, F> x(
                                             r.getcommgrid(),
                                             r.getTotalLength(), 0.0);
                                         kcycle(h, level + 1, x, r, opts);
                                         r = x;
                                       },
                                       0, opts.kcycle_ksp_type, opts2);
                           }
                         }),
                     opts);
}

// Try to add an elimination level to the hierarchy
template <class T, class F, class DER>
nonstd::optional<combblas::SpParMat<T, F, DER>> add_elimination_level(
    const combblas::SpParMat<T, F, DER>& A, Hierarchy<T, F, DER>& hierarchy,
    Options opts) {
  if (opts.always_elim) {
    auto&& elim = elimination(A, opts.elim_degree, opts.elim_factor,
                              opts.elim_iters, opts.verbose);
    if (elim) {  // check that elimination actually worked
      auto res = nonstd::make_optional(
          triple_product<combblas::PlusTimesSRing<F, F>>(elim->R, A, elim->P));
      hierarchy.emplace_back(std::move(elim));

      return res;
    }
  }
  return nonstd::nullopt;
}

template <class T, class F, class DER>
combblas::SpParMat<T, F, DER> add_aggregation_level(
    const combblas::SpParMat<T, F, DER>& A, Hierarchy<T, F, DER>& hierarchy,
    Options opts) {
  auto agg = aggregation(A, opts);
  auto res = triple_product<combblas::PlusTimesSRing<F, F>>(agg->R, A, agg->P);
  hierarchy.emplace_back(std::move(agg));
  return res;
}

// Generate a multigrid hierarchy
template <class T, class F, class DER>
std::pair<Hierarchy<T, F, DER>, F> make_hierarchy(
    const combblas::SpParMat<T, F, DER>& A, Options opts) {
  if (opts.kcycle && std::string(opts.outer_ksp_type) == KSPCG) {
    if (A.getcommgrid()->GetRank() == 0) {
      std::cout
          << "Warning: Using nonlinear K-cycles with CG acceleration. This "
             "can cause divergence. Set out_ksp_type to FCG to correct."
          << std::endl;
    }
  }

  // bound for amount of work per process
  T bound;
  if (opts.redist_bound > 0) {
    bound = opts.redist_bound;
  } else {
    // TODO: should this be matrix size independent?
    bound = A.getnnz() / A.getcommgrid()->GetSize() / opts.redist_factor;
  }

  auto start_h = MPI_Wtime();
  Hierarchy<T, F, DER> hierarchy;

  combblas::SpParMat<T, F, DER> current_A = A;
  // we gradually drop processes as the problem size gets smaller. This keeps
  // track of which processes are still doing work
  bool in_comm = true;

  while (hierarchy.size() < opts.max_levels &&
         (hierarchy.size() == 0 || hierarchy.back()->nnz() > opts.min_size ||
          hierarchy.back()->name() == "redist")) {
    // first try and add an elimination level
    auto&& A_ = add_elimination_level(current_A, hierarchy, opts);
    if (A_) current_A = std::move(A_.value());

    // now add an aggregation level
    current_A = add_aggregation_level(current_A, hierarchy, opts);

    // now try and add a redistribution level
    std::shared_ptr<combblas::CommGrid> status;
    if (opts.redist && should_redist<T, F>(current_A, bound)) {
      auto old_comm = current_A.getcommgrid();
      status = redist<T, F, DER>(current_A, bound, opts.verbose);
      hierarchy.emplace_back(
          new RedistributeLevel<T, F, DER>(old_comm, status));
      if (!status) {
        in_comm = false;
        break;
      }
    }
  }

  // assume only one process remaining
  // assert(current_A.getcommgrid()->GetSize() == 1);
  if (in_comm) {
    // TODO: do something other than a single node solve
    // get ijk for all blocks
    // iterate over local storage
    auto t = gather_ijk(current_A);

    // construct local matrix
    T x_size = current_A.getnrow();
    T y_size = current_A.getncol();

    // construct petsc solver for local matrix
    // TODO: preallocate
    PetscErrorCode ierr;
    Mat A;
    MatCreateSeqAIJ(MPI_COMM_SELF, x_size, y_size, PETSC_DEFAULT, NULL, &A);
    MatNullSpace nullsp;
    MatNullSpaceCreate(MPI_COMM_SELF, PETSC_TRUE, 0, PETSC_NULL, &nullsp);
    MatSetNullSpace(A, nullsp);

    for (T i = 0; i < std::get<0>(t).size(); i++) {
      MatSetValue(A, std::get<0>(t)[i], std::get<1>(t)[i], std::get<2>(t)[i],
                  INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // TODO: free when done
    KSP direct;
    KSPCreate(MPI_COMM_SELF, &direct);
    KSPSetType(direct, KSPCG);
    KSPSetOperators(direct, A, A);
    PC pc;
    PCCreate(MPI_COMM_SELF, &pc);
    ierr = PCSetType(pc, PCSOR); PETSC_CHECK_ERROR(ierr);
    PCSetOperators(pc, A, A);
    ierr = PCSetUp(pc); PETSC_CHECK_ERROR(ierr);
    ierr = KSPSetPC(direct, pc); PETSC_CHECK_ERROR(ierr);
    ierr = KSPSetUp(direct); PETSC_CHECK_ERROR(ierr);

    hierarchy.emplace_back(
        new CoarseLevel<T, F, DER>(current_A, KSPWrapper(direct, A)));
  }

  auto stop_h = MPI_Wtime();

  return std::make_pair(move(hierarchy), stop_h - start_h);
}

template <class T, class F, class DER>
void print_hierarchy(const Hierarchy<T, F, DER>& hierarchy) {
  auto commgrid = hierarchy[0]->commgrid();
  if (commgrid) {
    auto rank = commgrid->GetRank();
    int64_t nnz = log10(hierarchy[0]->nnz()) + 1;
    auto width_nnz = std::to_string(nnz + nnz / 3);
    int64_t size = log10(hierarchy[0]->size()) + 1;
    auto width_size = std::to_string(size + size / 3);

    if (rank == 0)
      printf(("%5s  %" + width_size + "s  %" + width_nnz + "s  %6s  %9s %5s\n")
                 .c_str(),
             "Level", "Size", "NNZ", "Type", "Comm Size", "Imb");
    for (int i = 0; i < hierarchy.size(); i++) {
      // auto imb = hierarchy[i].A.LoadImbalance();
      double imb = 0;
      auto nrow = hierarchy[i]->size();
      auto nnz = hierarchy[i]->nnz();
      if (rank == 0)
        printf(("%'5lld  %'" + width_size + "lld  %'" + width_nnz +
                "d  %6s  %9d %5.2f\n")
                   .c_str(),
               i, nrow, nnz, hierarchy[i]->name().c_str(),
               hierarchy[i]->commgrid()->GetSize(), imb);
    }
  }
}

// export a hierarchy to a directory. Useful for debugging
template <class T, class F, class DER, class Vec>
void export_hierarchy(std::string directory,
                      const Hierarchy<T, F, DER>& hierarchy) {
  std::ofstream meta(directory + "meta");
  meta << hierarchy.size() << std::endl;

  for (int i = 0; i < hierarchy.size(); i++) {
    MPI_Barrier(hierarchy[i].A.getcommgrid()->GetWorld());
    hierarchy[i].A.SaveGathered(directory + "A" + std::to_string(i) + ".mtx");
    MPI_Barrier(hierarchy[i].A.getcommgrid()->GetWorld());
    hierarchy[i].R.SaveGathered(directory + "R" + std::to_string(i) + ".mtx");
    MPI_Barrier(hierarchy[i].A.getcommgrid()->GetWorld());
    hierarchy[i].P.SaveGathered(directory + "P" + std::to_string(i) + ".mtx");
    MPI_Barrier(hierarchy[i].A.getcommgrid()->GetWorld());
    meta << hierarchy[i].is_elim << std::endl;
    if (hierarchy[i].is_elim)
      hierarchy[i].Q.SaveGathered(directory + "Q" + std::to_string(i) + ".mtx");
  }
}

// solve Lx=b using an already constructed Multigrid hierarchy
template <class T, class F, class DER, class Vec>
ConvergenceInfo<T, F> solve(const Hierarchy<T, F, DER>& hierarchy, const Vec& b,
                            Options opts, int from=0) {
  // TODO: handle redistribution with from
  auto start = MPI_Wtime();

  bool do_once = opts.top_elim && hierarchy[from]->name() == "elim";

  Vec rhs = b;

  int base_level = from;
  if (do_once) {
    rhs = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(
        reinterpret_cast<EliminationLevel<T, F, DER>*>(hierarchy[from].get())->R,
        b);
    base_level = from+1;
  }

  combblas::DenseParVec<T, F> x(rhs.getcommgrid(), rhs.getTotalLength(), 0.0);
  auto cvinfo = petsc_fcg(
      hierarchy[base_level]->matrix(), x, rhs,
      [&](combblas::DenseParVec<T, F>& r) {
        combblas::DenseParVec<T, F> x(r.getcommgrid(), r.getTotalLength(), 0.0);
        if (opts.kcycle) {
          kcycle(hierarchy, base_level, x, r, opts);
        } else {
          vcycle(hierarchy, base_level, x, r, opts);
        }
        r = x;
      },
      cycle_complexity(hierarchy, base_level, opts), opts.outer_ksp_type, opts);
  auto solution = cvinfo.solution;

  if (do_once) {
    auto lvl =
        reinterpret_cast<EliminationLevel<T, F, DER>*>(hierarchy[from].get());
    solution = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(lvl->P, solution);
    solution += combblas::SpMV<combblas::PlusTimesSRing<F, F>>(lvl->Q, b);
  }
  auto end = MPI_Wtime();

  auto r = residual(hierarchy[from]->matrix(), solution, b);
  auto res = sqrt(dot(r, r));
  if (opts.verbose_solve &&
      hierarchy[from]->matrix().getcommgrid()->GetRank() == 0)
    std::cout << "Ending residual: " << res << std::endl;
  if (opts.verbose_solve &&
      hierarchy[from]->matrix().getcommgrid()->GetRank() == 0) {
    std::cout << cvinfo.residuals[from] << " " << cvinfo.residuals.back()
              << std::endl;
  }

  return {cvinfo.residuals, cvinfo.iters(), end - start,
          cvinfo.work() /* FIXME: add interp */, solution};
}

template <class T, class F, class DER>
F cycle_complexity(const Hierarchy<T, F, DER>& hierarchy, int start,
                   Options opts) {
  F sum = 0;
  int iters = 1;
  for (std::size_t i = start; i < hierarchy.size(); i++) {
    sum += iters * hierarchy[i]->work(opts);
    // if we are doing kcycles, then each lower level is run multiple times
    if (!hierarchy[i]->is_exact() && opts.kcycle) iters *= opts.kcycle_iters;
  }

  sum = sum / hierarchy[0]->nnz();
  mxx::bcast(sum, 0, hierarchy[0]->matrix().getcommgrid()->GetWorld());

  return sum;
}
}  // namespace ligmg
