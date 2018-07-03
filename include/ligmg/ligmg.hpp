#pragma once

#include "aggregation.hpp"
#include "cc.hpp"
#include "cg.hpp"
#include "elimination.hpp"
#include "multigrid.hpp"
#include "options.hpp"
#include "util.hpp"

namespace ligmg {
template <class T, class F, class DER> struct Solver {
  Hierarchy<T, F, DER> hierarchy;
  Options opts;
  double setup_time;

  Solver(const combblas::SpParMat<T, F, DER> &A, Options opts) : opts(opts) {
    initialize_petsc();
    auto t = make_hierarchy(A, opts);
    hierarchy = move(t.first);
    setup_time = t.second;
  }
  Solver(const combblas::SpParMat<T, F, DER> &A) : opts(default_options()) {
    initialize_petsc();
    auto t = make_hierarchy(A, opts);
    hierarchy = move(t.first);
    setup_time = t.second;
  }

  // solve the system of equations with the given right hand side
  ConvergenceInfo<T, F> solve(const combblas::DenseParVec<T, F> &b) const {
    return ligmg::solve(hierarchy, b, opts);
  }

  // solve the system of equations with the given right hand side from a given level
  ConvergenceInfo<T, F> solve(const combblas::DenseParVec<T, F> &b, int from) const {
    return ligmg::solve(hierarchy, b, opts, from);
  }

  void print() const { print_hierarchy(hierarchy); }

  void dump_csv(std::ostream &os, const ConvergenceInfo<T, F> &cvinfo) {
    print_csv_header(os);
    print_csv(os, "ligmg", hierarchy[0]->nnz(), setup_time, cvinfo, opts);
  }

  void dump_hierarchy(std::string filename) {
    mkdir(filename.c_str(), 0777);
    for (std::size_t i = 0; i < hierarchy.size(); i++) {
      hierarchy[i]->dump(filename + "/" + std::to_string(i) + "_");
    }
  }

  // initialize petsc if it is not initialized already
  void initialize_petsc() {
    PetscBool initd;
    PetscInitialized(&initd);
    if(!initd) PetscInitializeNoArguments();
  }
};

template <class T, class F, class DER>
inline std::ostream &operator<<(std::ostream &os, Solver<T, F, DER> solver) {
  // FIXME: fill me in
  return os;
}
}
