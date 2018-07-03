#pragma once

#include "levels.hpp"
#include "options.hpp"
#include <CombBLAS/CombBLAS.h>
#include <functional>
#include <iostream>

template <class T, class F, class DER>
std::ostream &operator<<(std::ostream &os,
                         const combblas::SpParMat<T, F, DER> &mat) {
  if (mat.getcommgrid()->GetRank() == 0) {
    os << mat.getnrow() << "x" << mat.getncol() << " combblas::SpParMat";
  }
  return os;
}

namespace ligmg {

// An Multigrid level that does partial elimination of low degree nodes
template <class T, class F, class DER>
struct EliminationLevel : Level<T, F, DER> {
  combblas::SpParMat<T, F, DER> A;
  combblas::SpParMat<T, F, DER> P;
  combblas::SpParMat<T, F, DER> R;
  combblas::SpParMat<T, F, DER> Q;
  EliminationLevel(combblas::SpParMat<T, F, DER> A,
                   combblas::SpParMat<T, F, DER> P,
                   combblas::SpParMat<T, F, DER> R,
                   combblas::SpParMat<T, F, DER> Q)
      : A(A), P(P), R(R), Q(Q) {}

  virtual void do_level(combblas::DenseParVec<T, F> &x,
                        const combblas::DenseParVec<T, F> &b,
                        std::function<void(combblas::DenseParVec<T, F> &,
                                           const combblas::DenseParVec<T, F> &)>
                            recurse,
                        Options opts) {
    using SR = combblas::PlusTimesSRing<F, F>;

    // restrict current rhs
    combblas::DenseParVec<T, F> r = combblas::SpMV<SR>(R, b);
    // initial guess is all zeros
    combblas::DenseParVec<T, F> xc(A.getcommgrid(), r.getTotalLength(), 0);

    recurse(xc, r);

    // interpolate
    // x = P*xc + Q*b
    x = combblas::SpMV<SR>(P, xc);
    x += combblas::SpMV<SR>(Q, b);
  }
  virtual bool is_exact() { return true; }
  virtual combblas::SpParMat<T, F, DER> &matrix() { return A; }
  virtual T nnz() { return A.getnnz(); }
  virtual T size() { return A.getnrow(); }
  virtual std::string name() { return "elim"; }
  virtual T work(Options opts) { return P.getnnz() + R.getnnz() + Q.getnnz(); }
  virtual std::shared_ptr<combblas::CommGrid> commgrid() {
    return A.getcommgrid();
  }
  virtual void dump(std::string basename) {
    write_petsc(A, basename + "A.petsc");
    write_petsc(P, basename + "P.petsc");
    write_petsc(R, basename + "R.petsc");
    write_petsc(Q, basename + "Q.petsc");
  }
};

// compute degrees from laplacian matrix
template <class T, class F, class D>
combblas::DenseParVec<T, F>
compute_degrees(const combblas::SpParMat<T, F, D> &mat) {
  combblas::SpParMat<T, F, D> tmp = mat;
  tmp.Apply([](F &x) { return (x == 0) ? 0 : 1; });
  auto degrees = tmp.Reduce(combblas::Row, std::plus<F>(), 0);
  degrees.Apply(
      [](F &x) { return x - 1; }); // laplacian will have one on the diagonal
  return degrees;
}

template <class T> struct Indx {
  T index;
  T degree; // if degree is -1 then this node will be eliminated
  Indx(T a, T b) : index(a), degree(b){};
  Indx() : index(-1), degree(INT_MAX){};
};

template <class T>
std::ostream &operator<<(std::ostream &os, const Indx<T> &indx) {
  os << "Indx(" << indx.index << ", " << indx.degree << ")";
  return os;
}

// semiring for computing vertices to eliminate
// + operator chooses eliminated vertex (degree < 0) then the vertex with
// smallest std::hash
// * operator is just identity
// id element is (-1,INT_MAX)
MPI_Op indxHashMinOp;

template <class T>
void indxHashMinFunc(void *invec, void *inoutvec, int *len,
                     MPI_Datatype *datatype) {
  auto pinvec = static_cast<Indx<T> *>(invec);
  auto pinoutvec = static_cast<Indx<T> *>(inoutvec);
  for (int i = 0; i < *len; i++) {
    if (pinvec[i].degree < 0) {
      pinoutvec[i] = pinvec[i];
    } else if (pinoutvec[i].degree < 0) {
      pinoutvec[i] = pinoutvec[i];
    } else if (std::hash<T>{}(pinvec[i].index) <
               std::hash<T>{}(pinoutvec[i].index)) {
      // if (pinvec[i].index < pinoutvec[i].index) {
      pinoutvec[i] = pinvec[i];
    } else {
      // do nothing
    }
  }
}

template <class T, class F> struct IndxHashMin {
  typedef Indx<T> T_promote;
  static Indx<T> id() { return Indx<T>(); };
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return indxHashMinOp; };

  // TODO: eliminate high degree nodes first?
  static Indx<T> add(const Indx<T> &arg1, const Indx<T> &arg2) {
    if (arg1.index == -1)
      return arg2;
    if (arg2.index == -1)
      return arg1;
    if (arg1.degree < 0) // arg1 already selected for elimination
      return arg1;
    if (arg2.degree < 0) // arg2 already selected for elimination
      return arg2;
    if (std::hash<T>{}(arg1.index) < std::hash<T>{}(arg2.index))
      return arg1;
    return arg2;
  }

  static Indx<T> multiply(const F &arg1, const Indx<T> &arg2) {
    if (arg1 != 0) {
      return arg2;
    } else {
      return Indx<T>();
    }
  }

  static void axpy(const F a, const Indx<T> &x, Indx<T> &y) {
    y = add(y, multiply(a, x));
  }
};

// Main elimination routine
template <class T, class F, class D>
std::unique_ptr<EliminationLevel<T, F, D>>
single_elimination(const combblas::SpParMat<T, F, D> &mat, int max_degree,
                   F elim_factor, bool verbose) {
  // create mpi_op for IndxHashMin semiring
  // TODO: move to mxx
  MPI_Op_create((MPI_User_function *)indxHashMinFunc<T>, false, &indxHashMinOp);

  auto degrees = compute_degrees(mat);

  auto commgrid = mat.getcommgrid();
  auto DiagWorld = commgrid->GetDiagWorld();

  T c_off;
  T r_off;
  mat.GetPlaceInGlobalGrid(r_off, c_off);
  T nrow = mat.getnrow();
  T ncol = mat.getncol();

  // Add each vertex's index with its degree.
  combblas::DenseParVec<T, Indx<T>> indx_degrees(
      degrees.getcommgrid(), degrees.getLocalLength(), Indx<T>(), Indx<T>());
  // indx_degrees.data().resize(degrees.data().size());
  T off = c_off;
  // std::cout << "offset is " << off << std::endl;
  for (T i = 0; i < indx_degrees.getLocalLength(); i++) {
    indx_degrees.data()[i] = Indx<T>(i + off, degrees.data()[i]);
  }

  // Get all vertices with max_degree or less
  // TODO: could probably fuse this with the matrix multiply
  // TODO: should probably use combblas::FullyDistVec instead
  combblas::SpParVec<T, Indx<T>> candidates =
      indx_degrees.Find([=](Indx<T> x) { return x.degree <= max_degree; });

  // std::cout << "Number of candidates: " << candidates.getnnz() << std::endl;
  // candidates.DebugPrint();

  // Make sure we don't eliminate two neighbors.
  // + operator chooses the vertex with smallest std::hash
  // * operator is just identity
  // id element is (-1,INT_MAX)
  // See IndxHashMin
  // Resulting std::vector has index of smallest std::hash
  combblas::SpParVec<T, Indx<T>> to_elim =
      combblas::SpMV<IndxHashMin<T, F>>(mat, candidates);

  // to_elim.DebugPrint();

  // count number of eliminated vertices
  T count = 0;
  for (T i = 0; i < to_elim.getind().size(); i++) {
    if (to_elim.getind()[i] + off == to_elim.getnum()[i].index) {
      count++;
    }
  }
  count = mat.seq().getncol() - count;

  // offset of locally renumbered vertices
  T local_no_elim_offset = 0;
  if (DiagWorld != MPI_COMM_NULL) {
    local_no_elim_offset = mxx::scan<T>(count, std::plus<T>(), DiagWorld);
    local_no_elim_offset -= count;
  }

  // number of non-eliminated vertices
  T num_no_elim = local_no_elim_offset + count;
  mxx::bcast(num_no_elim, commgrid->GetSize() - 1, commgrid->GetWorld());

  if (num_no_elim > mat.getnrow() * elim_factor) {
    return nullptr;
  }

  // TODO: show percentage of candidate vertices that were eliminated
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "Eliminated " << nrow - num_no_elim << " vertices"
              << std::endl;

  // Mapping from old id to new (no-elimed) id. Has -1 for eliminated vertices.
  auto inds = std::vector<T>(mat.seq().getncol());
  count = 0;   // current non-elim index
  T l_off = 0; // current local offset
  for (T i = 0; i < to_elim.getind().size(); i++) {
    if (to_elim.getind()[i] + off == to_elim.getnum()[i].index) {
      T elim_ind = to_elim.getnum()[i].index;

      // fill to eliminated index
      while (l_off < elim_ind - off) {
        inds[l_off] = count + local_no_elim_offset;
        count++;
        l_off++;
      }

      // eliminated index gets -1
      inds[l_off] = -1;
      l_off++;
    }
  }

  // fill remaining
  while (l_off < inds.size()) {
    inds[l_off] = count + local_no_elim_offset;
    count++;
    l_off++;
  }

  auto inds_x = inds;
  auto inds_y = inds;
  // Distribute inds down columns and across rows
  mxx::bcast(inds_y, commgrid->GetRankInProcRow(), commgrid->GetColWorld());
  mxx::bcast(inds_x, commgrid->GetRankInProcCol(), commgrid->GetRowWorld());

  // TODO: Distribute diagonal across row
  std::vector<F> diag_x(inds_x.size());
  if (DiagWorld != MPI_COMM_NULL) {
    auto seq = mat.seq();
    for (auto colit = seq.begcol(); colit != seq.endcol();
         ++colit) { // iterate over columns
      auto j = colit.colid();
      for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
        auto i = nzit.rowid();
        auto k = nzit.value();
        if (i == j) {
          diag_x[i] = k;
        }
      }
    }
  }
  mxx::bcast(diag_x, commgrid->GetRankInProcRow(), commgrid->GetColWorld());

  // Create P & Q matrices
  std::vector<T> Pi;
  std::vector<T> Pj;
  std::vector<F> Pk;
  std::vector<T> Qi; // Injection operator
  std::vector<T> Qj;
  std::vector<F> Qk;

  auto seq = mat.seq();
  for (auto colit = seq.begcol(); colit != seq.endcol();
       ++colit) { // iterate over columns
    auto j = colit.colid();
    T new_id = inds_y.at(j);
    if (new_id != -1) {                 // if our column is not eliminated
      if (DiagWorld != MPI_COMM_NULL) { // on diagonal
        for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
          auto i = nzit.rowid();
          auto k = nzit.value();
          if (i == j) {
            Pi.push_back(j + c_off);
            Pj.push_back(new_id);
            Pk.push_back(1);
          }
        }
      }
    } else { // eliminated vertex
      // iterate over edges, if other end of edge is non-eliminated, add
      // 1/edge_weight from this vertex to the other vertex's new id
      for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
        auto i = nzit.rowid();
        auto k = nzit.value();
        if (i + r_off != j + c_off) {
          Pi.push_back(j + c_off);
          Pj.push_back(inds_x.at(i));
          assert(inds_x.at(i) >= 0); // eliminated node should not be connected
                                     // to another eliminated node
          Pk.push_back(1 / diag_x.at(j) * -k);
        } else { // diagonal elements go into Q
          Qi.push_back(i + r_off);
          Qj.push_back(j + c_off);
          Qk.push_back(1 / diag_x.at(j));
        }
      }
    }
  }

  auto P = combblas::SpParMat<T, F, D>(commgrid->GetWorld());
  P.MatrixFromIJK(nrow, num_no_elim, Pi, Pj, Pk, false);

  auto R = P;
  R.Transpose();

  auto Q = combblas::SpParMat<T, F, D>(commgrid->GetWorld());
  Q.MatrixFromIJK(nrow, ncol, Qi, Qj, Qk, false);

  MPI_Op_free(&indxHashMinOp);

  return std::unique_ptr<EliminationLevel<T, F, D>>(new EliminationLevel<T, F, D>(mat, P, R, Q));
}

template <class T, class F, class D>
std::unique_ptr<EliminationLevel<T, F, D>>
elimination(const combblas::SpParMat<T, F, D> &mat, int max_degree,
            F elim_factor, int iters, bool verbose) {
  auto lvl = single_elimination(mat, max_degree, elim_factor, verbose);
  if (!lvl)
    return lvl;
  auto Q = lvl->Q;
  auto P = lvl->P;
  auto R = lvl->R;
  auto A = lvl->A;

  // FIXME: I believe this is broken in some way. elim_iters > 1 causes poorer
  // performance
  for (int i = 1; i < iters; i++) {
    // Q <- Q + P Q_i R
    // P <- P P_i
    // A <- P^T A P
    A = triple_product<combblas::PlusTimesSRing<F, F>>(lvl->R, lvl->A, lvl->P);
    lvl = single_elimination(A, max_degree, elim_factor, verbose);
    if (!lvl)
      break;
    Q = combblas::EWiseApply<F, D>(
        Q, triple_product<combblas::PlusTimesSRing<F, F>>(P, lvl->Q, R),
        std::plus<F>(), false, 0.0);
    P = combblas::PSpGEMM<combblas::PlusTimesSRing<F, F>>(P, lvl->P);
    R = combblas::PSpGEMM<combblas::PlusTimesSRing<F, F>>(lvl->R, R);
  }

  return std::unique_ptr<EliminationLevel<T, F, D>>(
      new EliminationLevel<T, F, D>(mat, std::move(P), std::move(R),
                                    std::move(Q)));
}
}

namespace combblas {
// to please CombBLAS
template <> struct promote_trait<double, ligmg::Indx<int>> {
  typedef ligmg::Indx<int> T_promote;
};
template <> struct promote_trait<double, ligmg::Indx<int64_t>> {
  typedef ligmg::Indx<int64_t> T_promote;
};
}
