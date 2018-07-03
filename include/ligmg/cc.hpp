#pragma once
// connected components on a graph

#include <limits>
#include <mxx/reduction.hpp>
#include <CombBLAS/CombBLAS.h>
#include "util.hpp"

namespace ligmg {

// semiring for maximum element
template <class T1, class T2>
struct MaxSRing {
  static T2 id() { return std::numeric_limits<T2>::min(); };
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_MAX; };
  static T2 add(const T2& arg1, const T2& arg2) { return std::max(arg1, arg2); }
  static T2 multiply(const T1& arg1, const T2& arg2) { return arg2; }
  static void axpy(T1 a, const T2& x, T2& y) { y = std::max(y, x); }
};

// get all connected components of the graph
// the resulting std::vector contains the id each vertex belongs to
template <class T, class F, template <class, class> class DER>
combblas::DenseParVec<T, T> connected_components(const combblas::SpParMat<T, F, DER<T, F>>& graph) {
  // We repeatedly run matrix multiplication with a semiring that chooses the
  // largest index vertex from amongst all its neighbors. When the output of
  // the multiplication is the same as the input, we are done.

  // have to convert the graph to our index type because combblas::SpMV doesn't promote
  // results correctly
  combblas::SpParMat<T, T, DER<T, T>> A = graph;

  // connected component each vertex belongs to (cc[i] = id of component)
  combblas::DenseParVec<T, T> cc(A.getcommgrid(), A.getnrow(), 0, 0);
  cc.iota(A.getnrow(), 0);  // start with cc[i] = i
  combblas::DenseParVec<T, T> cc_old(A.getcommgrid(), A.getnrow(), 0,
                           0);  // values from previous round
  while (true) {
    bool all_same = mxx::allreduce(int(cc.data() == cc_old.data()),
                                   [](int a, int b) { return a && b; });
    if (all_same) {
      break;
    }
    cc_old = cc;
    cc = combblas::SpMV<MaxSRing<T, T>>(A, cc);
    // A does not have a diagonal, so we do std::max with previous result
    for (size_t i = 0; i < cc.data().size(); i++) {
      cc.data()[i] = std::max(cc.data()[i], cc_old.data()[i]);
    }
  }

  return cc;
}

// get the largest connected component of the graph
template <class T, class F, class DER>
combblas::SpParMat<T, F, DER> largest_cc(const combblas::SpParMat<T, F, DER>& graph) {
  // get all connected components
  combblas::DenseParVec<T, T> cc = connected_components(graph);

  // find largest component
  // TODO: could probably communicate less
  std::vector<uint64_t> counts(cc.getTotalLength(), 0);
  if (cc.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    for (std::size_t i = 0; i < cc.data().size(); i++) {
      counts[cc.data()[i]] += 1;
    }
  }
  counts = mxx::allreduce(counts, std::plus<uint64_t>(),
                          cc.getcommgrid()->GetWorld());
  auto it = max_element(counts.begin(), counts.end());
  auto cc_id = distance(counts.begin(), it);

  // CombBLAS's SubsRef function does not work to constrict the matrix so we
  // reduce ourself
  // FIXME

  // renumber vertices in largest cc
  auto cc_ids = mxx::allgatherv(cc.data(), graph.getcommgrid()->GetWorld());
  auto ids = std::vector<T>(cc_ids.size(), -1);
  T next_id = 0;
  for (size_t i = 0; i < cc_ids.size(); i++) {
    if (cc_ids[i] == cc_id) {
      ids[i] = next_id;
      next_id++;
    }
  }

  // std::get only edges connecting two vertices of largest cc
  std::vector<T> is;
  std::vector<T> js;
  std::vector<F> ks;
  dcsc_iter(graph, [&](T x, T y, F v) {
    if (cc_ids[x] == cc_id && cc_ids[y] == cc_id) {
      is.push_back(ids[x]);
      js.push_back(ids[y]);
      ks.push_back(v);
    }
  });

  // construct matrix
  combblas::SpParMat<T, F, DER> res(graph.getcommgrid());
  res.MatrixFromIJK(next_id, next_id, is, js, ks);
  return res;
}
}
