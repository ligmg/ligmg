#pragma once

#include "combblas_extra.hpp"
#include "serialize.hpp"
#include "smoothing.hpp"
#include "util.hpp"
#include <CombBLAS/CombBLAS.h>
#include <functional>
#include <mxx/reduction.hpp>
#include <set>
#include <vector>

// required fix for lapacke
// see
// https://stackoverflow.com/questions/24853450/errors-using-lapack-c-header-in-c-with-visual-studio-2010
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

// TODO: does P have zero rows?

namespace ligmg {

template <class T> struct AggState;
// Status of a node during aggregation
// Seed: will cause adjacent neighbors to aggregate to it
// SeedNeighbor: A node directly aggregated to the seed
// Decided: already aggregated with a neighbor
// DecidedRoot: Seed that has aggregated with enough neighbors
// Undecided: node looking to aggregate with a neighbor
enum class ST { Seed, SeedNeighbor, Decided, DecidedRoot, Undecided };

template <class T>
combblas::DenseParVec<T, T> state_vec_to_aggregates(
    const combblas::DenseParVec<T, AggState<T>> &state_vec) {
  combblas::DenseParVec<T, T> aggregates(state_vec.getcommgrid(),
                                         state_vec.data().size(), 0, 0);
  for (T i = 0; i < state_vec.data().size(); i++) {
    switch (state_vec.data()[i].state) {
    case ST::Seed:
    case ST::Decided:
    case ST::DecidedRoot:
      aggregates.data()[i] = state_vec.data()[i].ind;
      break;
    case ST::Undecided:
      aggregates.data()[i] = -1;
      break;
    }
  }
  return aggregates;
}

// Create `num_tvs` test vectors containing random data. If `ortho` is set,
// then each test vector has mean 0.
template <class T, class F, class DER>
std::vector<combblas::DenseParVec<T, F>>
create_tvs(const combblas::SpParMat<T, F, DER> &A, int num_tvs, bool ortho) {
  std::vector<combblas::DenseParVec<T, F>> tvs;

  for (int i = 0; i < num_tvs; i++) {
    tvs.emplace_back(rand_vec(A));
    if (ortho)
      orthagonalize(tvs.back());
  }

  return tvs;
}

// Project mean out of each aggregate
template <class T, class F, class DER>
void project_mean(const combblas::SpParMat<T, F, DER> &R,
                  const combblas::DenseParVec<T, F> &agg_sizes,
                  std::vector<combblas::DenseParVec<T, F>> &tvs) {
  auto P = R;
  P.Transpose();
  for (auto &tv : tvs) {
    auto agg_means = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(R, tv);
    for (T i = 0; i < agg_means.data().size(); i++) {
      agg_means.data()[i] /= agg_sizes.data()[i];
    }

    auto projected_agg_means =
        combblas::SpMV<combblas::PlusTimesSRing<F, F>>(P, agg_means);

    for (T i = 0; i < projected_agg_means.data().size(); i++) {
      tv.data()[i] -= projected_agg_means.data()[i];
    }
  }
}

template <class T, class F, class DER>
std::vector<combblas::DenseParVec<T, F>>
smooth_vectors(const combblas::SpParMat<T, F, DER> &A,
               const combblas::SpParMat<T, F, DER> &R,
               const combblas::DenseParVec<T, F> &agg_sizes, int num_tvs,
               int num_iters) {
  auto commgrid = A.getcommgrid();

  // TODO: do all rvs together in one DenseParMat
  auto tvs = create_tvs(A, num_tvs, true);
  // TODO: random rhs?
  auto b = const_vec(A, 0.0);

  for (int i = 0; i < num_iters; i++) {
    // smooth each test vector
    for (auto &tv : tvs) {
      // jacobi(A, tv, b);
      // R*x
      F weight = 2.0 / 3.0;
      auto Rx = combblas::SpMV<IgnoreDiagSR<F, F>>(A, tv);
      if (A.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
        auto seq = A.seq();
        for (auto colit = seq.begcol(); colit != seq.endcol();
             ++colit) { // iterate over columns
          auto j = colit.colid();
          for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
            auto i = nzit.rowid();
            auto k = nzit.value();
            if (i == j) {
              tv.data()[i] =
                  weight * (-Rx.data()[i]) / k + (1 - weight) * tv.data()[i];
            }
          }
        }
      }
    }
    project_mean(R, agg_sizes, tvs);
  }

  return tvs;
}

template <class T, class F>
std::pair<std::vector<Triplet<T, F>>, T>
renumber_entries(const std::vector<Triplet<T, F>> &entries) {
  std::vector<Triplet<T, F>> res;
  std::map<T, T> ids;
  T count = 0;
  for (auto e : entries) {
    auto x = ids.find(e.row);
    T rowid;
    if (x != ids.end()) {
      rowid = x->second;
    } else {
      ids[e.row] = count;
      rowid = count;
      count++;
    }

    auto y = ids.find(e.col);
    T colid;
    if (y != ids.end()) {
      colid = y->second;
    } else {
      ids[e.col] = count;
      colid = count;
      count++;
    }

    res.emplace_back(rowid, colid, e.value);
  }

  return std::make_pair(res, count);
}

template <class T, class F> struct TVEntry {
  T id{-1};
  std::vector<F> row;
  TVEntry(T id, const std::vector<F> &row) : id(id), row(row) {}
  TVEntry() : id(), row() {}
  TVEntry(int) : id(), row() {}
};

// Construct a piecewise constant prolongation operator from a vector of
// aggregates. Each entry in the vector indicates which aggregate the vertex
// belongs too.
// This restriction operator will have zero rows aggregate ids are not
// contiguous.
// TODO: get rid of zero rows
template <class T, class F, class DER>
combblas::SpParMat<T, F, DER> piecewise_constant_restriction(
    const combblas::DenseParVec<T, AggState<T>> &state_vec,
    bool include_undecided) {
  std::vector<T> Ri;
  std::vector<T> Rj;
  std::vector<F> Rk;
  auto off = state_vec.offset();

  if (state_vec.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    auto loc_length = state_vec.getLocalLength();
    Ri.reserve(loc_length);
    Rj.reserve(loc_length);
    Rk.reserve(loc_length);
    for (T i = 0; i < state_vec.data().size(); i++) {
      if ((!include_undecided && state_vec.data()[i].state != ST::Undecided) ||
          include_undecided) {
        Rj.push_back(i + off);
        Rk.push_back(1);

        auto agg = state_vec.data().at(i).ind;
        assert(agg >= 0);
        Ri.push_back(agg);
      }
    }
  }

  combblas::SpParMat<T, F, DER> R(state_vec.getcommgrid());
  R.MatrixFromIJK(state_vec.getTotalLength(), state_vec.getTotalLength(), Ri,
                  Rj, Rk);
  return R;
}

// Figure out which aggregates are poorly performing. Returned set holds the bad
// aggregates on the current process.
template <class T, class F, class DER>
std::set<T> bad_aggregates(const combblas::SpParMat<T, F, DER> &R,
                           const std::vector<combblas::DenseParVec<T, F>> &tvs,
                           combblas::DenseParVec<T, F> &agg_sizes,
                           NormType normtype, F threshold) {
  combblas::DenseParVec<T, F> norm_2(R.getcommgrid(), tvs[0].getLocalLength(),
                                     0, 0);
  combblas::DenseParVec<T, F> norm_inf(R.getcommgrid(), tvs[0].getLocalLength(),
                                       0, 0);

  if (R.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    // Decide if an aggregate should be split
    for (T i = 0; i < tvs[0].data().size(); i++) {
      F row_2 = 0;
      F row_inf = 0;
      for (T j = 0; j < tvs.size(); j++) {
        row_2 += std::pow(tvs.at(j).data().at(i), 2);
        row_inf = std::max(std::abs(tvs.at(j).data().at(i)), row_inf);
      }
      norm_2.data().at(i) = row_2 / tvs.size();
      norm_inf.data().at(i) = row_inf;
    }
  }

  auto norm_agg_2 = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(R, norm_2);
  auto norm_agg_inf =
      combblas::SpMV<combblas::PlusTimesSRing<F, F>>(R, norm_inf);

  std::set<T> bad_aggs; // TODO: use bitvector?
  for (T i = 0; i < norm_agg_2.data().size(); i++) {
    bool norm2 =
        (normtype == NormType::Norm2 || normtype == NormType::NormBoth) &&
        norm_agg_2.data().at(i) > threshold;
    bool norminf =
        (normtype == NormType::NormInf || normtype == NormType::NormBoth) &&
        norm_agg_inf.data().at(i) > threshold;
    if ((norm2 || norminf) && agg_sizes.data().at(i) > 1) {
      bad_aggs.insert(i);
    }
  }

  // TODO: avoid allgathering the bad aggregates
  std::vector<T> bad_aggs_flat(bad_aggs.begin(), bad_aggs.end());
  auto all_bad_aggs_flat =
      mxx::allgatherv(bad_aggs_flat, R.getcommgrid()->GetDiagWorld());
  std::set<T> all_bad_aggs(all_bad_aggs_flat.begin(), all_bad_aggs_flat.end());

  return all_bad_aggs;
}

template <class T, class F> struct AppendSR {
  static mxx::custom_op<std::vector<T>> m_op;
  static std::vector<T> id() { return {}; }
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return m_op.get_op(); }
  static std::vector<T> add(const T &arg1, const std::vector<T> &arg2) {
    auto res = arg2;
    res.push_back(arg1);
    return res;
  }
  static T multiply(const T &arg1, const F &arg2) { return arg1; }
  static void axpy(F a, const T &x, std::vector<T> &y) { y.push_back(x); }
};

template <class T, class F>
mxx::custom_op<std::vector<T>> AppendSR<T, F>::m_op =
    mxx::custom_op<std::vector<T>>([](std::vector<T> &a, std::vector<T> b) {
      return b.insert(b.end(), a.begin(), a.end());
    });

template <class T> struct Id {
  T agg_id;
  T id;
};

template <class L, class R> struct SelectRightSR {
  typedef R T_promote;
  static R id() { return -1; };
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_MAX; };
  static R add(const R &arg1, const R &arg2) { return std::max(arg1, arg2); }
  static R multiply(const L &arg1, const R &arg2) { return arg2; }
  static void axpy(L a, const R &x, R &y) { y = std::max(y, x); }
};

template <class L, class R> struct SelectRightSRNoMax {
  typedef R T_promote;
  static R id() { return -1; };
  static bool returnedSAID() { return false; }
  static MPI_Op mpi_op() { return MPI_MAX; };
  static R add(const R &arg1, const R &arg2) {
    throw "unimplemented";
    return arg1;
  }
  static R multiply(const L &arg1, const R &arg2) { return arg2; }
  static void axpy(L a, const R &x, R &y) { y = std::max(y, x); }
};

// TODO: this is currently all to all on the diagonal... can it be improved?
// Gather `tvs` into bundles by aggregate if said aggregate is in `bad_aggs`
template <class T, class F, class DER>
std::map<T, std::pair<std::vector<T>, std::vector<F>>>
gather_test_vectors(const combblas::SpParMat<T, F, DER> &R,
                    const std::vector<combblas::DenseParVec<T, F>> &tvs,
                    const combblas::DenseParVec<T, F> &agg_sizes,
                    const std::set<T> &bad_aggs) {
  // push aggregate id to every vertex
  // TODO: could make this sparse with bad_aggs
  auto P = R;
  P.Transpose();
  auto agg_ids = combblas::DenseParVec<T, T>::generate(
      tvs[0].getcommgrid(), P.getncol(), [](T i) { return i; });
  for (T i = 0; i < agg_ids.data().size(); i++) {
    assert(agg_ids.data()[i] == i);
  }
  auto agg_ids_per_vertex = combblas::SpMV<SelectRightSR<T, T>>(P, agg_ids);

  // TODO: this code just seems really ugly and inelegant

  // bucket by reciever
  std::vector<std::pair<std::vector<Id<T>>, std::vector<F>>> buckets(
      R.getcommgrid()->GetDiagSize());

  for (T j = 0; j < tvs[0].data().size(); j++) {
    auto global_index = tvs[0].offset() + j;
    if (bad_aggs.find(agg_ids_per_vertex.data()[j]) != bad_aggs.end()) {
      auto proc = proc_index(
          agg_ids,
          agg_ids_per_vertex.data()[j]); // send to owner of this aggregate
      buckets[proc].first.push_back(
          {agg_ids_per_vertex.data()[j], global_index});
      for (T i = 0; i < tvs.size(); i++) {
        buckets[proc].second.push_back(tvs[i].data()[j]);
      }
    }
  }

  // pack buckets
  std::vector<size_t> send_sizes_ids;
  std::vector<Id<T>> send_tv_ids;
  std::vector<size_t> send_sizes_entries;
  std::vector<F> send_tv_entries;
  for (T i = 0; i < buckets.size(); i++) {
    auto &p = buckets[i];
    send_sizes_ids.push_back(p.first.size());
    send_tv_ids.insert(send_tv_ids.end(), p.first.begin(), p.first.end());
    send_sizes_entries.push_back(p.second.size());
    send_tv_entries.insert(send_tv_entries.end(), p.second.begin(),
                           p.second.end());
  }

  // Push tv entries to single process
  // TODO: can compute recieve sizes using aggregate information
  // TODO: use spmv structured communication
  auto recieved_ids = mxx::all2allv(send_tv_ids, send_sizes_ids,
                                    R.getcommgrid()->GetDiagWorld());
  auto recieved_entries = mxx::all2allv(send_tv_entries, send_sizes_entries,
                                        R.getcommgrid()->GetDiagWorld());

  // group by aggregate
  // TODO: can use vector instead of map?
  std::map<T, std::pair<std::vector<T>, std::vector<F>>> aggs;
  for (T i = 0; i < recieved_ids.size(); i++) {
    auto &id = recieved_ids[i];
    auto &agg = aggs[id.agg_id];
    agg.first.push_back(id.id);
    agg.second.insert(agg.second.end(),
                      recieved_entries.begin() + tvs.size() * i,
                      recieved_entries.begin() + tvs.size() * (i + 1));
  }

  return aggs;
}

template <class T, class F>
inline void encode(std::streambuf &os, const TVEntry<T, F> &x) {
  serialize::encode(os, x.id);
  serialize::encode(os, x.row);
}

template <class T, class F>
inline void decode(std::streambuf &os, TVEntry<T, F> &x) {
  serialize::decode(os, x.id);
  serialize::decode(os, x.row);
}

// Gather test vector entries associated with each aggregate onto a single
// process (per aggregate). Only entries in `bad_aggs` are sent.
template <class T, class F>
std::map<T, std::vector<TVEntry<T, F>>>
gather_tv_entries(const std::vector<combblas::DenseParVec<T, F>> &tvs,
                  const combblas::DenseParVec<T, AggState<T>> &state_vec,
                  T num_aggregates, const std::set<T> bad_aggs) {
  auto commgrid = state_vec.getcommgrid();

  // send test vector entries
  // entries are sent to the node with id = aggeregate id % commgrid size
  std::vector<mxx::future<void>> sends;
  std::vector<std::string> sends_data; // need to keep data alive for send
  auto comm = mxx::comm(commgrid->GetWorld());
  auto off = state_vec.offset();
  for (T target = 0; target < commgrid->GetSize(); target++) {
    std::map<T, std::vector<TVEntry<T, F>>> for_target;
    for (T i = 0; i < state_vec.data().size(); i++) {
      auto agg = state_vec.data().at(i).ind;
      if (agg >= 0 && bad_aggs.find(agg) != bad_aggs.end()) {
        // construct test vector row
        std::vector<F> tv_row;
        for (auto &vec : tvs) {
          tv_row.push_back(vec.data()[i]);
        }

        TVEntry<T, F> entry(i + off, tv_row);
        for_target[agg].push_back(entry);
      }
    }

    // perform the send
    sends_data.emplace_back(serialize::encoded(for_target));
    sends.emplace_back(comm.isend(sends_data.back(), target));
  }

  std::map<T, std::vector<TVEntry<T, F>>> local_entries;
  for (T target = 0; target < commgrid->GetSize(); target++) {
    // TODO: irecv?
    auto buf = comm.recv<serialize::buffer>(target);
    auto recvd = serialize::decoded<decltype(local_entries)>(buf);
    for (auto &p : recvd) {
      for (auto &e : p.second) {
        local_entries[p.first].push_back(e);
      }
    }
  }

  return local_entries;
}

template <class F>
std::vector<F> first_eigenvector(int m, std::vector<F> &&tvs) {
  // convert entries to matrix
  std::vector<F> data;
  int n = tvs.size() / m;

  // perform svd
  // TODO: template this over matrix type
  std::vector<F> s(n);
  std::vector<F> u(m * n);
  std::vector<F> superb(n - 1);
  int res = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'N', m, n, tvs.data(), n,
                           s.data(), u.data(), n, NULL, n, superb.data());

  // unsure of the ordering of singular values, so always choose the largest
  int off = 0;
  if (s[n - 1] > s[0])
    off = n - 1;
  assert(res == 0);

  std::vector<F> ret;
  for (int i = 0; i < m; i++) {
    ret.push_back(n * i + off);
  }

  return ret;
}

// Split an aggregate into to parts. The root of each part is returned.
template <class T, class F>
std::pair<T, T> split_aggregate(const std::vector<T> &indices,
                                std::vector<F> &&tvs) {
  assert(indices.size() > 1);

  int n = tvs.size() / indices.size();
  int m = indices.size();
  auto u = first_eigenvector(m, std::move(tvs));

  // largest positive and negative entries in eigenvector become new roots
  F minimum = 0;
  T minimum_ind = -1;
  F maximum = 0;
  T maximum_ind = -1;
  for (T i = 0; i < m; i++) {
    auto val = u.at(i);
    if (val < minimum) {
      minimum = val;
      minimum_ind = i;
    }
    if (val > maximum) {
      maximum = val;
      maximum_ind = i;
    }
  }
  assert(minimum_ind >= 0 && maximum_ind >= 0);
  return std::make_pair(minimum_ind, maximum_ind);
}

// Scatter data from amongst a DenseParVec distribution. Technically an
// all-to-all.
template <class D, class T, class F, class Func>
std::vector<D> scatter(const std::vector<D> &data, Func func,
                       const combblas::DenseParVec<T, F> &vec) {
  auto diag_size = vec.getcommgrid()->GetDiagSize();
  std::vector<std::vector<D>> sendbuf(diag_size);
  for (const D &x : data) {
    auto index = func(x);
    assert(index < vec.getTotalLength());
    auto proc = index / vec.getTypicalLocLength();
    if (proc >= diag_size)
      proc = diag_size - 1; // Last entry is larger
    sendbuf.at(proc).push_back(x);
  }

  std::vector<serialize::buffer> sendbuf_serialized;
  for (auto &v : sendbuf) {
    sendbuf_serialized.push_back(serialize::encoded(v));
  }

  auto recvd =
      mxx::all2all(sendbuf_serialized, vec.getcommgrid()->GetDiagWorld());
  std::vector<D> res;
  for (auto &x : recvd) {
    auto deserialized = serialize::decoded<std::vector<D>>(x);
    res.insert(res.end(), deserialized.begin(), deserialized.end());
  }
  return res;
}

template <class T, class F> struct NeighborEdge {
  F edge_weight;
  F eig_val;
  T agg;
  // These constructors are hacks to allow conversion of a SpParMat
  NeighborEdge(F a) {}
  NeighborEdge() {}
};

template <class T, class F, class DER, class H>
std::map<T, H>
scatter_to_vertices_sparse(const combblas::SpParMat<T, F, DER> &P,
                           const std::map<T, H> &info) {
  std::vector<std::pair<T, F>> packed(info.begin(), info.end());
  mxx::all2all_func(
      packed, [&P](std::pair<T, F> x) { return proc_index_col(P, x.first); });
  return std::map<T, H>(packed.begin(), packed.end());
}

// *: (edge weight, neighbor 1st eigenvector value, agg id) ->
//    (1st eigenvector, agg id) -> vote weight
// +: vote weight -> vote weight -> sum of vote weights
template <class T, class F> struct CRVoteSR {
  typedef F T_promote;
  static F id() { return 0; }
  static bool returnedSAID() { return false; }
  static F add(const F &arg1, const F &arg2) { return arg1 + arg2; }
  static F multiply(const NeighborEdge<T, F> &arg1,
                    const std::pair<F, T> &arg2) {
    if (arg2.second == arg1.agg) {
      if ((arg1.eig_val > 0 && arg2.first > 0) ||
          (arg1.eig_val < 0 && arg2.first < 0)) {
        return arg1.edge_weight * 1.0 / (arg2.first - arg1.eig_val + 0.0001);
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
  static void axpy(const NeighborEdge<T, F> &a, const std::pair<F, T> &x,
                   F &y) {
    y += multiply(a, x);
  }
};

template <class T> struct MinMax {
  T min;
  T max;
};

// *: 1 -> (vote total, id) -> (vote total, id)
// +: min and max by vote total
template <class T, class F> struct MaxMinId {
  typedef MinMax<std::pair<F, T>> T_promote;
  static MinMax<std::pair<F, T>> id() { return {{0, -1}, {0, -1}}; }
  static bool returnedSAID() { return false; }
  static MinMax<std::pair<F, T>> add(const MinMax<std::pair<F, T>> &arg1,
                                     const MinMax<std::pair<F, T>> &arg2) {
    std::pair<F, T> min =
        (arg1.min.first < arg2.min.first) ? arg1.min : arg2.min;
    std::pair<F, T> max =
        (arg1.max.first > arg2.max.first) ? arg1.max : arg2.max;
    return {min, max};
  }
  static MinMax<std::pair<F, T>> multiply(const F &arg1,
                                          const std::pair<F, T> &arg2) {
    return {arg2, arg2};
  }
  static void axpy(const F &a, const std::pair<F, T> &x,
                   MinMax<std::pair<F, T>> &y) {
    y = add(multiply(a, x), y);
  }
};

template <class T, class F, class DER>
combblas::DenseParVec<T, T>
number_by_aggregate(const combblas::SpParMat<T, F, DER> &P) {
  combblas::DenseParVec<T, T> agg_ids(P.getcommgrid(), 0);
  agg_ids.iota(P.getncol(), 0);
  return combblas::SpMV<combblas::PlusTimesSRing<T, T>>(P, agg_ids);
}

// Split aggregates by voting based of difference in 1st eigenvector of tvs
template <class T, class F, template <class, class> class DER>
combblas::DenseParVec<T, AggState<T>>
split_aggregates_voting(const combblas::SpParMat<T, F, DER<T, F>> &A,
                        const combblas::SpParMat<T, F, DER<T, F>> &R,
                        const combblas::SpParMat<T, F, DER<T, F>> &P,
                        const std::vector<combblas::DenseParVec<T, F>> &tvs,
                        const combblas::DenseParVec<T, AggState<T>> &state_vec,
                        combblas::DenseParVec<T, F> &agg_sizes,
                        NormType normtype, F threshold, bool verbose) {
  auto bad_aggs = bad_aggregates(R, tvs, agg_sizes, normtype, threshold);
  auto agg_tvs = gather_test_vectors(R, tvs, agg_sizes, bad_aggs);
  auto agg_ids = number_by_aggregate(P);

  // 1. Fill dense (sparse?) vector with value of each vertex in 1st eigenvector
  // 2. SpMV with only neighbors in aggregate, using edge weight * 1/(abs(x1 -
  // x2))
  //      Vector has (1st eig value, agg id)
  //      Matrix is (edge weight, eig value for row, agg id for row) TODO:
  //      correct? (weight, eig value, add id) * (eig value, agg id) = (agg id
  //      == agg id) ? weight * eig value * eig value : 0
  //      + is just +

  // Really want some kind of broadcasting operation here
  // broadcast across rows
  std::vector<T> local_agg_ids = agg_ids.data();
  mxx::bcast(local_agg_ids, A.getcommgrid()->GetRankInProcCol(),
             A.getcommgrid()->GetRowWorld());

  // compute 1st eigenvalue on each aggregate and scatter it to the process
  // owning the vertex
  std::map<T, F> first_eig;
  for (auto &p : agg_tvs) {
    auto &indices = p.second.first;
    auto u = first_eigenvector(indices.size(), std::move(p.second.second));
    for (T i = 0; i < indices.size(); i++) {
      first_eig[indices[i]] = u[i];
    }
  }

  auto local_eig = scatter_to_vertices_sparse(P, first_eig);
  std::vector<std::pair<T, F>> local_eig_vec(local_eig.begin(),
                                             local_eig.end());
  mxx::bcast(local_eig_vec, A.getcommgrid()->GetRankInProcCol(),
             A.getcommgrid()->GetRowWorld());
  local_eig = std::map<T, F>(local_eig_vec.begin(), local_eig_vec.end());

  // build sparse right hand side
  combblas::SpParVec<T, std::pair<F, T>> x(P.getcommgrid());
  for (auto &p : local_eig) {
    auto local_id = p.first - state_vec.offset();
    x.getind().push_back(local_id);
    x.getnum().push_back({p.second, agg_ids.data()[local_id]});
  }
  x.getlength() = state_vec.getLocalLength();

  // build left matrix
  combblas::SpParMat<T, NeighborEdge<T, F>, DER<T, NeighborEdge<T, F>>>
      neighbor_mat = A;
  dcsc_local_iter(neighbor_mat, [&](T row, T col, NeighborEdge<T, F> &val) {
    val.eig_val = local_eig[row];
    val.agg = local_agg_ids[row];
  });

  // vote for root
  combblas::SpParVec<T, F> votes =
      combblas::SpMV<CRVoteSR<T, F>>(neighbor_mat, x);

  combblas::SpParVec<T, std::pair<F, T>> votes_paired(votes.getcommgrid(), {});
  for (T i = 0; i < votes.getnum().size(); i++) {
    votes_paired.getnum().push_back(
        {votes.getnum()[i], votes.getind()[i] + votes.offset()});
    votes_paired.getind().push_back(votes.getind()[i]);
  }
  votes_paired.getlength() = votes.getlength();

  combblas::SpParVec<T, MinMax<std::pair<F, T>>> new_roots =
      combblas::SpMV<MaxMinId<T, F>>(R, votes_paired);

  combblas::SpParVec<T, MinMax<std::pair<F, T>>> local_roots =
      combblas::SpMV<SelectRightSRNoMax<F, MinMax<std::pair<F, T>>>>(P,
                                                                     new_roots);

  // Reset aggregate states
  auto new_state_vec = state_vec;
  for (T i = 0; i < new_state_vec.data().size(); i++) {
    T current_id = new_state_vec.offset();
    auto &state = new_state_vec.data()[i];
    if (state.state == ST::Decided) {
      state = {current_id, ST::Undecided};
    } else if (state.state == ST::DecidedRoot) {
      state = {current_id, ST::Seed};
    } else if (bad_aggs.find(agg_ids.data()[i]) != bad_aggs.end()) {
      state = {current_id, ST::Undecided};
    }
  }

  // Set roots from the voting
  for (T i = 0; i < local_roots.getind().size(); i++) {
    auto &minmax = local_roots.getnum()[i];
    T ind = local_roots.getind()[i];
    T current_id = local_roots.getind()[i] + local_roots.offset();
    if (minmax.min.second == current_id || minmax.max.second == current_id) {
      new_state_vec.data()[ind] = {current_id, ST::Seed};
    }
  }

  return new_state_vec;
}

template <class T, class F, template <class, class> class DER>
combblas::DenseParVec<T, AggState<T>>
split_aggregates(const combblas::SpParMat<T, F, DER<T, F>> &A,
                 const combblas::SpParMat<T, F, DER<T, F>> &R,
                 const combblas::SpParMat<T, F, DER<T, F>> &P,
                 const std::vector<combblas::DenseParVec<T, F>> &tvs,
                 const combblas::DenseParVec<T, AggState<T>> &state_vec,
                 combblas::DenseParVec<T, F> &agg_sizes, NormType normtype,
                 F threshold, bool verbose) {
  auto bad_aggs = bad_aggregates(R, tvs, agg_sizes, normtype, threshold);
  auto agg_tvs = gather_test_vectors(R, tvs, agg_sizes, bad_aggs);
  auto agg_ids = number_by_aggregate(P);

  std::vector<T> local_agg_ids = agg_ids.data();
  mxx::bcast(local_agg_ids, A.getcommgrid()->GetRankInProcCol(),
             A.getcommgrid()->GetRowWorld());

  // split aggregate by sign of vertex in first eigenvector
  // XXX: Will I run into issues if P has zero rows?
  std::map<T, T> new_aggregates;
  for (auto &p : agg_tvs) {
    auto &indices = p.second.first;
    auto u = first_eigenvector(indices.size(), std::move(p.second.second));
    bool found_first = false;
    T first;
    for (int i = 0; i < indices.size(); i++) {
      if (u[i] < 0) {
        if (!found_first) {
          first = indices[i];
        }
        new_aggregates[indices[i]] = first;
      }
    }
  }

  combblas::DenseParVec<T, AggState<T>> new_state_vec = state_vec;

  // scatter new aggregate ids
  auto local_new_aggregates = scatter_to_vertices_sparse(P, new_aggregates);
  for (auto p : local_new_aggregates) {
    new_state_vec.data().at(p.first - new_state_vec.offset()) = {
        p.second, (p.first == p.second) ? ST::Seed : ST::Decided};
  }

  return new_state_vec;
}

// Generate new roots by splitting aggregates that are "poor"
template <class T, class F, template <class, class> class DER>
combblas::DenseParVec<T, AggState<T>> compatible_relaxation_roots(
    const combblas::SpParMat<T, F, DER<T, F>> &A,
    combblas::DenseParVec<T, AggState<T>> &state_vec, F threshold, int num_tvs,
    int iters, bool verbose, NormType norm,
    std::vector<std::vector<combblas::DenseParVec<T, F>>> &cr_tvs) {
  auto R = piecewise_constant_restriction<T, F, DER<T, F>>(state_vec, false);
  auto P = R;
  P.Transpose();

  struct NZAdd {
    T operator()(F a, T b) { return ((a != 0) ? 1 : 0) + b; }
  };
  combblas::DenseParVec<T, F> agg_sizes = R.Reduce(combblas::Row, NZAdd(), 0);
  auto tvs = smooth_vectors(A, R, agg_sizes, num_tvs, iters);
  cr_tvs.push_back(tvs);

  return split_aggregates_voting<T, F, DER>(A, R, P, tvs, state_vec, agg_sizes,
                                            norm, threshold, verbose);
}

// Split new aggregates in half if they are "poor"
template <class T, class F, template <class, class> class DER>
combblas::DenseParVec<T, AggState<T>> compatible_relaxation_splitting(
    const combblas::SpParMat<T, F, DER<T, F>> &A,
    combblas::DenseParVec<T, AggState<T>> &state_vec, F threshold, int num_tvs,
    int iters, bool verbose, NormType norm,
    std::vector<std::vector<combblas::DenseParVec<T, F>>> &cr_tvs) {
  auto R = piecewise_constant_restriction<T, F, DER<T, F>>(state_vec, false);
  auto P = R;
  P.Transpose();

  struct NZAdd {
    T operator()(F a, T b) { return ((a != 0) ? 1 : 0) + b; }
  };
  combblas::DenseParVec<T, F> agg_sizes = R.Reduce(combblas::Row, NZAdd(), 0);
  for (auto s : agg_sizes.data()) {
    assert(s > 0);
  }
  auto tvs = smooth_vectors(A, R, agg_sizes, num_tvs, iters);
  cr_tvs.push_back(tvs);

  return split_aggregates<T, F, DER>(A, R, P, tvs, state_vec, agg_sizes, norm,
                                     threshold, verbose);
}

} // namespace ligmg

namespace mxx {
template <class T> MXX_CUSTOM_TEMPLATE_STRUCT(ligmg::Id<T>, agg_id, id);
template <class T> MXX_CUSTOM_TEMPLATE_STRUCT(ligmg::MinMax<T>, min, max);
} // namespace mxx

namespace combblas {
template <class T, class F> struct promote_trait<T, ligmg::MinMax<F>> {
  typedef ligmg::MinMax<F> T_promote;
};
template <class F, class G>
struct promote_trait<ligmg::NeighborEdge<F, G>, std::pair<G, F>> {
  typedef F T_promote;
};
template <class F, class G> struct promote_trait<G, std::pair<G, F>> {
  typedef std::pair<G, F> T_promote;
};
} // namespace combblas
