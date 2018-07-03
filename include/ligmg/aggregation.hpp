#pragma once

#include <CombBLAS/CombBLAS.h>
#include <limits>
#include <utility>
#include "combblas_extra.hpp"
#include "levels.hpp"
#include "smoothing.hpp"
#include "util.hpp"

namespace ligmg {

template <class T> struct AggState;
// Status of a node during aggregation
// Seed: will cause adjacent neighbors to aggregate to it
// SeedNeighbor: A node directly aggregated to the seed
// Decided: already aggregated with a neighbor
// DecidedRoot: Seed that has aggregated with enough neighbors
// Undecided: node looking to aggregate with a neighbor
enum class ST { Seed, SeedNeighbor, Decided, DecidedRoot, Undecided };

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

inline std::ostream &operator<<(std::ostream &os, const ST &st) {
  if (st == ST::Seed) {
    os << "Seed";
  } else if (st == ST::Decided) {
    os << "Decided";
  } else if (st == ST::SeedNeighbor) {
    os << "SeedNeighbor";
  } else if (st == ST::DecidedRoot) {
    os << "DecidedRoot";
  } else if (st == ST::Undecided) {
    os << "Undecided";
  } else {
    assert(false);
  }
  return os;
}

// State of a node during aggregation
template <class T>
struct AggState {
  T ind{-1};  // who to aggregate to. If ind == this nodes index then this node
              // is a
              // seed
  ST state{ST::Undecided};
  explicit AggState(T) {}
  AggState() = default;
  AggState(T a, ST b) : ind(a), state(b) {}
};

template <class T>
std::ostream &operator<<(std::ostream &os, const AggState<T> &st) {
  os << "AggState(" << st.ind << ", " << st.state << ")";
  return os;
}

template <class T, class F, class DER>
struct AggregationLevel : Level<T, F, DER> {
  using Smoother = std::function<void(combblas::DenseParVec<T, F> &,
                                      const combblas::DenseParVec<T, F> &)>;
  combblas::SpParMat<T, F, DER> A;
  combblas::SpParMat<T, F, DER> P;
  combblas::SpParMat<T, F, DER> R;
  combblas::SpParMat<T, F, DER> strength;
  std::vector<combblas::DenseParVec<T, AggState<T>>> agg_states;
  std::vector<combblas::DenseParVec<T, T>> voting_for;
  // pre and post smoother are separate so they can do different iteration
  // counts
  Smoother presmoother;
  Smoother postsmoother;
  std::vector<std::vector<combblas::DenseParVec<T, F>>> cr_tvs;

  AggregationLevel(
      combblas::SpParMat<T, F, DER> A, combblas::SpParMat<T, F, DER> P,
      combblas::SpParMat<T, F, DER> R, combblas::SpParMat<T, F, DER> strength,
      std::vector<combblas::DenseParVec<T, AggState<T>>> agg_states,
      std::vector<combblas::DenseParVec<T, T>> voting_for, Smoother presmoother,
      Smoother postsmoother,
      std::vector<std::vector<combblas::DenseParVec<T, F>>> cr_tvs)
      : A(std::move(A)),
        P(std::move(P)),
        R(std::move(R)),
        strength(std::move(strength)),
        agg_states(std::move(agg_states)),
        voting_for(std::move(voting_for)),
        presmoother(std::move(presmoother)),
        postsmoother(std::move(postsmoother)),
        cr_tvs(std::move(cr_tvs)) {}

  virtual void do_level(combblas::DenseParVec<T, F> &x,
                        const combblas::DenseParVec<T, F> &b,
                        std::function<void(combblas::DenseParVec<T, F> &,
                                           const combblas::DenseParVec<T, F> &)>
                            recurse,
                        Options opts) {
    using SR = combblas::PlusTimesSRing<F, F>;

    // pre-smooth
    presmoother(x, b);

    // residual
    combblas::DenseParVec<T, F> r = residual(A, x, b);
    // restrict
    combblas::DenseParVec<T, F> bc = combblas::SpMV<SR>(R, r);

    // coarse solution guess
    combblas::DenseParVec<T, F> xc(bc.getcommgrid(), bc.getTotalLength(), 0);

    recurse(xc, bc);

    // interpolate
    combblas::DenseParVec<T, F> x2 = combblas::SpMV<SR>(P, xc);
    // residual correction
    x += x2;

    postsmoother(x, b);

    // constant correction
    if (opts.constant_correction) {
      for (auto &v : x.data()) v = v * 4.0 / 3.0;
    }
  }
  virtual bool is_exact() { return false; }
  virtual combblas::SpParMat<T, F, DER> &matrix() { return A; }
  virtual T nnz() { return A.getnnz(); }
  virtual T size() { return A.getnrow(); }
  virtual std::string name() { return "agg"; }
  virtual T work(Options opts) {
    return (opts.pre_smooth + opts.post_smooth + 1) * A.getnnz() + R.getnnz() +
           P.getnnz() + A.getnrow();
  }
  virtual std::shared_ptr<combblas::CommGrid> commgrid() {
    return A.getcommgrid();
  }
  virtual void dump(std::string basename) {
    write_petsc(A, basename + "A.petsc");
    write_petsc(P, basename + "P.petsc");
    write_petsc(R, basename + "R.petsc");
    write_petsc(strength, basename + "strength.petsc");
    for (int i = 0; i < agg_states.size(); i++) {
      write_petsc(agg_states[i],
                  basename + "agg_state_" + std::to_string(i) + ".petsc");
    }
    for (int i = 0; i < voting_for.size(); i++) {
      write_petsc(voting_for[i],
                  basename + "voting_for_" + std::to_string(i) + ".petsc");
    }
    for (int i = 0; i < cr_tvs.size(); i++) {
      for (T j = 0; j < cr_tvs[i].size(); j++) {
        write_petsc(cr_tvs[i][j], basename + "cr_tv_" + std::to_string(i) +
                                      "_" + std::to_string(j) + ".petsc");
      }
    }
  }
};

template <class T, class F, class DER>
combblas::SpParMat<T, F, DER> affinity_matrix(
    const combblas::SpParMat<T, F, DER> &A, int num_tvs, int smoothing_iters,
    double thresh, StrengthMetric str, AffScaling scaling) {
  auto commgrid = A.getcommgrid();
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);

  // Compute distance for each local edge
  combblas::SpParMat<T, F, DER> Aff = A;
  if (str == StrengthMetric::Affinity ||
      str == StrengthMetric::AlgebraicDistance) {
    // initialize test std::vectors
    auto tvs = create_tvs(A, num_tvs, true);

    // zero rhs
    auto b = const_vec(A, 0.0);
    // auto b = rand_vec(A);
    // orthagonalize(b);

    // smooth each test vector
    // TODO: smooth all vectors at once
    for (auto &tv : tvs) {
      for (int j = 0; j < smoothing_iters; j++) {
        jacobi(A, tv, b);
      }
    }

    // Push diagonal test std::vector entries to columns and rows
    std::vector<std::vector<F>> tvs_x(num_tvs);
    std::vector<std::vector<F>> tvs_y(num_tvs);

    for (int i = 0; i < num_tvs; i++) {
      tvs_x.at(i) = tvs.at(i).data();
      tvs_y.at(i) = tvs.at(i).data();
      mxx::bcast(tvs_x.at(i), commgrid->GetRankInProcCol(),
                 commgrid->GetRowWorld());
      mxx::bcast(tvs_y.at(i), commgrid->GetRankInProcRow(),
                 commgrid->GetColWorld());
    }

    // iterate over local storage
    dcsc_local_iter(Aff, [=](T &rowid, T &colid, F &val) {
      if (rowid + r_off != colid + c_off) {  // ignore diagonal entries
        if (str == StrengthMetric::Affinity) {
          F top_sum = 0;
          for (int x = 0; x < num_tvs; x++)
            top_sum += tvs_y.at(x).at(colid) * tvs_x.at(x).at(rowid);
          F bot1_sum = 0;
          for (int x = 0; x < num_tvs; x++)
            bot1_sum += tvs_y.at(x).at(colid) * tvs_y.at(x).at(colid);
          F bot2_sum = 0;
          for (int x = 0; x < num_tvs; x++)
            bot2_sum += tvs_x.at(x).at(rowid) * tvs_x.at(x).at(rowid);
          val = top_sum * top_sum / (bot1_sum * bot1_sum * bot2_sum * bot2_sum);
        } else {
          std::vector<F> vals;
          for (int x = 0; x < num_tvs; x++) {
            vals.emplace_back(
                abs(tvs_y.at(x).at(colid) - tvs_x.at(x).at(rowid)));
          }
          val = 1 / (*max_element(vals.begin(), vals.end()));
        }
      } else {
        val = 0;
      }
    });
  } else {
    Aff.Apply([](const F &x) { return (x < 0) ? -x : 0; });
  }

  // row or column scaling
  combblas::FullyDistVec<T, F> sums(
      Aff.getcommgrid(), Aff.getnrow(),
      0);  // DimApply only works on combblas::FullyDistVec
  if (scaling == AffScaling::Row || scaling == AffScaling::RowColumn ||
      scaling == AffScaling::Sym) {
    Aff.Reduce(sums, combblas::Row, std::plus<F>(), 0.0);
    Aff.DimApply(combblas::Row, sums, [=](F a, F b) { return a / b; });
  }
  if (scaling == AffScaling::Column || scaling == AffScaling::RowColumn) {
    Aff.Reduce(sums, combblas::Column, std::plus<F>(), 0.0);
    Aff.DimApply(combblas::Column, sums, [=](F a, F b) { return a / b; });
  }
  if (scaling == AffScaling::Sym) {
    combblas::SpParMat<T, F, DER> Aff_copy = Aff;
    Aff_copy.Transpose();
    Aff =
        combblas::EWiseApply<F, DER>(Aff_copy, Aff, std::plus<F>(), false, 0.0);
    Aff.Apply([](const F &x) { return x / 2; });
  }
  if (scaling == AffScaling::Max) {
    // scale by maximum in column and row
    // DimApply only works on combblas::FullyDistVec
    combblas::FullyDistVec<T, F> col_max(commgrid, Aff.getnrow(), 0.0);
    combblas::FullyDistVec<T, F> row_max(commgrid, Aff.getnrow(), 0.0);
    Aff.Reduce(col_max, combblas::Column,
               [](F a, F b) { return std::max(a, b); }, 0.0, MPI_MAX);
    Aff.Reduce(row_max, combblas::Row, [](F a, F b) { return std::max(a, b); },
               0.0, MPI_MAX);

    // TODO: cleanup/optimize

    // Gather FullyDistVec to diagonal (same layout as DenseParVec) then
    // broadcast along proc cols
    auto local_col_max = mxx::gatherv(
        col_max.data(), commgrid->GetRankInProcCol(), commgrid->GetRowWorld());
    mxx::bcast(local_col_max, commgrid->GetRankInProcRow(),
               commgrid->GetColWorld());

    // all row entries already exist in the proc row
    auto local_row_max =
        mxx::allgatherv(row_max.data(), commgrid->GetRowWorld());

    assert(local_col_max.size() == Aff.getlocalcols());
    assert(local_row_max.size() == Aff.getlocalrows());

    // max from row and column
    dcsc_local_iter(Aff, [&](T &x, T &y, F &v) {
      v = v / std::max(local_row_max.at(x), local_col_max.at(y));
    });
  }

  // thresholding doesn't seem to do much
  Aff.seq().Prune([=](F x) { return x < thresh; }, true);

  return Aff;
}

// State of a node during voting aggregation
// priority is Seed > Undecided > Decided
enum class VoteAggState { Seed, Undecided, Decided };

// comparison operator for VoteAggState
inline bool gt(VoteAggState a, VoteAggState b) {
  if (a == VoteAggState::Seed) return true;
  if (b == VoteAggState::Seed) return false;
  if (a == VoteAggState::Undecided) return true;
  if (b == VoteAggState::Undecided) return false;
  return true;
}

// Holds required information for combblas::SpMV voting
template <class T, class F>
struct VoteAgg {
  VoteAggState st{VoteAggState::Decided};  // Did we find a seed
  T ind{-1};                               // index we are voting for
  F weight{-1};                            // Weight of connection
  explicit VoteAgg(T) {}
  VoteAgg() = default;
  VoteAgg(VoteAggState s, T i, F w) : st(s), ind(i), weight(w) {}

  // static constexpr auto datatype = std::make_std::tuple(&VoteAgg<T,F>::ind,
  // &VoteAgg<T,F>::weight);
};

template <class T, class F>
inline std::ostream &operator<<(std::ostream &os, const VoteAgg<T, F> &va) {
  os << va.ind << " " << va.weight << std::endl;
  return os;
}

// We need a way to set the weight threshold for aggregation in the semiring.
// CombBLAS does not allow any state in its semirings, so we keep a global
// variable. This will cause issues if we intersperse multiple aggregation
// routines (this should never happen).
static double ITER_THRESH;

// Voting aggregation semiring
// * determines if a node can be aggregated to
// + selects the best choice of aggregate
template <class T, class F, bool neighborneighbor>
struct AggSR {
  static VoteAgg<T, F> id() { return {}; }
  static bool returnedSAID() { return false; }

  static VoteAgg<T, F> add(const VoteAgg<T, F> &arg1,
                           const VoteAgg<T, F> &arg2) {
    // if states are the same, choose the std::min. Otherwise choose best state:
    // Seed > Undecided > Decided
    if (arg1.st == arg2.st) return (arg1.weight > arg2.weight) ? arg1 : arg2;

    return gt(arg1.st, arg2.st) ? arg1 : arg2;
  }

  static VoteAgg<T, F> multiply(const AggState<T> &arg1, const F &weight) {
    if (arg1.state == ST::Seed ||
        (neighborneighbor && arg1.state == ST::SeedNeighbor)) {
      if (weight > ITER_THRESH) {
        return {VoteAggState::Seed, arg1.ind, weight};
      } else {
        return {VoteAggState::Decided, arg1.ind, weight};
      }
    } else if (arg1.state == ST::Decided || arg1.state == ST::DecidedRoot) {
      return {VoteAggState::Decided, arg1.ind, weight};
    } else if (arg1.state == ST::Undecided) {
      return {VoteAggState::Undecided, arg1.ind, weight};
    }
    assert(false || "unreachable");
    return {VoteAggState::Decided, arg1.ind, weight};
  }
  static void axpy(const F &a, const AggState<T> &x, VoteAgg<T, F> &y) {
    y = add(y, multiply(x, a));
  }
  static MPI_Op mpi_op() {
    static MPI_Op mpiop;
    static bool exists = false;
    if (exists)
      return mpiop;
    else {
      MPI_Op_create(MPI_func, true, &mpiop);
      exists = true;
      return mpiop;
    }
  }

  static void MPI_func(void *invec, void *inoutvec, int *len,
                       MPI_Datatype *datatype) {
    auto pinvec = static_cast<VoteAgg<T, F> *>(invec);
    auto pinoutvec = static_cast<VoteAgg<T, F> *>(inoutvec);
    for (int i = 0; i < *len; i++) {
      pinoutvec[i] = add(pinvec[i], pinoutvec[i]);
    }
  }
};

// Using the given affinity matrix, have each node vote for a neighbor to
// become a root or to find a neighboring root to aggregate to
//
// The actual voting routine is located in the AggSR struct
template <class T, class F, template <class, class> class S>
combblas::DenseParVec<T, VoteAgg<T, F>> vote(
    const combblas::SpParMat<T, F, S<T, F>> &affinity_matrix,
    const combblas::DenseParVec<T, AggState<T>> &state_vec, F threshold,
    bool neighborneighbor) {
  ITER_THRESH = threshold;  // set the global threshold
  if (neighborneighbor) {
    return combblas::SpMV<AggSR<T, F, true>>(affinity_matrix, state_vec);
  } else {
    return combblas::SpMV<AggSR<T, F, false>>(affinity_matrix, state_vec);
  }
}

// Communicate votes amongst all processes and tally votes
// Also records if only one node voted for a single other node
template <class T, class F>
std::pair<std::vector<F>, std::vector<T>> tally_votes(
    const combblas::DenseParVec<T, VoteAgg<T, F>> &votes,
    const combblas::DenseParVec<T, AggState<T>> &state_vec,
    const std::vector<F> &previous_votes, F iter_thresh) {
  // DesnseParVec only exists on the diagonal
  if (votes.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    std::vector<F> new_votes(
        previous_votes.size());  // have to count new votes separately,
                                 // otherwise we double count votes
                                 // from previous rounds

    // -1 indicates no one has voted for this node
    // -2 indicates more than one node has voted for this one
    // a positive number indicates the single node that voted for this one
    std::vector<T> voting_for(votes.data().size(), -1);

    for (T i = 0; i < votes.data().size(); i++) {
      auto ind = votes.data().at(i).ind;
      auto global_i = votes.offset() + i;
      if (state_vec.data()[i].state == ST::Undecided && ind >= 0 &&
          votes.data()[i].weight >= iter_thresh && ind != global_i) {
        new_votes.at(ind) += votes.data()[i].weight;

        voting_for.at(i) = ind;
      }
    }

    new_votes = mxx::allreduce(new_votes, std::plus<F>(),
                               votes.getcommgrid()->GetDiagWorld());
    // add old votes in
    for (T i = 0; i < new_votes.size(); i++)
      new_votes[i] += previous_votes[i];

    // communicate voting for
    auto x = mxx::allgatherv(voting_for, votes.getcommgrid()->GetDiagWorld());
    return std::make_pair(new_votes, x);
  }
  return std::make_pair(std::vector<F>(), std::vector<T>());
}

// Determine the external/internal edge weight ratio for every vertex
template <class T, class F, class DER>
combblas::DenseParVec<T, F> ext_int_ratio(
    const combblas::SpParMat<T, F, DER> &A,
    const combblas::DenseParVec<T, AggState<T>> &statuses) {
  // TODO: restructure as combblas::SpMV
  auto commgrid = A.getcommgrid();

  auto neighbor_ids = statuses.data();
  auto aggregate_ids = statuses.data();

  // send statuses down columns
  mxx::bcast(neighbor_ids, commgrid->GetRankInProcRow(),
             commgrid->GetColWorld());
  mxx::bcast(aggregate_ids, commgrid->GetRankInProcCol(),
             commgrid->GetRowWorld());

  std::vector<F> local_ext_sums(A.getlocalrows(), 0);
  std::vector<F> local_int_sums(A.getlocalrows(), 0);

  dcsc_global_local_iter(A, [&](T g_row, T l_row, T g_col, T l_col, F &val) {
    if (g_row != g_col) {
      auto my_agg = aggregate_ids.at(l_row).ind;
      switch (aggregate_ids.at(l_row).state) {
        case ST::Decided:
        case ST::DecidedRoot:
        case ST::Seed:
        case ST::SeedNeighbor:
          switch (neighbor_ids.at(l_col).state) {
            case ST::Decided:
            case ST::DecidedRoot:
            case ST::Seed:
            case ST::SeedNeighbor:
              if (neighbor_ids.at(l_col).ind ==
                  my_agg) {                       // internal connection
                local_int_sums.at(l_row) -= val;  // val is negative edge weight
              } else {
                local_ext_sums.at(l_row) -= val;  // val is negative edge weight
              }
              break;
            default:
              local_ext_sums.at(l_row) -= val;  // val is negative edge weight
              break;
          }
          break;
        default:
          break;
      }
    }
  });

  auto ext_sums = mxx::reduce(local_ext_sums, commgrid->GetRankInProcCol(),
                              std::plus<F>(), commgrid->GetRowWorld());
  auto int_sums = mxx::reduce(local_int_sums, commgrid->GetRankInProcCol(),
                              std::plus<F>(), commgrid->GetRowWorld());

  combblas::DenseParVec<T, F> ratios(commgrid, ext_sums.size(), 0, 0);

  if (commgrid->GetDiagWorld() != MPI_COMM_NULL) {
    for (std::size_t i = 0; i < ext_sums.size(); i++) {
      ratios.data().at(i) = ext_sums.at(i) / int_sums.at(i);
    }
  }

  return ratios;
}

template <class T>
void reject_aggregates(combblas::DenseParVec<T, AggState<T>> &states,
                       int64_t max_size) {
  auto commgrid = states.getcommgrid();
  if (commgrid->GetDiagWorld() != MPI_COMM_NULL) {
    // compute sizes of each aggregate
    std::vector<int> sizes(states.getTotalLength(), 0);
    for (auto st : states.data()) {
      switch (st.state) {
        case ST::Seed:
        case ST::Decided:
        case ST::DecidedRoot:
        case ST::SeedNeighbor:
          sizes.at(st.ind)++;
          break;
        default:
          break;
      }
    }
    mxx::allreduce(sizes, std::plus<int>(), commgrid->GetDiagWorld());

    // aggregate to vertex map
    std::map<T, std::vector<T>> root_map;
    for (T i = 0; i < states.data().size(); i++) {
      switch (states.data().at(i).state) {
        case ST::Decided:
        case ST::SeedNeighbor:
          root_map[states.data().at(i).ind].push_back(i);
          break;
        default:
          break;
      }
    }

    // remove vertices such that the size of the aggregate is <=
    // opts.max_agg_size
    // TODO: consider connection strength when removing vertices
    for (auto &p : root_map) {
      auto root = p.first;
      auto size = sizes[root];
      auto &connected = p.second;
      if (size > max_size) {
        T local_count = connected.size() * connected.size() / (size - 1);
        for (T i = 0; i < local_count; i++) {
          states.data().at(connected.at(i)) = AggState<T>();
        }

        // set root as decided
        auto root_local = root - states.offset();
        if (root_local >= 0 && root_local < states.data().size()) {
          states.data().at(root_local).state = ST::DecidedRoot;
        }
      }
    }
  }
}

// Count the number of undecided vertices
template <class T>
T count_undecided(combblas::DenseParVec<T, AggState<T>> &states) {
  T count = 0;
  for (auto st : states.data()) {
    if (st.state == ST::Undecided) {
      count++;
    }
  }

  return mxx::reduce(count, 0, std::plus<T>(),
                     states.getcommgrid()->GetWorld());
}

// Forms aggregates based on votes
template <class T, class F>
void mark_nodes(const combblas::DenseParVec<T, VoteAgg<T, F>> &votes_aggs,
                const std::vector<T> &voting_for, const std::vector<F> &votes,
                combblas::DenseParVec<T, AggState<T>> &state_vec,
                const combblas::DenseParVec<T, F> &col_sum, T off, Options opts,
                bool first_iter, bool neighborneighbor) {
  for (T i = 0; i < votes_aggs.data().size(); i++) {
    assert(i + off < votes.size());
    assert(state_vec.data().size() == votes_aggs.data().size());
    auto state = state_vec.data().at(i).state;
    if (state == ST::Decided || state == ST::DecidedRoot ||
        state == ST::SeedNeighbor) {
      // do nothing for decided nodes
    } else if (state == ST::Seed) {
      if (votes.at(i + off) > opts.max_agg_size) {  // if our seed is
                                                    // connected to enough
                                                    // nodes, make it decided
        state_vec.data().at(i).state = ST::DecidedRoot;
        // TODO: set all SeedNeighbors as Decided
      }
    } else if (voting_for.at(i + off) >= 0 && first_iter &&
               voting_for.at(voting_for.at(i + off)) == i + off) {
      // node voted for neighbor and neighbor voted for node
      // node with lower has becomes seed
      // if (std::hash<T>{}(i + off) < std::hash<T>{}(voting_for.at(i + off)))
      // {
      assert(i + off != voting_for.at(i + off));
      if (i + off < voting_for.at(i + off)) {
        state_vec.data().at(i).state = ST::Seed;
        state_vec.data().at(i).ind = i + off;
      } else {
        state_vec.data().at(i).state =
            (neighborneighbor) ? ST::SeedNeighbor : ST::Decided;
        state_vec.data().at(i).ind = voting_for.at(i + off);
      }
    } else if (votes_aggs.data().at(i).st ==
               VoteAggState::Seed) {  // Found a seed, connect to it
      state_vec.data().at(i).state =
          (neighborneighbor) ? ST::SeedNeighbor : ST::Decided;
      state_vec.data().at(i).ind = votes_aggs.data().at(i).ind;
    } else if (votes.at(i + off) >=
               opts.votes_to_seed * col_sum.data()[i]) {  // if node is voted
                                                          // for enough, make
                                                          // it a seed
      state_vec.data().at(i).state = ST::Seed;
      state_vec.data().at(i).ind = i + off;
    }
  }
}

// Force vertices out of thier aggregate if the sum of external edges weights
// incident to the vertex is a factor larger than the internal incident edges.
template <class T, class F, class DER>
void conductance_rejection(const combblas::SpParMat<T, F, DER> &A,
                           combblas::DenseParVec<T, AggState<T>> &state_vec,
                           const Options &opts) {
  T count = 0;
  // seems to work best when we just do it on the last iteration
  // TODO: move this into CR?
  auto ratios = ext_int_ratio(A, state_vec);
  for (T i = 0; i < state_vec.data().size(); i++) {
    if (ratios.data()[i] > opts.rejection_ratio &&
        (state_vec.data()[i].state == ST::Decided ||
         state_vec.data()[i].state == ST::SeedNeighbor)) {
      state_vec.data()[i] = {i + state_vec.offset(), (opts.rejection_to_seed)
                                                         ? ST::Seed
                                                         : ST::Undecided};
      count++;
    }
  }

  if (opts.verbose) {
    count = mxx::allreduce(count, std::plus<T>(), A.getcommgrid()->GetWorld());
    debug_print(opts, A.getcommgrid()->GetWorld())
        << "rejected " << count << " vertice(s)" << std::endl;
  }
}

// Assign each unaggregated vertex to its own aggregate. Also renumber
// aggregates so they are contiguous.
template <class T>
std::pair<combblas::DenseParVec<T, T>, T> renumber_unaggregated(
    const combblas::DenseParVec<T, AggState<T>> &state_vec) {
  T num_aggs;
  auto aggregates = state_vec_to_aggregates(state_vec);
  auto commgrid = state_vec.getcommgrid();

  if (commgrid->GetDiagWorld() != MPI_COMM_NULL) {
    // count total number of aggregates
    // each unaggregated vertex is its own aggregate
    T num_aggs_local = std::count_if(
        state_vec.data().begin(), state_vec.data().end(),
        [](const AggState<T> &st) {
          return st.state == ST::Undecided || st.state == ST::DecidedRoot ||
                 st.state == ST::Seed;
        });

    T glob_off =
        mxx::scan<T>(num_aggs_local, std::plus<T>(), commgrid->GetDiagWorld());
    glob_off -= num_aggs_local;

    // Renumber aggregates to be contiguous
    // TODO: Is it possible to avoid an N length vector on each process?
    T off = state_vec.offset();
    std::vector<T> renumbered_inds(aggregates.getTotalLength(), -1);
    for (T i = 0; i < aggregates.data().size(); i++) {
      switch (state_vec.data()[i].state) {
        case ST::DecidedRoot:
        case ST::Seed:
        case ST::Undecided:
          renumbered_inds[i + off] = glob_off;
          glob_off++;
          break;
      }
    }

    // Share new indices along the diagonal
    renumbered_inds =
        mxx::allreduce(renumbered_inds, [](T a, T b) { return std::max(a, b); },
                       commgrid->GetDiagWorld());

    for (T i = 0; i < aggregates.data().size(); i++) {
      T new_ind = renumbered_inds.at(state_vec.data()[i].ind);
      assert(new_ind >= 0);
      aggregates.data()[i] = new_ind;
    }

    num_aggs = glob_off;  // to broadcast later
  }

  mxx::bcast(num_aggs, commgrid->GetSize() - 1, commgrid->GetWorld());
  return std::make_pair(aggregates, num_aggs);
}

// Main aggregation routine. Somewhat based on LAMG
// Uses combblas::SpMV to do voting for best aggregation candidates
template <class T, class F, template <class, class> class S>
std::unique_ptr<AggregationLevel<T, F, S<T, F>>> aggregation(
    const combblas::SpParMat<T, F, S<T, F>> &A, Options opts) {
  auto commgrid = A.getcommgrid();
  mxx::comm comm(commgrid->GetWorld());
  auto &debug = debug_print(opts, comm);

  combblas::DenseParVec<T, AggState<T>> state_vec(
      commgrid, A.getlocalrows(), {-1, ST::Undecided}, {-1, ST::Undecided});

  // Assign indices
  T off;
  if (commgrid->GetDiagWorld() != MPI_COMM_NULL) {
    off = state_vec.getTypicalLocLength() * commgrid->GetDiagRank();
    for (T i = 0; i < state_vec.data().size(); i++) {
      state_vec.data().at(i).ind = i + off;
    }
  } else {
    off = std::numeric_limits<T>::min();
  }

  // Get affinity matrix
  debug << "Building affinity matrix..." << std::flush;
  auto Aff = affinity_matrix(A, opts.num_tvs, opts.tvs_smooth, opts.thresh,
                             opts.strength_metric, opts.aff_scaling);
  combblas::DenseParVec<T, F> col_sum(Aff.getcommgrid(), Aff.getnrow(), 0);
  Aff.Reduce(col_sum, combblas::Column, std::plus<F>(), 0.0);

  debug << "done" << std::endl;

  std::vector<F> votes;  // maybe use a more efficient type?
  T tl = state_vec.getTotalLength();
  if (commgrid->GetDiagWorld() != MPI_COMM_NULL) votes.resize(tl, 0);

  debug << "Voting iterations..." << std::flush;
  combblas::DenseParVec<T, VoteAgg<T, F>> votes_aggs(A.getcommgrid(), 0);

  // TODO: only store when requested
  std::vector<combblas::DenseParVec<T, AggState<T>>>
      agg_states;  // store aggregation states
  std::vector<combblas::DenseParVec<T, T>>
      voting_fors;  // store who votes for who
  std::vector<std::vector<combblas::DenseParVec<T, F>>> cr_tvs;

  // step our algorithm
  // for (int outer_iter = 0; outer_iter < 4; outer_iter++) {
  for (int iter = 0; iter < opts.agg_iters; iter++) {
    // set the threshold for this iteration
    F iter_thresh = opts.agg_init_factor * pow(opts.agg_iter_factor, iter);
    // perform the actual voting
    debug << "voting..." << std::endl;
    votes_aggs = vote(Aff, state_vec, iter_thresh, opts.neighbor_of_neighbor);

    debug << "tallying..." << std::endl;
    auto x = tally_votes(votes_aggs, state_vec, votes, iter_thresh);
    votes = x.first;
    auto &voting_for = x.second;

    debug << "marking..." << std::endl;
    // mark all seeds and neighbors as decided
    mark_nodes(votes_aggs, voting_for, votes, state_vec, col_sum, off, opts,
               iter == 0, opts.neighbor_of_neighbor);

    // Reject large aggregates
    if (opts.max_agg_size > 0) {
      debug << "splitting large aggregates..." << std::endl;
      reject_aggregates(state_vec, opts.max_agg_size);
    }

    if (iter >= opts.agg_iters / 2 && opts.rejection_ratio >= 0) {
      debug << "rejecting poor conductance vertices..." << std::endl;
      conductance_rejection(A, state_vec, opts);
    }

    // accumulate debugging information
    // TODO: only do this when asked for debugging info
    agg_states.push_back(state_vec);
    combblas::DenseParVec<T, T> tmp(commgrid, voting_for.size(), 0, 0);
    for (T i = 0; i < tmp.data().size(); i++) {
      tmp.data()[i] = voting_for[i];
    }
    voting_fors.push_back(tmp);

    auto undecided = count_undecided(state_vec);
    debug << undecided << " unaggregated vertices" << std::endl;

    debug << iter << "." << std::flush;
  }

  // if (outer_iter != 3) {
  //   debug << "CR splitting..." << std::flush;
  //   compatible_relaxation_roots(A, state_vec, opts.cr_threshold, opts.cr_tvs,
  //                               opts.cr_smoothing_iters, opts.verbose,
  //                               opts.cr_norm, cr_tvs);
  //   debug << "done" << std::endl;
  // }
  // }
  debug << "done" << std::endl;

  // Split aggregates that perform poorly
  // debug << "CR splitting..." << std::flush;
  // compatible_relaxation_splitting(A, state_vec, opts.cr_threshold,
  // opts.cr_tvs,
  //                                 opts.cr_smoothing_iters, opts.verbose,
  //                                 opts.cr_norm, cr_tvs);
  // debug << "done" << std::endl;

  // Renumber unaggregated vertices
  // auto aggs = renumber_unaggregated(state_vec);

  // Create P matrix
  debug << "Creating R..." << std::flush;
  auto R = piecewise_constant_restriction<T, F, S<T, F>>(state_vec, true);
  remove_zero_rows(R);
  debug << "done" << std::endl;
  combblas::SpParMat<T, F, S<T, F>> P(R);
  P.Transpose();

  auto presmoother =
      create_smoother(A, opts.pre_smooth, opts.smoother_type, opts);
  auto postsmoother =
      create_smoother(A, opts.post_smooth, opts.smoother_type, opts);

  return std::unique_ptr<AggregationLevel<T, F, S<T, F>>>(
      new AggregationLevel<T, F, S<T, F>>(A, P, R, Aff, agg_states, voting_fors,
                                          presmoother, postsmoother, cr_tvs));
}
}  // namespace ligmg

namespace mxx {
template <>
struct datatype_builder<ligmg::ST> {
  static MPI_Datatype get_type() {
    return MPI_INT;  // TODO: XXX: THIS IS A MAJOR HACK. ST is not guaranteed to
                     // be 32 bytes
  }
  static size_t num_basic_elements() { return 1; }
};

template <class T>
MXX_CUSTOM_TEMPLATE_STRUCT(ligmg::AggState<T>, ind, state);
}  // namespace mxx

namespace combblas {
template <class T>
struct promote_trait<T, ligmg::AggState<int64_t>> {
  typedef ligmg::VoteAgg<int64_t, T> T_promote;
};
template <class T>
struct promote_trait<T, ligmg::AggState<int>> {
  typedef ligmg::VoteAgg<int, T> T_promote;
};
}  // namespace combblas
