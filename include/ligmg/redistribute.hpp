#pragma once

#include "levels.hpp"
#include <CombBLAS/CombBLAS.h>

namespace ligmg {

// check if we should redistribute
template <class T, class F, class DER>
bool should_redist(combblas::SpParMat<T, F, DER> &A, T bound) {
  // TODO: use nrows instead?
  return floor(sqrt(A.getnnz() / bound)) < A.getcommgrid()->GetGridRows() &&
         A.getcommgrid()->GetSize() > 1;
  // return A.getnnz() > bound*A.getcommgrid()->GetSize(); // upper bound work?
  // TODO: lower bound instead?
}

// Moves a matrix onto a smaller communication grid
template <class T, class F, class DER>
std::shared_ptr<combblas::CommGrid> redist(combblas::SpParMat<T, F, DER> &A,
                                           T bound, bool verbose) {
  auto commgrid = A.getcommgrid();
  T nnz = A.getnnz();
  // x^2 * bound = nnz
  // sqrt(nnz/bound) = x
  T old_rows = commgrid->GetGridRows();
  T nrows = std::max<T>(T(floor(sqrt(F(nnz) / bound))), 1); // TODO: use ceil?
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "Restricting commgrid from " << commgrid->GetSize() << " to "
              << nrows * nrows << std::endl;

  // construct new communicator
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "Constructing new commgrid..." << std::flush;
  bool in_new_comm = commgrid->GetRankInProcRow() < nrows &&
                     commgrid->GetRankInProcCol() < nrows;
  // std::cout << commgrid->GetRankInProcRow() << " " <<
  // commgrid->GetRankInProcCol()
  // << " " << commgrid->GetRank() << " " << (in_new_comm?"true":"false") <<
  // std::endl;
  MPI_Comm new_comm;
  auto mxx_comm = mxx::comm(commgrid->GetWorld());
  MPI_Comm_split(commgrid->GetWorld(), in_new_comm ? 1 : MPI_UNDEFINED,
                 commgrid->GetRank(), &new_comm);
  auto new_commgrid =
      (new_comm != MPI_COMM_NULL)
          ? std::make_shared<combblas::CommGrid>(new_comm, nrows, nrows)
          : std::shared_ptr<combblas::CommGrid>();
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "done." << std::endl;

  if (verbose && commgrid->GetRank() == 0)
    std::cout << "Redistributing matrix..." << std::flush;
  // std::get local elements
  std::vector<T> is;
  std::vector<T> js;
  std::vector<F> ks;
  // to appease mxx. Needs non-NULL pointer
  auto local_nnz = std::max<T>(A.seq().getnnz(), 1);
  is.reserve(local_nnz);
  js.reserve(local_nnz);
  ks.reserve(local_nnz);
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  auto seq = A.seq();
  for (auto colit = seq.begcol(); colit != seq.endcol();
       ++colit) { // iterate over columns
    T j = colit.colid();
    for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
      T i = nzit.rowid();
      F k = nzit.value();
      is.push_back(r_off + i);
      js.push_back(c_off + j);
      ks.push_back(k);
    }
  }

  // redistribute matrix
  // Each block outside on smaller communicator sends its data to rank %
  // nrows*nrows
  // TODO: better places to send?
  if (!in_new_comm) {
    int dest = (commgrid->GetRankInProcRow() % nrows) +
               (commgrid->GetRankInProcCol() % nrows) * old_rows;
    // std::cout << commgrid->GetRank() << " sending to " << dest << " " <<
    // is.size()
    // << std::endl;
    mxx_comm.send(is, dest, 1);
    mxx_comm.send(js, dest, 2);
    mxx_comm.send(ks, dest, 3);
  } else {
    for (T i = 0; i < commgrid->GetSize(); i++) {
      auto pc = commgrid->GetRankInProcCol(i);
      auto pr = commgrid->GetRankInProcRow(i);
      if ((pc >= nrows || pr >= nrows) &&
          pr % nrows + (pc % nrows) * old_rows == commgrid->GetRank()) {
        // std::cout << commgrid->GetRank() << " recving from " << i <<
        // std::endl;
        std::vector<T> new_is = mxx_comm.recv<std::vector<T>>(i, 1);
        std::vector<T> new_js = mxx_comm.recv<std::vector<T>>(i, 2);
        std::vector<F> new_ks = mxx_comm.recv<std::vector<F>>(i, 3);
        is.insert(is.end(), new_is.begin(), new_is.end());
        js.insert(js.end(), new_js.begin(), new_js.end());
        ks.insert(ks.end(), new_ks.begin(), new_ks.end());
      }
    }
  }

  mxx_comm.barrier(); // TODO: needed?
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "sent entries..." << std::flush;

  auto rows = A.getnrow();
  if (new_commgrid) {
    combblas::SpParMat<T, F, DER> A_new(new_commgrid);
    A_new.MatrixFromIJK(rows, rows, is, js, ks, false);

    A = A_new;
  }
  if (verbose && commgrid->GetRank() == 0)
    std::cout << "done" << std::endl;
  mxx_comm.barrier(); // TODO: needed?

  return new_commgrid;
}

// redistribute std::vector to match matrix. Can either distribute up or down.
template <class T, class F>
void redist(std::shared_ptr<combblas::CommGrid> from_grid,
            std::shared_ptr<combblas::CommGrid> to_grid,
            const combblas::DenseParVec<T, F> &from,
            combblas::DenseParVec<T, F> &to, bool shrink, bool verbose) {
  int rank_in_from = -1;
  int rank_in_to = -1;
  if (from_grid && from_grid->GetDiagWorld() != MPI_COMM_NULL)
    rank_in_from = from_grid->GetDiagRank();
  if (to_grid && to_grid->GetDiagWorld() != MPI_COMM_NULL)
    rank_in_to = to_grid->GetDiagRank();

  // ignore no diagonals
  if (rank_in_to != -1 || rank_in_from != -1) {
    // larger communicator. All communication is done here
    mxx::comm mxx_comm(shrink ? from_grid->GetDiagWorld()
                              : to_grid->GetDiagWorld());

    T from_loc_len = -1;
    if (rank_in_from != -1)
      from_loc_len = from.getTypicalLocLength();
    mxx::bcast(from_loc_len, 0,
               mxx_comm); // assume 0 rank is the same on both comms
    T to_loc_len = -1;
    if (rank_in_to != -1)
      to_loc_len = to.getTypicalLocLength();
    mxx::bcast(to_loc_len, 0,
               mxx_comm); // assume 0 rank is the same on both comms

    T to_comm_size = (rank_in_to != -1) ? to_grid->GetGridRows() : -1;
    mxx::bcast(to_comm_size, 0, mxx_comm);
    T from_comm_size = (rank_in_from != -1) ? from_grid->GetGridRows() : -1;
    mxx::bcast(from_comm_size, 0, mxx_comm);

    mxx::requests reqs; // to wait on asyncs
    std::vector<mxx::future<void>>
        futures; // need to hold onto futures, they wait
                 // when they go out of scope

    // send data
    if (rank_in_from != -1) {
      T off = from_loc_len * rank_in_from; // local offset in global std::vector
      T other_off =
          to_loc_len * rank_in_to; // local offset in global std::vector

      // smallest process this proc can send to is this procs smallest element
      // index / to_loc_len
      T smallest_proc = rank_in_from * from_loc_len / to_loc_len;
      // largest proc this proc can send to is this procs largest element index
      // / to_loc_len
      T largest_proc =
          std::min(((rank_in_from + 1) * from_loc_len - 1) / to_loc_len,
                   to_comm_size - 1);

      for (T i = smallest_proc; i <= largest_proc; i++) {
        // range to send
        T other_smallest_idx = i * to_loc_len;
        T other_largest_idx = (i == to_comm_size - 1) ? from.getTotalLength()
                                                      : (i + 1) * to_loc_len;
        T local_smallest_idx = std::max<T>(other_smallest_idx, off);
        T local_largest_idx =
            std::min<T>(other_largest_idx, off + from.data().size());

        if (i == rank_in_to) { // copy instead of send to self
          assert(from.data().size() >=
                 (local_smallest_idx - off) +
                     (local_largest_idx - local_smallest_idx));
          assert(to.data().size() >=
                 (local_smallest_idx - other_off) +
                     (local_largest_idx - local_smallest_idx));
          copy_n(from.data().begin() + (local_smallest_idx - off),
                 local_largest_idx - local_smallest_idx,
                 to.data().begin() + (local_smallest_idx - other_off));
        } else {
          if (verbose)
            std::cout << mxx_comm.rank() << " sending to " << i << " "
                      << local_smallest_idx << ":" << local_largest_idx
                      << std::endl;
          assert(from.data().size() >=
                 (local_smallest_idx - off) +
                     (local_largest_idx - local_smallest_idx));
          // TODO: check when future is forced
          // isend(i, from.data().data() + (local_smallest_idx - off),
          //          local_largest_idx - local_smallest_idx);
          futures.emplace_back(std::move(
              mxx_comm.isend<F>(from.data().data() + (local_smallest_idx - off),
                                local_largest_idx - local_smallest_idx, i)));
        }
      }
    }

    // receive data
    if (rank_in_to != -1) {
      T off = to_loc_len * rank_in_to; // local offset in global std::vector
      T other_off =
          from_loc_len * rank_in_from; // local offset in global std::vector

      // smallest process this proc can send to is this procs smallest element
      // index / to_loc_len
      T smallest_proc = rank_in_to * to_loc_len / from_loc_len;
      // largest proc this proc can send to is this procs largest element index
      // / to_loc_len
      T largest_proc =
          std::min(((rank_in_to + 1) * to_loc_len - 1) / from_loc_len,
                   from_comm_size - 1);

      for (T i = smallest_proc; i <= largest_proc; i++) {
        // range to send
        T other_smallest_idx = i * from_loc_len;
        T other_largest_idx = (i == from_comm_size - 1)
                                  ? to.getTotalLength()
                                  : (i + 1) * from_loc_len;
        T local_smallest_idx = std::max<T>(other_smallest_idx, off);
        T local_largest_idx =
            std::min<T>(other_largest_idx, off + to.data().size());

        if (i == rank_in_from) {
          // do nothing, local copy has already happened
        } else {
          if (verbose)
            std::cout << mxx_comm.rank() << " receiving from " << i << " "
                      << local_smallest_idx << ":" << local_largest_idx
                      << std::endl;
          assert(to.data().size() >=
                 (local_smallest_idx - off) +
                     (local_largest_idx - local_smallest_idx));
          // TODO: try async
          // irecv(i, to.data().data() + (local_smallest_idx - off),
          //       local_largest_idx - local_smallest_idx);
          futures.emplace_back(std::move(
              mxx_comm.irecv_into(to.data().data() + (local_smallest_idx - off),
                                  local_largest_idx - local_smallest_idx, i)));
        }
      }
    }

    // std::cout << mxx_comm.rank() << "waiting" << std::endl;
    // MPI_Ibarrier(mxx_comm, &reqs.add());
    // reqs.wait();
    // std::cout << mxx_comm.rank() << "done" << std::endl;
  }

  // if (shrink)
  //   MPI_Barrier(from_grid->GetWorld());
  // else
  //   MPI_Barrier(to_grid->GetWorld());
}

class undefined_t {};

template <class T, class F, class DER>
struct RedistributeLevel : Level<T, F, DER> {
  std::shared_ptr<combblas::CommGrid> fine_comm;
  std::shared_ptr<combblas::CommGrid> coarse_comm;
  RedistributeLevel(std::shared_ptr<combblas::CommGrid> fine_comm,
                    std::shared_ptr<combblas::CommGrid> coarse_comm)
      : fine_comm(std::move(fine_comm)), coarse_comm(std::move(coarse_comm)) {}
  RedistributeLevel() = delete;

  virtual void do_level(combblas::DenseParVec<T, F> &x,
                        const combblas::DenseParVec<T, F> &b,
                        std::function<void(combblas::DenseParVec<T, F> &,
                                           const combblas::DenseParVec<T, F> &)>
                            recurse,
                        Options opts) {
    combblas::DenseParVec<T, F> xc(coarse_comm, x.getTotalLength(), 0);
    combblas::DenseParVec<T, F> bc(coarse_comm, x.getTotalLength(), 0);
    redist(fine_comm, coarse_comm, x, xc, true, opts.verbose);
    redist(fine_comm, coarse_comm, b, bc, true, opts.verbose);

    if (coarse_comm) {
      // only continue on the coarse communicator
      recurse(xc, bc);
    }

    redist(coarse_comm, fine_comm, xc, x, false, opts.verbose);
  }
  virtual bool is_exact() { return true; }
  virtual T nnz() { return -1; }
  virtual T size() { return -1; }
  virtual T work(Options opts) { return 0; }
  virtual combblas::SpParMat<T, F, DER> &matrix() { throw undefined_t(); }
  virtual std::string name() { return "redist"; }
  virtual std::shared_ptr<combblas::CommGrid> commgrid() { return fine_comm; }
  virtual void dump(std::string basename) {}
};
}
