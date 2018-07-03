#pragma once

#include <CombBLAS/CombBLAS.h>
#include <execinfo.h>
#include <libgen.h>
#include <petscmat.h>
#include <petscviewer.h>
#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include "options.hpp"

namespace ligmg {

// Orthagonalize a std::vector against the zero std::vector.
template <class T, class F>
void orthagonalize(combblas::DenseParVec<T, F> &v) {
  double sum = v.Reduce(std::plus<double>(), 0);
  sum = sum / v.getTotalLength();
  v.Apply([=](double x) { return x - sum; });
}

// from
// https://{stackoverflow.com/questions/11826554/standard-no-op-output-stream/11826666#11826666
class NullBuffer : public std::streambuf {
 public:
  int overflow(int c) { return c; }
};
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(&m_sb) {}

 private:
  NullBuffer m_sb;
};

std::ostream &debug_print(const Options &opts, const mxx::comm &comm) {
  static NullStream null_stream;
  if (comm.rank() == 0 && opts.verbose)
    return std::cout;
  else
    return null_stream;
}

void handler(int sig) {
  void *array[10];
  size_t size;

  // std::get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

void handler2() {
  void *array[10];
  size_t size;

  // std::get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

template <class IT, class NT, class DER>
combblas::SpParMat<IT, NT, DER> Identity(
    std::shared_ptr<combblas::CommGrid> grid, IT size) {
  combblas::SpParMat<IT, NT, DER> ident(grid);
  std::vector<IT> i;
  std::vector<IT> j;
  std::vector<NT> k;
  if (grid->GetDiagWorld() != MPI_COMM_NULL) {
    auto rank = grid->GetDiagRank();
    auto dsize = grid->GetDiagSize();
    auto off = rank * size / dsize;
    auto loc_size = (rank == dsize - 1) ? size - off : size / dsize;
    for (IT x = 0; x < loc_size; x++) {
      i.push_back(x + off);
      j.push_back(x + off);
      k.push_back(1);
    }
  }
  ident.MatrixFromIJK(size, size, i, j, k);

  return ident;
}

template <class SR, class IT, class NT, class DER>
combblas::SpParMat<IT, NT, DER> triple_product(
    const combblas::SpParMat<IT, NT, DER> &A,
    const combblas::SpParMat<IT, NT, DER> &B,
    const combblas::SpParMat<IT, NT, DER> &C) {
  auto tmp =
      combblas::PSpGEMM<SR>(const_cast<combblas::SpParMat<IT, NT, DER> &>(A),
                            const_cast<combblas::SpParMat<IT, NT, DER> &>(
                                B));  // arguments really aren't modified
  return combblas::PSpGEMM<SR>(
      tmp, const_cast<combblas::SpParMat<IT, NT, DER> &>(C));
}

// Dot product of two vectors
template <class T, class F>
F dot(const combblas::DenseParVec<T, F> &a,
      const combblas::DenseParVec<T, F> &b) {
  F sum = 0;
  if (a.getcommgrid()->GetDiagWorld() != MPI_COMM_NULL) {
    T len = a.data().size();
    for (T i = 0; i < len; i++) {
      sum += a.data().at(i) * b.data().at(i);
    }
  }

  return mxx::allreduce(sum, std::plus<F>(), a.getcommgrid()->GetWorld());
}

// Compute b - A*x
template <class T, class F, class D, class Vec>
Vec residual(const combblas::SpParMat<T, F, D> &A, const Vec &x, const Vec &b) {
  // r = b-A*x
  Vec r = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(A, x);
  r.EWise(b, [](F a, F b) { return b - a; });

  return r;
}

template <class T, class F, class D, class Vec>
F residual_norm(const combblas::SpParMat<T, F, D> &A, const Vec &x,
                const Vec &b) {
  auto res = residual(A, x, b);
  auto r = sqrt(dot(res, res));

  return r;
}

template <class T, class F>
struct ConvergenceInfo {
  const std::vector<F> residuals;
  const int _iters;
  const double _time;
  const double _work;
  const combblas::DenseParVec<T, F> solution;

  // work per digit of accuracy
  double wda() const { return -1.0 / log10(ecf()); }
  // cycle complexity, the amount of work per cycle
  double cycle_complexity() const { return _work / _iters; }
  // work, the number of fine grid multiplies required to solve the problem
  double work() const { return _work; }
  // effective convergence factor, convergence factor normalized by cycle
  // complexity
  double ecf() const {
    return pow(rel_residual(), 1.0 / (_iters * cycle_complexity()));
  }
  // relative change in residual from initial solution to final solution
  double rel_residual() const { return residuals.back() / residuals.front(); }
  // convergence factor, the reduction in residual per cycle
  double cf() const { return pow(rel_residual(), 1.0 / _iters); }
  // total solve time
  double time() const { return _time; }
  // number of iterations
  int iters() const { return _iters; }
};

template <class T, class F>
std::ostream &operator<<(std::ostream &os, const ConvergenceInfo<T, F> &cv) {
  std::cout << "iters    error  comp  conv econv   wda     time" << std::endl;
  printf("%5d %.2e %5.2f %.3f %.3f %5.2f %.2e", cv.iters(), cv.rel_residual(),
         cv.cycle_complexity(), cv.cf(), cv.ecf(), cv.wda(), cv.time());
  return os;
}

// from
// http://stackoverflow.com/questions/6417817/easy-way-to-remove-extension-from-a-filename
std::string remove_ext(const std::string &fullname) {
  size_t lastindex = fullname.find_last_of(".");
  return fullname.substr(0, lastindex);
}

void print_csv_header(std::ostream &os) {
  print_options_csv_header(os);
  os << ','
     << "graph,solver,iters,rel_error,cycle_complexity,wda,setup_time,"
        "solve_time,nnz,nodes"
     << std::endl;
}

template <class T, class F>
void print_csv(std::ostream &os, const std::string &name, int64_t nnz,
               double setup_time, const ConvergenceInfo<T, F> &cv,
               Options opts) {
  auto tmp = strdup(opts.load_file.c_str());
  std::string graph = remove_ext(basename(tmp));
  os << opts << ',' << graph << ',' << name << ',' << cv.iters() << ','
     << cv.rel_residual() << ',' << cv.cycle_complexity() << ',' << cv.wda()
     << ',' << setup_time << ',' << cv.time() << ',' << nnz << ','
     << mxx::comm(MPI_COMM_WORLD).size() << std::endl;
}

// Write a std::vector to a file
template <class T>
void write_vec(T vec, std::string filename) {
  auto ofs = std::ofstream(filename);
  auto v = mxx::gatherv(vec.data(), 0, vec.getcommgrid()->GetWorld());
  for (auto e : v) {
    ofs << e << std::endl;
  }
}

// Gather combblas::DenseParVec to one process
template <class T, class F>
std::vector<F> gather_vec(const combblas::DenseParVec<T, F> &v) {
  return mxx::gatherv(v.data(), 0, v.getcommgrid()->GetWorld());
}

// Iterate over local entries in `combblas::SpParMat` with global indices
template <class T, class F, class DER, class Function>
void dcsc_iter(const combblas::SpParMat<T, F, DER> &A, Function f) {
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  if (A.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = A.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      for (T j = dcsc->cp[i]; j < dcsc->cp[i + 1]; ++j) {
        T rowid = dcsc->ir[j];
        f(rowid + r_off, colid + c_off, dcsc->numx[j]);
      }
    }
  }
}

// Iterate over local entries in `combblas::SpParMat` with global and local
// indices
template <class T, class F, class DER, class Function>
void dcsc_global_local_iter(const combblas::SpParMat<T, F, DER> &A,
                            Function f) {
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  if (A.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = A.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      for (T j = dcsc->cp[i]; j < dcsc->cp[i + 1]; ++j) {
        T rowid = dcsc->ir[j];
        f(rowid + r_off, rowid, colid + c_off, colid, dcsc->numx[j]);
      }
    }
  }
}

// Iterate over local entries in `combblas::SpParMat` with local indices
template <class T, class F, class DER, class Function>
void dcsc_local_iter(combblas::SpParMat<T, F, DER> &A, Function f) {
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  if (A.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = A.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      for (T j = dcsc->cp[i]; j < dcsc->cp[i + 1]; ++j) {
        T rowid = dcsc->ir[j];
        f(rowid, colid, dcsc->numx[j]);
      }
    }
  }
}

template <class T, class F, class DER, class Function>
void dcsc_local_iter(const combblas::SpParMat<T, F, DER> &A, Function f) {
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  if (A.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = A.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      for (T j = dcsc->cp[i]; j < dcsc->cp[i + 1]; ++j) {
        T rowid = dcsc->ir[j];
        f(rowid, colid, dcsc->numx[j]);
      }
    }
  }
}

template <class T, class F, class DER, class Function>
void dcsc_local_col_iter(const combblas::SpParMat<T, F, DER> &A, Function f) {
  T c_off;
  T r_off;
  A.GetPlaceInGlobalGrid(r_off, c_off);
  if (A.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = A.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      T colstart = dcsc->cp[i];
      T collen = dcsc->cp[i + 1] - colstart;
      f(colid, dcsc->ir + colstart, dcsc->numx + colstart, collen);
    }
  }
}

// Transposes vector data on a commgrid
template <class F>
std::vector<F> transpose_data(
    const std::shared_ptr<combblas::CommGrid> &commgrid,
    const std::vector<F> &vec) {
  auto target = commgrid->GetComplementRank();
  auto src = commgrid->GetRank();
  int recv_count = -1;
  int send_count = vec.size();
  MPI_Sendrecv(&send_count, 1, MPI_INT, target, 0, &recv_count, 1, MPI_INT,
               target, 0, commgrid->GetWorld(), MPI_STATUS_IGNORE);
  std::vector<F> recv_buf(recv_count);
  MPI_Sendrecv(vec.data(), send_count, combblas::MPIType<F>(), target, 0,
               recv_buf.data(), recv_count, combblas::MPIType<F>(), target, 0,
               commgrid->GetWorld(), MPI_STATUS_IGNORE);
  return recv_buf;
}

// Gather a FullyDistVec along rows or columns in the Commgrid
template <class T, class F>
std::vector<F> distribute_down(const combblas::FullyDistVec<T, F> &v,
                               combblas::Dim dim) {
  switch (dim) {
    case combblas::Column:
      return mxx::allgatherv(v.data(), v.getcommgrid()->GetColWorld());
      break;
    case combblas::Row: {
      // transpose vector on the commgrid
      // FIXME: this is currently broken. Last proc in each row has more items
      // and needs to send them to the right place
      auto target = v.getcommgrid()->GetComplementRank();
      auto src = v.getcommgrid()->GetRank();
      int recv_count = -1;
      int send_count = v.data().size();
      MPI_Sendrecv(&send_count, 1, MPI_INT, target, 0, &recv_count, 1, MPI_INT,
                   target, 0, v.getcommgrid()->GetWorld(), MPI_STATUS_IGNORE);
      std::vector<F> recv_buf(recv_count);
      MPI_Sendrecv(v.data().data(), send_count, combblas::MPIType<F>(), target,
                   0, recv_buf.data(), recv_count, combblas::MPIType<F>(),
                   target, 0, v.getcommgrid()->GetWorld(), MPI_STATUS_IGNORE);
      return mxx::allgatherv(recv_buf, v.getcommgrid()->GetRowWorld());
    } break;
    default:
      return std::vector<F>();
  }
}

// Gather a combblas::SpParMat into ijk std::vectors on proc 0
template <class T, class F, class DER>
std::tuple<std::vector<T>, std::vector<T>, std::vector<F>> gather_ijk(
    const combblas::SpParMat<T, F, DER> &L) {
  std::vector<T> is;
  std::vector<T> js;
  std::vector<F> ks;

  T c_off;
  T r_off;
  L.GetPlaceInGlobalGrid(r_off, c_off);
  if (L.seq().getnnz() > 0) {
    combblas::Dcsc<T, F> *dcsc = L.seq().GetDCSC();
    for (T i = 0; i < dcsc->nzc; ++i) {
      T colid = dcsc->jc[i];
      for (T j = dcsc->cp[i]; j < dcsc->cp[i + 1]; ++j) {
        T rowid = dcsc->ir[j];
        is.push_back(rowid + r_off);
        js.push_back(colid + c_off);
        ks.push_back(dcsc->numx[j]);
      }
    }
  }

  // gather triplets
  is = mxx::gatherv(is, 0, L.getcommgrid()->GetWorld());
  js = mxx::gatherv(js, 0, L.getcommgrid()->GetWorld());
  ks = mxx::gatherv(ks, 0, L.getcommgrid()->GetWorld());

  return std::make_tuple(is, js, ks);
}

struct MatDeleter {
  void operator()(Mat x) { MatDestroy(&x); }
};

using MatWrapper = std::unique_ptr<std::remove_pointer<Mat>::type, MatDeleter>;

// Convert an combblas::SpParMat to a PETSc Mat object
// Assumes that nnz of the combblas::SpParMat fits in PetscInt
template <template <class, class> class DER>
MatWrapper to_petsc(
    combblas::SpParMat<PetscInt, PetscScalar, DER<PetscInt, PetscScalar>> A) {
  // construct local matrix
  PetscInt x_size = A.getnrow();
  PetscInt y_size = A.getncol();
  PetscInt loc_x_size = A.getlocalrows();
  PetscInt loc_y_size = A.getlocalcols();

  A.Transpose();

  std::vector<PetscInt> col_sizes(x_size);
  dcsc_local_col_iter(A,
                      [&](PetscInt colid, PetscInt *rowinds, PetscScalar *vals,
                          PetscInt len) { col_sizes[colid] = len; });

  // construct petsc solver for local matrix
  // TODO: preallocate
  Mat res;
  MatCreateAIJ(A.getcommgrid()->GetWorld(), loc_x_size, loc_y_size, x_size,
               y_size, 0, col_sizes.data(), 10, NULL, &res);
  // MatSetOption(res, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  combblas::Dcsc<PetscInt, PetscScalar> *dcsc = A.seq().GetDCSC();
  // FIXME: add local row and column offsets to these
  // FIXME: only works for symmetric matrix
  // std::vector<PetscInt> i(dcsc->jc, dcsc->jc + dcsc->nzc);
  // std::vector<PetscInt> j(dcsc->ir, dcsc->ir + dcsc->cp[dcsc->nzc]);
  // MatMPIAIJSetPreallocationCSR(res, i.data(), j.data(), dcsc->numx);
  //

  dcsc_local_col_iter(A, [&](PetscInt colid, PetscInt *rowinds,
                             PetscScalar *vals, PetscInt len) {
    std::vector<PetscInt> rinds(rowinds, rowinds + len);
    std::vector<PetscScalar> vs(vals, vals + len);
    std::vector<PetscInt> cs = {PetscInt(colid)};
    MatSetValuesBlocked(res, 1, cs.data(), len, rinds.data(), vs.data(),
                        INSERT_VALUES);
  });

  // dcsc_local_iter(A, [&](T& x, T& y, F& v) {
  //     if(y %1000 == 0) { std::cout << y << std::endl;}
  //     MatSetValue(res, x, y, v, INSERT_VALUES);
  //     });

  MatAssemblyBegin(res, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(res, MAT_FINAL_ASSEMBLY);

  return MatWrapper(res);
}

// inplace combblas::SpMV for use with PETSc
template <class T, class F, class S>
PetscErrorCode SpMV_inplace(Mat mat, Vec x, Vec y) {
  // std::get pointer to combblas matrix
  combblas::SpParMat<T, F, S> *A;
  MatShellGetContext(mat, &A);

  // copy x data to combblas array
  // TODO: can I use VecPlaceArray and friends to avoid making a copy on each
  // combblas::SpMV
  const F *ary;
  VecGetArrayRead(x, &ary);
  combblas::DenseParVec<T, F> xx(A->getcommgrid(), A->getnrow());
  copy_n(ary, xx.getLocalLength(), xx.data().begin());
  VecRestoreArrayRead(x, &ary);

  // perform combblas::SpMV
  auto res = combblas::SpMV<combblas::PlusTimesSRing<F, F>>(*A, xx);

  // copy data back
  F *ary_ret;
  VecGetArray(y, &ary_ret);
  copy_n(res.data().begin(), xx.getLocalLength(), ary_ret);
  VecRestoreArray(y, &ary_ret);

  return 0;
}

template <class T, class F, class S>
PetscErrorCode getdiag(Mat mat, Vec x) {
  // std::get pointer to combblas matrix
  combblas::SpParMat<T, F, S> *A;
  MatShellGetContext(mat, &A);

  F *ary;
  VecGetArray(x, &ary);

  T c_off;
  T r_off;
  A->GetPlaceInGlobalGrid(r_off, c_off);
  auto seq = A->seq();
  for (auto colit = seq.begcol(); colit != seq.endcol();
       ++colit) {  // iterate over columns
    T j = colit.colid();
    for (auto nzit = seq.begnz(colit); nzit != seq.endnz(colit); ++nzit) {
      T i = nzit.rowid();
      F k = nzit.value();
      if (r_off + i == c_off + j) {
        ary[i] = k;
      }
    }
  }

  VecRestoreArray(x, &ary);

  return 0;
}

// wrap a combblas::SpParMat in a PETSc MatShell. This Mat is valid only for the
// lifetime of the combblas SpParMat
template <template <class, class> class DER>
Mat wrap_combblas_mat(const combblas::SpParMat<PetscInt, PetscScalar,
                                               DER<PetscInt, PetscScalar>> *A) {
  PetscInt lrows = (A->getcommgrid()->GetDiagWorld() == MPI_COMM_NULL)
                       ? 0
                       : A->getlocalrows();
  PetscErrorCode ierr;
  Mat mat;
  MatCreateShell(A->getcommgrid()->GetWorld(), lrows, lrows, A->getnrow(),
                 A->getncol(), (void *)A,
                 &mat);  // TODO: need to add destructor for this shell
  MatShellSetOperation(
      mat, MATOP_MULT,
      (void (*)(void))
          SpMV_inplace<PetscInt, PetscScalar, DER<PetscInt, PetscScalar>>);
  MatShellSetOperation(
      mat, MATOP_GET_DIAGONAL,
      (void (*)(
          void))getdiag<PetscInt, PetscScalar, DER<PetscInt, PetscScalar>>);

  // set nullspace
  MatNullSpace nullsp;
  MatNullSpaceCreate(A->getcommgrid()->GetWorld(), PETSC_TRUE, 0, PETSC_NULL,
                     &nullsp);
  MatSetNullSpace(mat, nullsp);
  // TODO: handle destruction

  return mat;
}

template <class T, class F>
struct Triplet {
  T row;
  T col;
  F value;
  Triplet(T r, T c, F v) : row(r), col(c), value(v) {}
  Triplet() : row(-1), col(-1), value(0) {}
};

// convert a sequence of (row, col, value) triples into a PETSc Mat
MatWrapper to_petsc_local(
    PetscInt m, const std::vector<Triplet<PetscInt, PetscScalar>> &triplets) {
  Mat A;
  MatCreateSeqAIJ(PETSC_COMM_SELF, m, m, PETSC_DEFAULT, NULL, &A);
  for (auto t : triplets) {
    MatSetValue(A, t.row, t.col, t.value, INSERT_VALUES);
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  return MatWrapper(A);
}

// Create a PETSc Mat from arrays without copying
MatWrapper petsc_view(PetscInt m, PetscInt n, std::vector<PetscInt> &rowptr,
                      std::vector<PetscInt> &colvals,
                      std::vector<PetscScalar> &vals) {
  Mat A;
  MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, rowptr.data(),
                            colvals.data(), vals.data(), &A);
  return MatWrapper(A);
}

// Iterate through local contents of a PETSc Mat
template <class Function>
void petsc_row_local_iter(Mat A, Function func) {
  PetscInt m, n;
  MatGetSize(A, &m, &n);
  for (PetscInt i = 0; i < m; i++) {
    const PetscInt **cols;
    const PetscScalar **vals;
    PetscInt ncols;
    MatGetRow(A, i, &ncols, cols, vals);
    for (PetscInt j = 0; j < ncols; j++) {
      func(i, cols[j], vals[j]);
    }
    MatRestoreRow(A, i, &ncols, cols, vals);
  }
}

template <class T, class F, class DER>
void write_petsc(const combblas::SpParMat<T, F, DER> &A, std::string filename) {
  auto mat = to_petsc(A);

  if (A.getcommgrid()->GetRank() == 0) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(MPI_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE,
                          &viewer);
    MatView(mat.get(), viewer);
  }
}

template <class T, class F>
T proc_index(const combblas::DenseParVec<T, F> &vec, T index) {
  assert(index < vec.getTotalLength());
  T target = index / vec.getTypicalLocLength();
  T diag_size = vec.getcommgrid()->GetDiagSize();
  if (target >= diag_size) {
    target = diag_size - 1;
  }
  return target;
}

template <class T, class F, class DER>
T proc_index_col(const combblas::SpParMat<T, F, DER> &mat, T index) {
  assert(index < mat.getncol());
  T target = index / (mat.getncol() / mat.getcommgrid()->GetGridCols());
  T col_size = mat.getcommgrid()->GetGridCols();
  if (target >= col_size) {
    target = col_size - 1;
  }
  return target;
}

template <class T, class F, class DER>
void remove_zero_rows(combblas::SpParMat<T, F, DER> &mat) {
  combblas::FullyDistVec<T, F> row_sums(mat.getcommgrid(), mat.getnrow(), 0.0);
  mat.Reduce(row_sums, combblas::Row, std::plus<F>(), 0.0);
  auto rows = row_sums.FindInds([](F x) { return x != 0; });
  auto cols = combblas::DenseParVec<T, T>::generate(
      mat.getcommgrid(), mat.getncol(), [](T i) { return i; });

  mat(rows, cols, true);
}

template <class T, class F>
combblas::DenseParVec<T, F> read_vec(
    const std::string &filename, std::shared_ptr<combblas::CommGrid> commgrid) {
  std::ifstream ifs(filename);
  int size;
  ifs >> size;
  combblas::DenseParVec<T, F> vec(commgrid, size, 0.0);
  for (int i = 0; i < size; i++) {
    F x;
    ifs >> x;
    if (i >= vec.offset() && i < vec.offset() + vec.getLocalLength()) {
      vec.data()[i - vec.offset()] = x;
    }
  }
  return vec;
}

// from
// https://github.com/geodynamics/pylith/blob/master/libsrc/pylith/utils/error.h
#define PETSC_CHECK_ERROR(err)                                             \
  do {                                                                     \
    if (PetscUnlikely(err)) {                                              \
      PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, \
                 err, PETSC_ERROR_REPEAT, 0);                              \
      throw std::runtime_error("Error detected while in PETSc function."); \
    }                                                                      \
  } while (0)

}  // namespace ligmg

namespace mxx {
template <class T, class F>
MXX_CUSTOM_TEMPLATE_STRUCT(MXX_WRAP_TEMPLATE(ligmg::Triplet<T, F>), row, col,
                           value);
}
