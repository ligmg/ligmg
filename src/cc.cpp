#include "cc.hpp"
#include <mxx/reduction.hpp>
#include "CombBLAS.h"
#include "OptionParser.h"

using namespace optparse;
using namespace ligmg;

struct Opts {
  std::string load_file;
  std::string out_file;
  bool laplacian;
};

Opts parse_opts(int argc, char* argv[]) {
  // parse arguments
  OptionParser parser =
      OptionParser().description("Connected Components on a graph");

  parser.add_option("--laplacian")
      .dest("laplacian")
      .set_default(false)
      .help("Output matrix as a laplacian (default is adjacency matrix)")
      .action("store_true");

  optparse::Values options = parser.parse_args(argc, argv);
  std::vector<std::string> args = parser.args();
  if (args.size() != 2) {
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "Must provide input and output filenames" << std::endl;
    exit(1);
  }

  return {args[0], args[1], bool(options.get("laplacian"))};
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  {
    auto opts = parse_opts(argc, argv);

    combblas::SpParMat<int64_t, double, combblas::SpDCCols<int64_t, double>> A(MPI_COMM_WORLD);

    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "Reading " << opts.load_file << "..." << std::flush;
    auto start = MPI_Wtime();
    A.ParallelReadMM(opts.load_file, false);
    auto end = MPI_Wtime();
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "done (" << end - start << "s)" << std::endl;

    auto nnz = A.getnnz();
    auto size = A.getncol();
    setlocale(LC_NUMERIC, "en_US");
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      printf("Read %'lld x %'lld matrix with %'lld nonzeros\n", size, size,
             nnz);

    {
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Getting largest connected component..." << std::flush;
      auto start = MPI_Wtime();
      A = largest_cc(A);
      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }

    // convert adjacency matrix to laplacian
    if (opts.laplacian) {
      auto start = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Converting adjacency matrix to laplacian..." << std::flush;
      // reduce across rows to std::get weighted degree
      combblas::FullyDistVec<int64_t, double> weighted_degrees(A.getcommgrid(),
                                                     A.getnrow(), 0);
      A.Reduce(weighted_degrees, combblas::Row, std::plus<double>(), 0.0);

      // create triples for diagonal matrix
      int64_t offset = weighted_degrees.LengthUntil();

      std::vector<int64_t> is(weighted_degrees.data().size());
      iota(is.begin(), is.end(), offset);

      std::vector<int64_t> js(weighted_degrees.data().size());
      iota(js.begin(), js.end(), offset);

      combblas::SpParMat<int64_t, double, combblas::SpDCCols<int64_t, double>> D(A.getcommgrid());
      D.MatrixFromIJK(A.getnrow(), A.getncol(), is, js,
                      weighted_degrees.data());
      A = combblas::EWiseApply<double, combblas::SpDCCols<int64_t, double>>(
          D, A, [](double a, double b) { return a - b; },
          [](double a, double b) { return true; }, true, true, 0.0, 0.0);
      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }

    {
      auto start = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Writing matrix..." << std::flush;
      A.SaveGathered(opts.out_file);
      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }
  }
  MPI_Finalize();
}
