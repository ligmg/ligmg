#include <CombBLAS/CombBLAS.h>
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <mxx/reduction.hpp>
#include <numeric>
#include <ligmg/ligmg.hpp>
#include <set>
#include "parse_options.hpp"

// using INT_TYPE = int64_t;
using INT_TYPE = int;

#if ALLINEA_SAMPLE_SOLVE
#include "mapsampler_api.h"
#endif

int main(int argc, char* argv[]) {
  // set_terminate(handler2);
  // signal(SIGSEGV, handler);
  MPI_Init(&argc, &argv);
  PetscInitialize(NULL, NULL, 0, "");
  // PetscPopSignalHandler();

  auto rank = mxx::comm(MPI_COMM_WORLD).rank();
  srand(std::chrono::system_clock::now().time_since_epoch().count());
  // srand(0);

  auto opts = parse_options(argc, argv);

  {
    // read matrix from file
    auto A = combblas::SpParMat<INT_TYPE, double,
                                combblas::SpDCCols<INT_TYPE, double>>(
        MPI_COMM_WORLD);
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "Reading " << opts.load_file << "..." << std::flush;
    auto start = MPI_Wtime();
    A.ParallelReadMM(opts.load_file, opts.verbose);
    auto end = MPI_Wtime();
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "done (" << end - start << "s)" << std::endl;

    auto nnz = A.getnnz();
    auto size = A.getncol();
    setlocale(LC_NUMERIC, "en_US");
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0) {
      std::cout << "Read " << size << " x " << size << " matrix with " << nnz
                << " nonzeros" << std::endl;
    }

    // randomly permute matrix for load balancing
    if (opts.rand_perm && mxx::comm(MPI_COMM_WORLD).size() > 1) {
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Randomly permuting matrix..." << std::flush;
      auto start = MPI_Wtime();

      combblas::FullyDistVec<INT_TYPE, INT_TYPE> p(A.getcommgrid(), size, 0);
      p.iota(A.getnrow(), 0);
      p.RandPerm();
      A(p, p, true);

      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (opts.cc) {
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Getting largest connected component..." << std::flush;
      auto start = MPI_Wtime();
      A = ligmg::largest_cc(A);
      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }

    // convert adjacency matrix to laplacian
    if (opts.adj) {
      auto start = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Converting adjacency matrix to laplacian..."
                  << std::flush;
      // reduce across rows to std::get weighted degree
      combblas::FullyDistVec<INT_TYPE, double> weighted_degrees(A.getcommgrid(),
                                                                A.getnrow(), 0);
      A.Reduce(weighted_degrees, combblas::Row, std::plus<double>(), 0.0);

      // create triples for diagonal matrix
      INT_TYPE offset = weighted_degrees.LengthUntil();

      std::vector<INT_TYPE> is(weighted_degrees.data().size());
      iota(is.begin(), is.end(), offset);

      std::vector<INT_TYPE> js(weighted_degrees.data().size());
      iota(js.begin(), js.end(), offset);

      combblas::SpParMat<INT_TYPE, double, combblas::SpDCCols<INT_TYPE, double>>
          D(A.getcommgrid());
      D.MatrixFromIJK(A.getnrow(), A.getncol(), is, js,
                      weighted_degrees.data());
      A = combblas::EWiseApply<double, combblas::SpDCCols<INT_TYPE, double>>(
          D, A, [](double a, double b) { return a - b; },
          [](double, double) { return true; }, true, true, 0.0, 0.0);
      auto end = MPI_Wtime();
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done (" << end - start << "s)" << std::endl;
    }

    // verify A is a graph Laplacian
    /*
    {
      dcsc_iter(A, [](INT_TYPE x, INT_TYPE y, double v) {
        if (x == y) {
          assert(v > 0);
        } else {
          assert(v < 0);
        }
      });

      combblas::FullyDistVec<INT_TYPE, double> weighted_degrees(A.getcommgrid(),
                                                      A.getnrow(), 0);
      A.Reduce(weighted_degrees, combblas::Row, std::plus<double>(), 0.0);
      for (auto x : weighted_degrees.data()) {
        assert(x < 1e-8 && x > -1e-8);
      }
    }
    */

    auto b = combblas::DenseParVec<INT_TYPE, double>(A.getcommgrid(),
                                                     A.seq().getnrow(), 0, 0);
    if (!opts.rhs.empty()) {
      b = ligmg::read_vec<INT_TYPE, double>(opts.rhs, A.getcommgrid());
      ligmg::orthagonalize(b);
    } else {
      b.Apply([](double) { return double(rand()) / RAND_MAX; });
      ligmg::orthagonalize(b);
    }

    nonstd::optional<ligmg::ConvergenceInfo<INT_TYPE, double>> ligmg_res;
    nonstd::optional<
        ligmg::Solver<INT_TYPE, double, combblas::SpDCCols<INT_TYPE, double>>>
        solver;
    if (opts.test_ligmg) {
      // construct hierarchy
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "Constructing hierarchy..." << std::flush;
      solver.emplace(A, opts);
      if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
        std::cout << "done" << std::endl;

      // print solver details
      solver->print();

      if (rank == 0)
        std::cout << "Setup time: " << solver->setup_time << "s" << std::endl;

      // vcycle-cg solve
      ligmg_res.emplace(solver->solve(b));

      if (rank == 0) {
        std::cout << *ligmg_res << std::endl;
      }
    }

    nonstd::optional<ligmg::ConvergenceInfo<INT_TYPE, double>> cg_res;
    if (opts.test_cg) {
      auto x = combblas::DenseParVec<INT_TYPE, double>(A.getcommgrid(),
                                                       A.seq().getnrow(), 0, 0);
      cg_res.emplace(petsc_fcg(
          A, x, b,
          [&](combblas::DenseParVec<INT_TYPE, double>& r) {
            r = combblas::SpMV<ligmg::InvDiagSR<double, double>>(A, r);
          },
          A.getnrow()/A.getnnz(), KSPCG, opts));
      std::cout << "PCG-Jacobi" << std::endl;
      std::cout << *cg_res << std::endl;
    }

    if (rank == 0 && opts.csv.size() > 0) {
      auto of = std::ofstream(opts.csv);
      ligmg::print_csv_header(of);
      if (opts.test_ligmg)
        ligmg::print_csv(of, "ligmg", solver->hierarchy[0]->nnz(),
                         solver->setup_time, *ligmg_res, opts);
      if (opts.test_cg) ligmg::print_csv(of, "pcg", nnz, 0, *cg_res, opts);
    }

    if (!opts.dump.empty()) {
      solver->dump_hierarchy(opts.dump);
      auto of = std::ofstream(opts.dump + "/params.csv");
      solver->dump_csv(of, *ligmg_res);
    }

    // for(int i = 0; i < 5; i++) {
    //   auto b =
    //       combblas::DenseParVec<INT_TYPE, double>(A.getcommgrid(),
    //       A.seq().getnrow(), 0, 0);
    //   b.Apply([](double x) { return double(rand()) / RAND_MAX; });
    //   ligmg::orthagonalize(b);
    //   auto res = solver.solve(b);

    //   if (rank == 0) {
    //     std::cout << res << std::endl;
    //   }
    // }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  PetscFinalize();
  MPI_Finalize();
}
