#include <CombBLAS.h>
#include <mpi.h>

// Benchmark combblas::SpMV for a given matrix
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  {

    combblas::SpParMat<int, double, combblas::SpDCCols<int,double>> A(MPI_COMM_WORLD);
    A.ParallelReadMM(argv[1]);

    combblas::DenseParVec<int, double> x_diag(A.getcommgrid(), A.getnrow());
    combblas::FullyDistVec<int, double> x_dist(A.getcommgrid(), A.getnrow(), 0);

    int warm_up = 2;
    int iters = 100;

    auto rank = A.getcommgrid()->GetRank();

    if(rank == 0) std::cout << "Warming up diag" << std::endl;
    for(int i = 0; i < warm_up; i++) {
      auto res = combblas::SpMV<combblas::PlusTimesSRing<double, double>>(A, x_diag);
    }
    if(rank == 0) std::cout << "Benching diag" << std::endl;
    auto start_diag = MPI_Wtime();
    for(int i = 0; i < iters; i++) {
      auto res = combblas::SpMV<combblas::PlusTimesSRing<double, double>>(A, x_diag);
    }
    auto end_diag = MPI_Wtime();

    if(rank == 0) std::cout << "Warming up dist" << std::endl;
    for(int i = 0; i < warm_up; i++) {
      auto res = combblas::SpMV<combblas::PlusTimesSRing<double, double>>(A, x_dist);
    }
    if(rank == 0) std::cout << "Benching dist" << std::endl;
    auto start_dist = MPI_Wtime();
    for(int i = 0; i < iters; i++) {
      auto res = combblas::SpMV<combblas::PlusTimesSRing<double, double>>(A, x_dist);
    }
    auto end_dist = MPI_Wtime();

    if(rank == 0) {
      std::cout << "Diag " << (end_diag - start_diag)/iters << std::endl;
      std::cout << "Dist " << (end_dist - start_dist)/iters << std::endl;
    }
  }

  MPI_Finalize();
}
