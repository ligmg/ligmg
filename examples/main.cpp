#include <ligmg/ligmg.hpp>
#include <CombBLAS/CombBLAS.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  {
    auto A = combblas::SpParMat<int, double, combblas::SpDCCols<int, double>>(MPI_COMM_WORLD);
    // load matrix from file
    A.ParallelReadMM("my_matrix.mtx", false);

    // generate random rhs
    combblas::DenseParVec<int, double> rhs(A.getcommgrid(), A.seq().getnrow(), 0, 0);
    rhs.Apply([](double) { return double(rand()) / RAND_MAX; });
    ligmg::orthagonalize(rhs);

    // construct solver
    ligmg::Solver<int, double, combblas::SpDCCols<int, double>> solver(A, ligmg::default_options());

    // solve with rhs
    auto result = solver.solve(rhs);

    std::cout << result << std::endl;
  }

  MPI_Finalize();
}
