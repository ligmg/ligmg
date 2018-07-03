find_package(MPI REQUIRED QUIET)
find_package(mxx REQUIRED QUIET)
find_package(CombBLAS REQUIRED QUIET)
find_package(PETSc REQUIRED QUIET)

include("${CMAKE_CURRENT_LIST_DIR}/LIGMGTargets.cmake}")
