set(CALLERS_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
get_filename_component(CURRENT_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(CMAKE_MODULE_PATH "${CURRENT_CONFIG_DIR};${CMAKE_MODULE_PATH}")

# CMake doesn't export interface directories of a target. This is a hack to
# make these dependencies available to downstream users.
set(mxx_DIR @mxx_DIR@)
set(CombBLAS_DIR @CombBLAS_DIR@)
set(PETSC_DIR @PETSC_DIR@)
set(PETSC_ARCH @PETSC_ARCH@)

find_package(MPI REQUIRED QUIET)
find_package(mxx REQUIRED QUIET)
find_package(CombBLAS REQUIRED QUIET)
find_package(PETSc REQUIRED QUIET)

set(CMAKE_MODULE_PATH ${CALLERS_CMAKE_MODULE_PATH})

include("${CMAKE_CURRENT_LIST_DIR}/LIGMGTargets.cmake")
