cmake_minimum_required(VERSION 3.9)

project(LIGMG VERSION 0.1 LANGUAGES C CXX)

# put executables in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
if(ENV{CRAYPE_LINK_TYPE} STREQUAL "dynamic")
    SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE) # for Cray
endif()

# MPI
find_package(MPI REQUIRED)

# mxx
find_package(mxx REQUIRED)

# combblas
find_package(CombBLAS REQUIRED)
get_target_property(COMBBLAS_LOCATION combblas::CombBLAS INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Found CombBLAS: ${COMBBLAS_LOCATION}")

# petsc
find_package(PETSc REQUIRED)

add_library(LIGMG INTERFACE)
target_include_directories(LIGMG INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(LIGMG INTERFACE MPI::MPI_CXX combblas::CombBLAS PkgConfig::PETSC mxx::mxx)
target_compile_features(LIGMG INTERFACE cxx_std_11)


# Installation
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)

install(TARGETS LIGMG
        EXPORT LIGMGTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

export(EXPORT LIGMGTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/LIGMG/LIGMGTargets.cmake"
  NAMESPACE ligmg::
)
set(ConfigPackageLocation lib/cmake/LIGMG)
configure_package_config_file(cmake/LIGMGConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/LIGMG/LIGMGConfig.cmake"
  INSTALL_DESTINATION "${ConfigPackageLocation}"
)

install(EXPORT LIGMGTargets
  FILE
    LIGMGTargets.cmake
  NAMESPACE
    ligmg::
  DESTINATION
    ${ConfigPackageLocation}
)
install(
  FILES
    cmake/LIGMGConfig.cmake
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)

# Example executable
add_executable(ligmg src/main.cpp src/OptionParser.cpp)
target_link_libraries(ligmg LIGMG MPI::MPI_CXX)
target_include_directories(ligmg PRIVATE src)
target_compile_features(ligmg PRIVATE cxx_std_11)

option(BUILD_SOLVERS "Build solvers using SLEPc" OFF)
if(BUILD_SOLVERS)
  find_package(SLEPc REQUIRED)

  add_executable(eig-analysis src/eig-analysis.cpp)
  target_include_directories(eig-analysis PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(eig-analysis PkgConfig::PETSC PkgConfig::SLEPC MPI::MPI_CXX)

  add_executable(cond src/cond.cpp)
  target_include_directories(cond PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(cond PkgConfig::PETSC PkgConfig::SLEPC MPI::MPI_CXX)

  add_executable(ligmg_eigensolve src/ligmg_eigensolve.cpp src/OptionParser.cpp)
  target_link_libraries(ligmg_eigensolve PkgConfig::PETSC PkgConfig::SLEPC LIGMG combblas::CombBLAS)
endif()
