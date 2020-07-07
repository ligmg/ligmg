find_package(PkgConfig REQUIRED)
if(DEFINED PETSC_DIR)
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${PETSC_DIR}/lib/pkgconfig" )
  if(DEFINED PETSC_ARCH)
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig" )
  endif()
endif()
if(DEFINED ENV{PETSC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig" )
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{PETSC_DIR}/lib/pkgconfig" )
  set(PETSC_DIR $ENV{PETSC_DIR})
  set(PETSC_ARCH $ENV{PETSC_ARCH})
endif()
pkg_check_modules(PETSC IMPORTED_TARGET PETSc)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (PETSc
  FAIL_MESSAGE "PETSc could not be found. Please set PETSC_ARCH and PETSC_DIR in your environment."
  REQUIRED_VARS PETSC_FOUND)

