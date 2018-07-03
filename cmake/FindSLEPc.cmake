find_package(PkgConfig REQUIRED)
if(DEFINED ENV{SLEPC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig")
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig")
endif()
pkg_check_modules(SLEPC IMPORTED_TARGET SLEPc)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (SLEPc
  FAIL_MESSAGE "SLEPc could not be found. Please set SLEPC_DIR in your environment."
  REQUIRED_VARS SLEPC_FOUND)
