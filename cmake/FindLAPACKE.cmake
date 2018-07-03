find_package(PkgConfig REQUIRED)
set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/opt/lapack/lib/pkgconfig")
if(DEFINED ENV{CRAY_LIBSCI_PREFIX_DIR})
  set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib/pkgconfig")
endif()
pkg_search_module(LAPACKE IMPORTED_TARGET lapacke openblas libsci)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (LAPACKE
  "LAPACKE could not be found."
  LAPACKE_FOUND)