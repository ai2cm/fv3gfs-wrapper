{
  lib,
  stdenv,
  fv3config,
  buildPythonPackage, fv3, cython, jinja2, pkgconfig, numpy, pkg-config, gfortran
  , fms
  , mpich
  , esmf
  , nceplibs
  , netcdf
  , netcdffortran
  , lapack
  , blas
  , mpi4py
  , pyyaml
  , fv3gfs-util
  , netcdf4
  , gcsfs
  , llvmPackages
}:
buildPythonPackage rec {
      version = "0.1.0";
      pname = "fv3gfs-wrapper";

      src = ./.;
      checkInputs = [];

      # environmental variables needed for the wrapper
      PKG_CONFIG_PATH="${fv3}/lib/pkgconfig";
      MPI = "mpich";

      buildInputs = [
          fms
          esmf
          nceplibs
          netcdf
          netcdffortran
          lapack
          blas
          # this is key: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
          gfortran.cc.lib
          cython
          fv3
          mpich
      ] ++ lib.optional stdenv.isDarwin llvmPackages.openmp;

      nativeBuildInputs =  [
          pkg-config
          pkgconfig
          jinja2
          mpich
          gcsfs
          fv3config
          gfortran
          gfortran.cc
      ];


      preBuild = ''
      echo "RUNNING"
      export CC="${gfortran}/bin/gcc";
      export FC="${gfortran}/bin/gfortran"
      make -C lib
      '';

      propagatedBuildInputs = [
          mpi4py
          numpy
          pyyaml
          netcdf4
          fv3gfs-util
      ];
  }
