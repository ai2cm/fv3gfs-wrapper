with import <nixpkgs> {};
let p=  import lib/external/nix;
pythonl = python3.withPackages (ps: [
        ps.numcodecs
        ps.pip
  ]);
in
mkShell {
  name = "fv3";
  buildInputs = [
      p.fms
      p.esmf
      p.nceplibs
      p.fv3
      netcdf
      netcdffortran
      openssh # hidden dependency of openmpi
      lapack
      blas
      openmpi
      perl
      gfortran
      gfortran.cc
      getopt
      vim
      pkg-config
      pythonl
  ];

  SHELL = "${bash}/bin/bash";
  FMS_DIR="${p.fms}/include";
  ESMF_DIR="${p.esmf}";
  INCLUDE="-I${p.fms}/include -I${netcdffortran}/include -I${p.esmf}/include/";
  NCEPLIBS_DIR="${p.nceplibs}/lib";
  OMPI_CC="${gfortran.cc}/bin/gcc";

  shellHook = ''
    cp lib/external/nix/fv3/configure.fv3 lib/external/FV3/conf
    source .env/bin/activate
  '';
}
