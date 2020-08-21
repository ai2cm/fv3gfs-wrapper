with import <nixpkgs> {};
let p=  import lib/external/nix;
pythonl = python3.withPackages (ps: [
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
    # on darwwin the stack size maxes out at 65533
    # https://stackoverflow.com/questions/13245019/how-to-change-the-stack-size-using-ulimit-or-per-process-on-mac-os-x-for-a-c-or
    ulimit -s 65532
    
    export CC=gcc
    cp lib/external/nix/fv3/configure.fv3 lib/external/FV3/conf
    [[ -f .env/bin/activate ]] || python -m venv .env
    source .env/bin/activate
  '';
}
