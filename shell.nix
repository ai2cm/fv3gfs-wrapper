let
  pkgs = import lib/external/default.nix;
  packageOverrides = self: super: {

    dacite = self.buildPythonPackage rec {
          pname = "dacite";
          version = "1.6.0";
          src = super.fetchPypi {
            inherit pname version;
            sha256 = "sha256-1IEl7QoDUtPen0k7+YADgIj0Xz+ddJjwkLUKhH2qpt8";
          };
          # doesn't find pytest, not sure why, disabling tests for now.
          doCheck = false;
    };

    gcsfs = self.buildPythonPackage rec {
          pname = "gcsfs";
          version = "0.7.1";
          src = super.fetchPypi {
            inherit pname version;
            sha256 = "sha256-A2WN+/GnNNmHqrNjHgo0Kz1+JKJJmLTY0kkf3SEFNyA=";
          };
          propagatedBuildInputs = with self; [
            crcmod
            google-auth
            google-auth-oauthlib
            requests
            decorator
            fsspec
            aiohttp
            ujson
          ];

          # doesn't find pytest, not sure why, disabling tests for now.
          doCheck = false;
    };

    f90nml = self.buildPythonPackage rec {
          pname = "f90nml";
          version = "1.2";
          src = super.fetchPypi {
            inherit pname version;
            sha256 = "sha256-B/u5EB8hjOiczDQmTsgRFu3J7Qq2mtHNgxbxnqaUzS4";
          };
          doCheck = false;

    };


    fv3config = self.buildPythonPackage rec {
          pname = "fv3config";
          version = "0.7.1";
          src = super.fetchPypi {
            inherit pname version;
            sha256 = "sha256-ijkWwsmgNLGX4QNuqhpN93uhCmeQqD5uN1jArHT/52E";
          };
          propagatedBuildInputs = with self; [
            f90nml
            appdirs
            requests
            pyyaml
            gcsfs
            backoff
            dacite
            zarr
            xarray
            cftime
            numpy
            fsspec
            typing-extensions
          ];
          # doesn't find pytest, not sure why, disabling tests for now.
          doCheck = false;

    };

    fv3gfs-util = self.buildPythonPackage rec {
          pname = "fv3gfs-util";
          version = "0.6.0";
          src = pkgs.fetchFromGitHub {
            owner = "VulcanClimateModeling";
            rev = "v${version}";
            repo = pname;
            sha256 = "sha256-iWc7ti6gpWvdYvw6m4t50KukBDQXnsW+avH4CyYagGA=";
          };
          propagatedBuildInputs = with self;[
            zarr
            xarray
            cftime
            numpy
            fsspec
            typing-extensions
          ];
          # doesn't find pytest, not sure why, disabling tests for now.
          doCheck = false;

    };

    mpi4py = (super.mpi4py.override { mpi = pkgs.mpich; }).overridePythonAttrs { doCheck = false; } ;

    wrapper = self.callPackage ./default.nix { };
  };
  python3 = pkgs.python3.override { inherit packageOverrides; };
in
  python3.pkgs.wrapper

