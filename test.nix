let
  p = import ./shell.nix;
in
  with p.pkgs;
  mkShell {
    buildInputs = [ p.wrapper mpich python3 p.python.pkgs.fv3config p.python.pkgs.gcsfs ];
  }
