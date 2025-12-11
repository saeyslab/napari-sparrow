{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      envname = "napari-sparrow";
      pkgs = nixpkgs.legacyPackages.${system};

      uv-fhs = pkgs.buildFHSEnv {
        name = "uv-fhs";
        targetPkgs = pkgs: [
          pkgs.uv
          pkgs.libz
          pkgs.expat
          pkgs.libGL
          pkgs.glibc
          pkgs.glib
        ];
        runscript = "zsh";
        profile = ''
          uv venv --python 3.10
          uv pip install -e .[docs,testing]
          uv pip install jupyter
          source .venv/bin/activate
          export name=uv-fhs
        '';
      };

      mm-fhs = pkgs.buildFHSEnv {
        name = "mm-fhs";
        targetPkgs = pkgs: [
          pkgs.micromamba
          pkgs.expat
        ];
        runScript = "zsh";
        profile = ''
          export MAMBA_ROOT_PREFIX=./.mamba
          eval "$(micromamba shell hook -s zsh)"
          # micromamba create -f environment.yml -n ${envname}
          micromamba activate ${envname}
          pip install -e .[docs,testing]
          export name=mm-fhs
        '';
      };
    in
      {
        devShells.${system} = {
          default = mm-fhs.env;
          uv = uv-fhs.env;
        };
      };
}
