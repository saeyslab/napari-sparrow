{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      envname = "napari-sparrow";
      pkgs = nixpkgs.legacyPackages.${system};
      fhs = pkgs.buildFHSEnv {
        name = "fhs";
        targetPkgs = pkgs: [
          pkgs.micromamba
          pkgs.libz
          pkgs.expat
        ];
        runScript = "zsh";
        profile = ''
          export name=fhs

          export MAMBA_ROOT_PREFIX=./.mamba

          eval "$(micromamba shell hook -s zsh)"

          micromamba create -f environment.yml -n ${envname}

          micromamba activate ${envname}

          pip install -e .[plugin,testing,tiling,cli,docs]

          export name=fhs
        '';
      };
    in
      {
        devShells.${system}.default = fhs.env;
      };
}
