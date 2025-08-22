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
          pkgs.uv

          pkgs.libz
          pkgs.expat
        ];
        runScript = "zsh";
        profile = ''
          uv venv --python 3.10
          uv pip install -e .[docs]
          source .venv/bin/activate

          export name=fhs
        '';
      };
    in
      {
        devShells.${system}.default = fhs.env;
      };
}
