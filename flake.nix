{
  description = "My flake with dream2nix packages";

  inputs = {
    dream2nix.url = "github:nix-community/dream2nix";
    #dream2nix.url = "path:/home/satwik/projects/dream2nix";
    nixpkgs.follows = "dream2nix/nixpkgs";#.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = inputs @ {
    self,
    dream2nix,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
    	inherit system;
	config.allowUnfree = true;

    };
    packages = dream2nix.lib.evalModules {
      packageSets.nixpkgs = pkgs;
      modules = [
        ./default.nix
        {
          paths.projectRoot = ./.;
          paths.projectRootFile = "flake.nix";
          paths.package = ./.;
        }
      ];
    };
  in {
    # All packages defined in ./packages/<name> are automatically added to the flake outputs
    # e.g., 'packages/hello/default.nix' becomes '.#packages.hello'
    packages.${system}.default = packages;
    devShells.${system}.default = packages.devShell;

    nixConfig = {
    	extra-substituters = [
		"https://cache.nixos.org/"
		"https://nix-community.cachix.org"
		"https://cuda-maintainers.org"
	];
	extra-trusted-public-keys = [
		"cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
		"nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
		"cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
	];
    };
  };
}
