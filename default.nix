{
  config,
  dream2nix,
  lib,
  ...
}:
let
  pyproject = lib.importTOML (config.mkDerivation.src + /pyproject.toml);
in {
  imports = [
    dream2nix.modules.dream2nix.WIP-python-pdm
  ];
  inherit (pyproject.project) name version;
  # select python 3.11
  deps = {nixpkgs, ...}: {
    python = lib.mkForce nixpkgs.python311;
    python3 = lib.mkForce nixpkgs.python311;
    qt = nixpkgs.qt6;
    nodejs = nixpkgs.nodejs;
  };
  overrides = {
  	pyaudio = {config, lib, ...}: {
		deps = {nixpkgs, ...}: {
			inherit (nixpkgs) portaudio;
		};
		mkDerivation = {
			buildInputs = [
				config.deps.portaudio
			];
		};
	};
	timezonefinder = {config, lib, ...}: {
		mkDerivation.propagatedBuildInputs = [
			config.deps.python.pkgs.poetry-core
		];
	};
	casadi = {
	    env.autoPatchelfIgnoreMissingDeps = [
		"libhsl.so"
		"libworhp.so"
		"libsnopt7.so"
		"libknitro.so"
	    ];

	};
	onnxruntime-gpu = {config, lib, ...}: {
		deps = {nixpkgs, nixpkgs_cuda, ...}: {
			#inherit (nixpkgs_cuda.cudaPackages) cudatoolkit cuda_cccl libcublas libcurand libcusparse libcufft cudnn cuda_cudart;
			inherit (nixpkgs) oneDNN re2 onnxruntime;
		};
		mkDerivation = {
			src = config.deps.onnxruntime.dist;
			 unpackPhase = ''
			 	cp -r $src dist
				chmod +w dist
			'';
			 buildInputs= with config.deps; [
			 	onnxruntime.protobuf
			 	re2
				oneDNN
			];
		};

	};
  };
  pdm.lockfile = ./pdm.lock;
  pdm.pyproject = ./pyproject.toml;
  mkDerivation = {
    src = ./.;
    propagatedBuildInputs = [
      config.deps.python.pkgs.poetry-core

    ];
    doInstallCheck = false;
    doCheck = false;
  };



}
