{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "aoi-sensory-server-env";
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    onnxruntime
    libsndfile
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.libsndfile
      pkgs.onnxruntime
      pkgs.zlib
    ]}:$LD_LIBRARY_PATH"

    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    pip install --quiet -r requirements.txt
    echo "🌸 Aoi Sensory Hub listo en puerto 8888."
  '';
}
