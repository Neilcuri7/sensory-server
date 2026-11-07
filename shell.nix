{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "sensory-server-env";
  # Note: For CUDA GPU acceleration on 6GB VRAM GPUs (RTX 4050/3060),
  # ONNX Runtime automatically uses CUDAExecutionProvider when available with
  # arena_extend_strategy=kSameAsRequested to avoid VRAM overcommit/OOM.
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    onnxruntime
    libsndfile
    stdenv.cc.cc.lib
    libGL
    libGLU
    glib
    libxcb
    libx11
    libxext
    libxrender
    zlib
    espeak-ng
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.libsndfile
      pkgs.onnxruntime
      pkgs.zlib
      pkgs.libGL
      pkgs.libGLU
      pkgs.glib
      pkgs.libxcb
      pkgs.libx11
      pkgs.libxext
      pkgs.libxrender
      pkgs.espeak-ng
    ]}:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

    export ESPEAK_DATA_PATH="${pkgs.espeak-ng}/share/espeak-ng-data"

    if [ ! -d ".venv" ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install --quiet -r requirements.txt
    else
      source .venv/bin/activate
    fi
    echo "Sensory Server listo en puerto 8888."
  '';
}
