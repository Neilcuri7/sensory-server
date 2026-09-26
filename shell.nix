{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "aoi-sensory-server-env";
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    python312Packages.fastapi
    python312Packages.uvicorn
    python312Packages.pillow
    python312Packages.pydantic
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.soundfile
    python312Packages.python-multipart
    onnxruntime
    libsndfile
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    pip install --quiet -r requirements.txt
    echo "🌸 Aoi Sensory Hub listo en puerto 8888."
  '';
}
