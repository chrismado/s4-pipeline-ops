"""Download GGUF model files for llama.cpp benchmarks.

Usage:
    python scripts/download_models.py

Downloads Mistral 7B Instruct v0.3 GGUF files from HuggingFace
into the models/ directory. Requires `huggingface-hub` to be installed.
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = [
    {
        "repo": "MistralAI/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "description": "Mistral 7B Q4_K_M (~4.4 GB)",
    },
    {
        "repo": "MistralAI/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "mistral-7b-instruct-v0.3.Q8_0.gguf",
        "description": "Mistral 7B Q8_0 (~7.7 GB)",
    },
]


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface-hub first: pip install huggingface-hub")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading models to {MODELS_DIR}\n")

    for model in MODELS:
        dest = MODELS_DIR / model["filename"]
        if dest.exists():
            print(f"  [skip] {model['filename']} already exists")
            continue

        print(f"  Downloading {model['description']}...")
        try:
            path = hf_hub_download(
                repo_id=model["repo"],
                filename=model["filename"],
                local_dir=str(MODELS_DIR),
            )
            print(f"  [done] {path}")
        except Exception as e:
            print(f"  [error] {model['filename']}: {e}")
            print("  You can download manually from:")
            print(f"    https://huggingface.co/{model['repo']}")

    print("\nDone. Run benchmarks with: s4ops benchmark --backend llamacpp")


if __name__ == "__main__":
    main()
