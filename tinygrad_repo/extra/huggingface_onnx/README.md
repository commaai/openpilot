# HuggingFace ONNX

Tool for discovering, downloading, and validating ONNX models from HuggingFace.

## Extra Dependencies

```bash
pip install huggingface_hub pyyaml requests onnx onnxruntime numpy
```

## Huggingface Manager (discovering and downloading)

The `huggingface_manager.py` script discovers top ONNX models from HuggingFace, collects metadata, and optionally downloads them.

```bash
# Download top 50 models sorted by downloads
python huggingface_manager.py --limit 50 --download

# Just collect metadata (no download)
python huggingface_manager.py --limit 100

# Sort by likes instead of downloads
python huggingface_manager.py --limit 20 --sort likes --download

# Custom output file
python huggingface_manager.py --limit 10 --output my_models.yaml
```

### Output Format

The tool generates a YAML file with the following structure:

```yaml
repositories:
  "model-name":
    url: "https://huggingface.co/model-name"
    download_path: "/path/to/models/..."  # when --download used
    files:
      - file: "model.onnx"
        size: "90.91MB"
total_size: "2.45GB"
created_at: "2024-01-15T10:30:00Z"
```

## Run Models (validation)

The `run_models.py` script validates ONNX models against ONNX Runtime for correctness.

```bash
# Validate models from a YAML configuration file
python run_models.py --validate huggingface_repos.yaml

# Debug specific repository (downloads and validates all ONNX models)
python run_models.py --debug sentence-transformers/all-MiniLM-L6-v2

# Debug specific model file
python run_models.py --debug sentence-transformers/all-MiniLM-L6-v2/onnx/model.onnx

# Debug with model truncation for debugging and validating intermediate results
DEBUGONNX=1 python run_models.py --debug sentence-transformers/all-MiniLM-L6-v2/onnx/model.onnx --truncate 10
```
