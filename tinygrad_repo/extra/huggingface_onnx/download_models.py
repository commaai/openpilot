import yaml, argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_models(yaml_file: str, download_dir: str) -> None:
  with open(yaml_file, 'r') as f: metadata = yaml.safe_load(f)
  n = len(metadata["repositories"])

  for i, (model_id, model_data) in enumerate(metadata["repositories"].items()):
    print(f"Downloading {i+1}/{n}: {model_id}...")
    allow_patterns = [file_info["file"] for file_info in model_data["files"]]
    root_path = Path(snapshot_download(repo_id=model_id, allow_patterns=allow_patterns, cache_dir=download_dir))
    # download configs too (the sizes are small)
    snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=download_dir)
    print(f"Downloaded model files to: {root_path}")
    model_data["download_path"] = str(root_path)

  # Save the updated metadata back to the YAML file
  with open(yaml_file, 'w') as f: yaml.dump(metadata, f, sort_keys=False)
  print("Download completed according to YAML file.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download models from Huggingface Hub based on a YAML configuration file.")
  parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
  args = parser.parse_args()

  models_folder = Path(__file__).parent / "models"
  models_folder.mkdir(parents=True, exist_ok=True)
  download_models(args.input, str(models_folder))