import yaml, time, requests, argparse
from pathlib import Path
from huggingface_hub import list_models, HfApi
from tinygrad.helpers import tqdm

HUGGINGFACE_URL = "https://huggingface.co"
SKIPPED_FILES = [
  "fp16", "int8", "uint8", "quantized",      # numerical accuracy issues
  "avx2", "arm64", "avx512", "avx512_vnni",  # numerical accuracy issues
  "q4", "q4f16", "bnb4",                     # unimplemented quantization
  "model_O4",                                # requires non cpu ort runner and MemcpyFromHost op
  "merged",                                  # TODO implement attribute with graph type and Loop op
]
SKIPPED_REPO_PATHS = [
  # Invalid model-index
  "AdamCodd/vit-base-nsfw-detector",
  # TODO: implement attribute with graph type and Loop op
  "minishlab/potion-base-8M", "minishlab/M2V_base_output", "minishlab/potion-retrieval-32M",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, GroupQueryAttention
  "HuggingFaceTB/SmolLM2-360M-Instruct",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  # TODO: implmement RandomNormalLike
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", 'SimianLuo/LCM_Dreamshaper_v7',
  # TODO: implement NonZero
  "mangoapps/fb_zeroshot_mnli_onnx",
  # TODO huge Concat in here with 1024 (1, 3, 32, 32) Tensors, and maybe a MOD bug with const folding
  "briaai/RMBG-2.0",
]

def get_top_repos(n: int, sort: str) -> list[str]: # list["FacebookAI/xlm-roberta-large", ...]
  print(f"** Getting top {n} models sorted by {sort} **")
  repos = []
  i = 0
  for model in list_models(filter="onnx", sort=sort):
    if model.id in SKIPPED_REPO_PATHS: continue
    print(f"{i+1}/{n}: {model.id} ({getattr(model, sort)})")
    repos.append(model.id)
    i += 1
    if i == n: break
  return repos

def get_metadata(repos:list[str]) -> dict:
  api = HfApi()
  repos_metadata = {"repositories": {}}
  total_size = 0

  # TODO: speed head requests up with async?
  for repo in tqdm(repos, desc="Getting metadata"):
    files_metadata = []
    model_info = api.model_info(repo)

    for file in model_info.siblings:
      filename = file.rfilename
      if not (filename.endswith('.onnx') or filename.endswith('.onnx_data')): continue
      if any(skip_str in filename for skip_str in SKIPPED_FILES): continue
      head = requests.head(f"{HUGGINGFACE_URL}/{repo}/resolve/main/{filename}", allow_redirects=True)
      file_size = file.size or int(head.headers.get('Content-Length', 0))
      files_metadata.append({"file": filename, "size": f"{file_size/1e6:.2f}MB"})
      total_size += file_size

    repos_metadata["repositories"][repo] = {
      "url": f"{HUGGINGFACE_URL}/{repo}",
      "download_path": None,
      "files": files_metadata,
    }
  repos_metadata['total_size'] = f"{total_size/1e9:.2f}GB"
  repos_metadata['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
  return repos_metadata

if __name__ == "__main__":
  sort = "downloads" # recent 30 days downloads
  huggingface_onnx_dir = Path(__file__).parent

  parser = argparse.ArgumentParser(description="Produces a YAML file with metadata of top huggingface onnx models")
  parser.add_argument("--limit", type=int, required=True, help="Number of top repositories to process (e.g., 100)")
  parser.add_argument("--output", type=str, default="huggingface_repos.yaml", help="Output YAML file name to save the report")
  args = parser.parse_args()

  top_repos = get_top_repos(args.limit, sort)
  metadata = get_metadata(top_repos)
  yaml_path = huggingface_onnx_dir / args.output
  with open(yaml_path, 'w') as f:
    yaml.dump(metadata, f, sort_keys=False)
    print(f"YAML saved to: {str(yaml_path)}")
