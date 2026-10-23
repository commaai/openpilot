# pip install gdown
# Downloads the 2020 wikipedia dataset used for MLPerf BERT training
import os, hashlib
from pathlib import Path
import tarfile
import gdown
from tqdm import tqdm
from tinygrad.helpers import getenv

def gdrive_download(url:str, path:str):
  if not os.path.exists(path): gdown.download(url, path)

def wikipedia_uncompress_and_extract(file:str, path:str, small:bool=False):
  if not os.path.exists(os.path.join(path, "results4")):
    print("Uncompressing and extracting file...")
    with tarfile.open(file, 'r:gz') as tar:
      tar.extractall(path=path)
      os.remove(file)
      if small:
        for member in tar.getmembers(): tar.extract(path=path, member=member)
      else:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())): tar.extract(path=path, member=member)

def verify_checksum(folder_path:str, checksum_path:str):
  print("Verifying checksums...")
  with open(checksum_path, 'r') as f:
    for line in f:
      expected_checksum, folder_name = line.split()
      file_path = os.path.join(folder_path, folder_name[2:]) # remove './' from the start of the folder name
      hasher = hashlib.md5()
      with open(file_path, 'rb') as f:
        for buf in iter(lambda: f.read(4096), b''): hasher.update(buf)
      if hasher.hexdigest() != expected_checksum:
        raise ValueError(f"Checksum does not match for file: {file_path}")
  print("All checksums match.")

def download_wikipedia(path:str):
  # Links from: https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/dataset.md
  os.makedirs(path, exist_ok=True)
  gdrive_download("https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW", os.path.join(path, "bert_config.json"))
  gdrive_download("https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1", os.path.join(path, "vocab.txt"))
  gdrive_download("https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_", os.path.join(path, "model.ckpt-28252.data-00000-of-00001"))
  gdrive_download("https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0", os.path.join(path, "model.ckpt-28252.index"))
  gdrive_download("https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv", os.path.join(path, "model.ckpt-28252.meta"))
  with open(os.path.join(path, "checkpoint"), "w") as f: f.write('model_checkpoint_path: "model.ckpt-28252"\nall_model_checkpoint_paths: "model.ckpt-28252"')
  if getenv("WIKI_TRAIN", 0):
    gdrive_download("https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_", os.path.join(path, "bert_reference_results_text_md5.txt"))
    gdrive_download("https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_", os.path.join(path, "results_text.tar.gz"))
    wikipedia_uncompress_and_extract(os.path.join(path, "results_text.tar.gz"), path)
    if getenv("VERIFY_CHECKSUM", 0):
      verify_checksum(os.path.join(path, "results4"), os.path.join(path, "bert_reference_results_text_md5.txt"))

if __name__ == "__main__":
  download_wikipedia(getenv("BASEDIR", os.path.join(Path(__file__).parent / "wiki")))