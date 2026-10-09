from pathlib import Path

from examples.mlperf.dataloader import get_llama3_dataset
from tinygrad.helpers import getenv

BASEDIR = Path(getenv("BASEDIR", "/raid/datasets/c4-8b/"))
SAMPLES = getenv("SAMPLES", 1_200_000 * 32)
EVAL_SAMPLES = getenv("EVAL_SAMPLES", 1024)
SEQLEN = getenv("SEQLEN", 8192)
DATA_SEED = getenv("DATA_SEED", 5760)

get_llama3_dataset(SAMPLES, SEQLEN, BASEDIR, seed=DATA_SEED, val=False, small=True)
get_llama3_dataset(EVAL_SAMPLES, SEQLEN, BASEDIR, seed=0, val=True, small=True)
