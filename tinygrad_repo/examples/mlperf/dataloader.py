import os, random, pickle, queue, struct, math, functools, hashlib, time
from typing import List
from pathlib import Path
from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count

import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, Context, round_up, tqdm, OSX
from tinygrad.nn.state import TensorIO

### ResNet

class MyQueue:
  def __init__(self, multiple_readers=True, multiple_writers=True):
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = Lock() if multiple_readers else None
    self._wlock = Lock() if multiple_writers else None
  def get(self):
    if self._rlock: self._rlock.acquire()
    ret = pickle.loads(self._reader.recv_bytes())
    if self._rlock: self._rlock.release()
    return ret
  def put(self, obj):
    if self._wlock: self._wlock.acquire()
    self._writer.send_bytes(pickle.dumps(obj))
    if self._wlock: self._wlock.release()

def shuffled_indices(n, seed=None):
  rng = random.Random(seed)
  indices = {}
  for i in range(n-1, -1, -1):
    j = rng.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X:Tensor, seed):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.imagenet import center_crop, preprocess_train
  from PIL import Image

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn, val = _recv
      if fn is not None:
        img = Image.open(fn)
        img = img.convert('RGB') if img.mode != "RGB" else img

        if val:
          # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
          # sudo apt-get install libjpeg-dev
          # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
          img = center_crop(img)
          img = np.array(img)
        else:
          # reseed rng for determinism
          if seed is not None:
            np.random.seed(seed * 2 ** 10 + idx)
            random.seed(seed * 2 ** 10 + idx)
          img = preprocess_train(img)
      else:
        # pad data with training mean
        img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (224, 224, 1))

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.base.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)

def batch_load_resnet(batch_size=64, val=False, shuffle=True, seed=None, pad_first_batch=False):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()

  if pad_first_batch:
    FIRST_BATCH_PAD = round_up(len(files), batch_size) - len(files)
  else:
    FIRST_BATCH_PAD = 0
  file_count = FIRST_BATCH_PAD + len(files)
  BATCH_COUNT = min(32, file_count // batch_size)

  def _gen():
    for _ in range(FIRST_BATCH_PAD): yield -1
    yield from shuffled_indices(len(files), seed=seed) if shuffle else iter(range(len(files)))
  gen = iter(_gen())

  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fidx = next(gen)
      if fidx != -1:
        fn = files[fidx]
        q_in.put((idx, fn, val))
        Y[idx] = cir[fn.split("/")[-2]]
      else:
        # padding
        q_in.put((idx, None, val))
        Y[idx] = -1

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], Cookie(num)

  #q_in, q_out = MyQueue(multiple_writers=False), MyQueue(multiple_readers=False)
  q_in, q_out = Queue(), Queue()

  sz = (batch_size*BATCH_COUNT, 224, 224, 3)
  shm_name = "resnet_X_val" if val else "resnet_X_train"
  if not OSX and os.path.exists(f"/dev/shm/{shm_name}"): os.unlink(f"/dev/shm/{shm_name}")
  shm = shared_memory.SharedMemory(name=shm_name, create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    if OSX: X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    else: X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/{shm_name}")
    Y = [None] * (batch_size*BATCH_COUNT)

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, seed))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored unless pad_first_batch is True
    for _ in range(0, file_count//batch_size): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    try:
      shm.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

### BERT

def process_batch_bert(data: List[dict]) -> dict[str, Tensor]:
  return {
    "input_ids": Tensor(np.concatenate([s["input_ids"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
    "input_mask": Tensor(np.concatenate([s["input_mask"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
    "segment_ids": Tensor(np.concatenate([s["segment_ids"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
    "masked_lm_positions": Tensor(np.concatenate([s["masked_lm_positions"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
    "masked_lm_ids": Tensor(np.concatenate([s["masked_lm_ids"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
    "masked_lm_weights": Tensor(np.concatenate([s["masked_lm_weights"] for s in data], axis=0), dtype=dtypes.float32, device="CPU"),
    "next_sentence_labels": Tensor(np.concatenate([s["next_sentence_labels"] for s in data], axis=0), dtype=dtypes.int32, device="CPU"),
  }

def load_file(file: str):
  with open(file, "rb") as f:
    return pickle.load(f)

class InterleavedDataset:
  def __init__(self, files:List[str], cycle_length:int):
    self.dataset = files
    self.cycle_length = cycle_length
    self.queues = [queue.Queue() for _ in range(self.cycle_length)]
    for i in range(len(self.queues)): self.queues[i].queue.extend(load_file(self.dataset.pop(0)))
    self.queue_pointer = len(self.queues) - 1

  def get(self):
    # Round-robin across queues
    try:
      self.advance()
      return self.queues[self.queue_pointer].get_nowait()
    except queue.Empty:
      self.fill(self.queue_pointer)
      return self.get()

  def advance(self):
    self.queue_pointer = (self.queue_pointer + 1) % self.cycle_length

  def fill(self, queue_index: int):
    try:
      file = self.dataset.pop(0)
    except IndexError:
      return
    self.queues[queue_index].queue.extend(load_file(file))

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 394
def batch_load_train_bert(BS:int):
  from extra.datasets.wikipedia import get_wiki_train_files
  fs = sorted(get_wiki_train_files())
  train_files = []
  while fs: # TF shuffle
    random.shuffle(fs)
    train_files.append(fs.pop(0))

  cycle_length = min(getenv("NUM_CPU_THREADS", min(os.cpu_count(), 8)), len(train_files))
  assert cycle_length > 0, "cycle_length must be greater than 0"

  dataset = InterleavedDataset(train_files, cycle_length)
  while True:
    yield process_batch_bert([dataset.get() for _ in range(BS)])

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 416
def batch_load_val_bert(BS:int):
  file =  getenv("BASEDIR", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki") / "eval.pkl"
  dataset = load_file(file)
  idx = 0
  while True:
    start_idx = (idx * BS) % len(dataset)
    end_idx = ((idx + 1) * BS) % len(dataset)
    if start_idx < end_idx:
      yield process_batch_bert(dataset[start_idx:end_idx])
    else:  # wrap around the end to the beginning of the dataset
      yield process_batch_bert(dataset[start_idx:] + dataset[:end_idx])
    idx += 1

### UNET3D

def load_unet3d_data(preprocessed_dataset_dir, seed, queue_in, queue_out, X:Tensor, Y:Tensor):
  from extra.datasets.kits19 import rand_balanced_crop, rand_flip, random_brightness_augmentation, gaussian_noise

  while (data := queue_in.get()) is not None:
    idx, fn, val = data
    case_name = os.path.basename(fn).split("_x.npy")[0]
    x, y = np.load(preprocessed_dataset_dir / f"{case_name}_x.npy"), np.load(preprocessed_dataset_dir / f"{case_name}_y.npy")

    if not val:
      if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

      x, y = rand_balanced_crop(x, y)
      x, y = rand_flip(x, y)
      x, y = x.astype(np.float32), y.astype(np.uint8)
      x = random_brightness_augmentation(x)
      x = gaussian_noise(x)

    X[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = x.tobytes()
    Y[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = y.tobytes()

    queue_out.put(idx)
  queue_out.put(None)

def batch_load_unet3d(preprocessed_dataset_dir:Path, batch_size:int=6, val:bool=False, shuffle:bool=True, seed=None):
  assert preprocessed_dataset_dir is not None, "run preprocess_data on kits19"

  files = sorted(list(preprocessed_dataset_dir.glob("*_x.npy")))
  file_indices = list(range(len(files)))
  batch_count = min(32, len(files) // batch_size)

  queue_in, queue_out = Queue(), Queue()
  procs, data_out_count = [], [0] * batch_count
  shm_name_x, shm_name_y = "unet3d_x", "unet3d_y"
  sz = (batch_size * batch_count, 1, 128, 128, 128)
  if os.path.exists(f"/dev/shm/{shm_name_x}"): os.unlink(f"/dev/shm/{shm_name_x}")
  if os.path.exists(f"/dev/shm/{shm_name_y}"): os.unlink(f"/dev/shm/{shm_name_y}")
  shm_x = shared_memory.SharedMemory(name=shm_name_x, create=True, size=prod(sz))
  shm_y = shared_memory.SharedMemory(name=shm_name_y, create=True, size=prod(sz))

  shutdown = False
  class Cookie:
    def __init__(self, bc):
      self.bc = bc
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.bc)
        except StopIteration: pass

  def enqueue_batch(bc):
    for idx in range(bc * batch_size, (bc+1) * batch_size):
      fn = files[next(ds_iter)]
      queue_in.put((idx, fn, val))

  def shuffle_indices(file_indices, seed=None):
    rng = random.Random(seed)
    rng.shuffle(file_indices)

  if shuffle: shuffle_indices(file_indices, seed=seed)
  ds_iter = iter(file_indices)

  try:
    X = Tensor.empty(*sz, dtype=dtypes.float32, device=f"disk:/dev/shm/{shm_name_x}")
    Y = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/{shm_name_y}")

    for _ in range(cpu_count()):
      proc = Process(target=load_unet3d_data, args=(preprocessed_dataset_dir, seed, queue_in, queue_out, X, Y))
      proc.daemon = True
      proc.start()

      procs.append(proc)

    for bc in range(batch_count):
      enqueue_batch(bc)

    for _ in range(len(files) // batch_size):
      while True:
        bc = queue_out.get() // batch_size
        data_out_count[bc] += 1
        if data_out_count[bc] == batch_size: break

      data_out_count[bc] = 0
      yield X[bc * batch_size:(bc + 1) * batch_size], Y[bc * batch_size:(bc + 1) * batch_size], Cookie(bc)
  finally:
    shutdown = True

    for _ in procs: queue_in.put(None)
    queue_in.close()

    for _ in procs:
      while queue_out.get() is not None: pass
    queue_out.close()

    # shutdown processes
    for proc in procs: proc.join()

    shm_x.close()
    shm_y.close()
    try:
      shm_x.unlink()
      shm_y.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

### RetinaNet

def load_retinanet_data(base_dir:Path, val:bool, queue_in:Queue, queue_out:Queue,
                        imgs:Tensor, boxes:Tensor, labels:Tensor, matches:Tensor|None=None,
                        anchors:Tensor|None=None, seed:int|None=None):
  from extra.datasets.openimages import image_load, random_horizontal_flip, resize
  from examples.mlperf.helpers import box_iou, find_matches, generate_anchors
  import torch

  while (data:=queue_in.get()) is not None:
    idx, img, tgt = data
    img = image_load(base_dir, img["subset"], img["file_name"])

    if val:
      img = resize(img)[0]
    else:
      if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

      img, tgt = random_horizontal_flip(img, tgt)
      img, tgt, _ = resize(img, tgt=tgt)
      match_quality_matrix = box_iou(tgt["boxes"], (anchor := np.concatenate(generate_anchors((800, 800)))))
      match_idxs = find_matches(match_quality_matrix, allow_low_quality_matches=True)
      clipped_match_idxs = np.clip(match_idxs, 0, None)
      clipped_boxes, clipped_labels = tgt["boxes"][clipped_match_idxs], tgt["labels"][clipped_match_idxs]

      boxes[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = clipped_boxes.tobytes()
      labels[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = clipped_labels.tobytes()
      matches[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = match_idxs.tobytes()
      anchors[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = anchor.tobytes()

    imgs[idx].contiguous().realize().uop.base.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

    queue_out.put(idx)
  queue_out.put(None)

def batch_load_retinanet(dataset, val:bool, base_dir:Path, batch_size:int=32, shuffle:bool=True, seed:int|None=None):
  def _enqueue_batch(bc):
    from extra.datasets.openimages import prepare_target
    for idx in range(bc * batch_size, (bc+1) * batch_size):
      img = dataset.loadImgs(next(dataset_iter))[0]
      ann = dataset.loadAnns(dataset.getAnnIds(img_id:=img["id"]))
      tgt = prepare_target(ann, img_id, (img["height"], img["width"]))

      if img_ids is not None:
        img_ids[idx] = img_id

      if img_sizes is not None:
        img_sizes[idx] = tgt["image_size"]

      queue_in.put((idx, img, tgt))

  def _setup_shared_mem(shm_name:str, size:tuple[int, ...], dtype:dtypes) -> tuple[shared_memory.SharedMemory, Tensor]:
    if os.path.exists(f"/dev/shm/{shm_name}"): os.unlink(f"/dev/shm/{shm_name}")
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=prod(size))
    shm_tensor = Tensor.empty(*size, dtype=dtype, device=f"disk:/dev/shm/{shm_name}")
    return shm, shm_tensor

  image_ids = sorted(dataset.imgs.keys())
  batch_count = min(32, len(image_ids) // batch_size)

  queue_in, queue_out = Queue(), Queue()
  procs, data_out_count = [], [0] * batch_count

  shm_imgs, imgs = _setup_shared_mem("retinanet_imgs", (batch_size * batch_count, 800, 800, 3), dtypes.uint8)

  if val:
    boxes, labels, matches, anchors = None, None, None, None
    img_ids, img_sizes = [None] * (batch_size * batch_count), [None] * (batch_size * batch_count)
  else:
    img_ids, img_sizes = None, None
    shm_boxes, boxes = _setup_shared_mem("retinanet_boxes", (batch_size * batch_count, 120087, 4), dtypes.float32)
    shm_labels, labels = _setup_shared_mem("retinanet_labels", (batch_size * batch_count, 120087), dtypes.int64)
    shm_matches, matches = _setup_shared_mem("retinanet_matches", (batch_size * batch_count, 120087), dtypes.int64)
    shm_anchors, anchors = _setup_shared_mem("retinanet_anchors", (batch_size * batch_count, 120087, 4), dtypes.float64)

  shutdown = False
  class Cookie:
    def __init__(self, bc):
      self.bc = bc
    def __del__(self):
      if not shutdown:
        try: _enqueue_batch(self.bc)
        except StopIteration: pass

  def shuffle_indices(indices, seed):
    rng = random.Random(seed)
    rng.shuffle(indices)

  if shuffle: shuffle_indices(image_ids, seed=seed)
  dataset_iter = iter(image_ids)

  try:
    for _ in range(cpu_count()):
      proc = Process(
        target=load_retinanet_data,
        args=(base_dir, val, queue_in, queue_out, imgs, boxes, labels),
        kwargs={"matches": matches, "anchors": anchors, "seed": seed}
      )
      proc.daemon = True
      proc.start()
      procs.append(proc)

    for bc in range(batch_count):
      _enqueue_batch(bc)

    for _ in range(len(image_ids) // batch_size):
      while True:
        bc = queue_out.get() // batch_size
        data_out_count[bc] += 1
        if data_out_count[bc] == batch_size: break

      data_out_count[bc] = 0

      if val:
        yield (imgs[bc * batch_size:(bc + 1) * batch_size],
               img_ids[bc * batch_size:(bc + 1) * batch_size],
               img_sizes[bc * batch_size:(bc + 1) * batch_size],
               Cookie(bc))
      else:
        yield (imgs[bc * batch_size:(bc + 1) * batch_size],
               boxes[bc * batch_size:(bc + 1) * batch_size],
               labels[bc * batch_size:(bc + 1) * batch_size],
               matches[bc * batch_size:(bc + 1) * batch_size],
               anchors[bc * batch_size:(bc + 1) * batch_size],
               Cookie(bc))
  finally:
    shutdown = True

    for _ in procs: queue_in.put(None)
    queue_in.close()

    for _ in procs:
      while queue_out.get() is not None: pass
    queue_out.close()

    # shutdown processes
    for proc in procs: proc.join()

    shm_imgs.close()

    if not val:
      shm_boxes.close()
      shm_labels.close()
      shm_matches.close()
      shm_anchors.close()

    try:
      shm_imgs.unlink()

      if not val:
        shm_boxes.unlink()
        shm_labels.unlink()
        shm_matches.unlink()
        shm_anchors.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

# llama3

class BinIdxDataset:
  def __init__(self, base_path:Path):
    self.idx_t = Tensor(base_path.with_name(f"{base_path.name}.idx"))
    self.idx = TensorIO(self.idx_t)

    # parse idx file
    magic = self.idx.read(9)
    assert magic == b"MMIDIDX\x00\x00", "invalid index file format"
    version, = struct.unpack("<Q", self.idx.read(8))
    assert version == 1, "unsupported index version"
    dtype_code, = struct.unpack("<B", self.idx.read(1))
    self.dtype = {1:dtypes.uint8, 2:dtypes.int8, 3:dtypes.int16, 4:dtypes.int32, 5:dtypes.int64, 6:dtypes.float64, 7:dtypes.double, 8:dtypes.uint16}[dtype_code]
    self.count, = struct.unpack("<Q", self.idx.read(8))
    doc_count, = struct.unpack("<Q", self.idx.read(8))

    start = self.idx.tell()
    end = start + self.count * dtypes.int32.itemsize
    self.sizes = self.idx_t[start:end].bitcast(dtypes.int32).numpy()

    start = end
    end = start + self.count * dtypes.int64.itemsize
    self.pointers = self.idx_t[start:end].bitcast(dtypes.int64).numpy()

    start = end
    end = start + doc_count * dtypes.int64.itemsize
    self.doc_idx = self.idx_t[start:end].bitcast(dtypes.int64).numpy()

    # bin file
    self.bin_t = Tensor(base_path.with_name(f"{base_path.name}.bin"))

  def _index(self, idx) -> tuple[int, int]:
    return int(self.pointers[idx]), int(self.sizes[idx])

  def get(self, idx, offset:int=0, length:int|None=None):
    ptr, size = self._index(idx)
    if length is None: length = size - offset
    ptr += offset * self.dtype.itemsize
    return self.bin_t[ptr:ptr+length*self.dtype.itemsize].bitcast(self.dtype).to(None)

# https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/datasets.html
class GPTDataset:
  def __init__(self, base_path:Path, samples:int, seqlen:int, seed:int, shuffle:bool):
    self.samples, self.seqlen = samples, seqlen
    self.shuffle = shuffle
    self.rng = np.random.RandomState(seed)

    self.indexed_dataset = BinIdxDataset(base_path)

    # check for cache
    cache_hash = hashlib.sha256(f"{samples}:{seqlen}:{seed}:{shuffle}".encode()).hexdigest()
    cache_path = base_path.with_name(f"{base_path.name}.{cache_hash}.index_cache")
    print(f"try loading GPTDataset from {cache_path}...")
    if cache_path.exists():
      print("cache found, loading...")
      with open(cache_path, "rb") as f:
        self.doc_idx, self.sample_idx, self.shuffle_idx = pickle.load(f)
    else:
      print("cache not found, building index...")
      self.doc_idx = self._build_doc_idx()
      self.sample_idx = self._build_sample_idx()
      self.shuffle_idx = self._build_shuffle_idx()
      # save cache
      with open(cache_path, "wb") as f:
        pickle.dump((self.doc_idx, self.sample_idx, self.shuffle_idx), f)

  def __getitem__(self, idx):
    if idx is None:
      text = self._get(0)
    else:
      text = self._get(idx)

    return text

  def _get(self, idx):
    idx = self.shuffle_idx[idx]

    doc_idx_beg, doc_idx_beg_offset = self.sample_idx[idx]
    doc_idx_end, doc_idx_end_offset = self.sample_idx[idx + 1]

    doc_ids, sample_parts = [], []

    if doc_idx_beg == doc_idx_end:
      doc_ids.append(self.doc_idx[doc_idx_beg])

      sample_parts.append(
          self.indexed_dataset.get(
            int(self.doc_idx[doc_idx_beg]), offset=int(doc_idx_beg_offset), length=int(doc_idx_end_offset - doc_idx_beg_offset + 1)))
    else:
      for i in range(doc_idx_beg, doc_idx_end + 1):
        doc_ids.append(self.doc_idx[i])

        offset = 0 if i > doc_idx_beg else doc_idx_beg_offset
        length = None if i < doc_idx_end else int(doc_idx_end_offset + 1)
        sample_parts.append(self.indexed_dataset.get(int(self.doc_idx[i]), offset=int(offset), length=length))

    # concat all parts
    text = Tensor.cat(*sample_parts)

    return text

  @functools.cached_property
  def tokens_per_epoch(self) -> int:
    return sum(self.indexed_dataset.sizes.tolist())

  @functools.cached_property
  def num_epochs(self) -> int:
    # we need enough epochs to cover the requested amount of tokens
    num_epochs = 1
    num_tokens = self.tokens_per_epoch
    while num_tokens < self.samples * self.seqlen:
      num_epochs += 1
      num_tokens += self.tokens_per_epoch
    return num_epochs

  # https://github.com/NVIDIA/Megatron-LM/blob/94bd476bd840c2fd4c3ebfc7448c2af220f4832b/megatron/core/datasets/gpt_dataset.py#L558
  def _build_doc_idx(self):
    print(f"building doc_idx for {self.num_epochs=}, {self.indexed_dataset.count=}")
    st = time.perf_counter()
    # doc_idx = np.mgrid[:self.num_epochs, :self.indexed_dataset.count][1]
    doc_idx = np.arange(self.indexed_dataset.count).reshape(1, -1).repeat(self.num_epochs, axis=0).flatten()
    doc_idx = doc_idx.astype(np.int32)
    at = time.perf_counter()
    if self.shuffle: self.rng.shuffle(doc_idx)
    print(f"doc_idx built in {at - st:.3f}s, shuffled in {time.perf_counter() - at:.3f}s")
    return doc_idx

  def _build_sample_idx(self):
    print(f"building sample_idx for {self.samples=}, {self.seqlen=}, {self.doc_idx.shape[0]=}")
    sample_idx_max = max(self.doc_idx.shape[0], self.indexed_dataset.sizes.max())
    sample_idx = np.empty((self.samples + 1, 2), dtype=np.int64 if sample_idx_max > dtypes.int32.max else np.int32)

    sample_idx_idx, doc_idx_idx, doc_offset = 0, 0, 0
    sample_idx[sample_idx_idx, 0], sample_idx[sample_idx_idx, 1] = doc_idx_idx, doc_offset
    sample_idx_idx += 1

    for _ in tqdm(range(1, self.samples + 1)):
      remaining_seqlen = self.seqlen + 1
      while remaining_seqlen > 0:
        doc_idx = int(self.doc_idx[doc_idx_idx])
        doc_len = int(self.indexed_dataset.sizes[doc_idx]) - doc_offset
        remaining_seqlen -= doc_len
        if remaining_seqlen <= 0:
          doc_offset += remaining_seqlen + doc_len - 1
          remaining_seqlen = 0
        else:
          if doc_idx_idx == len(self.doc_idx) - 1:
            assert sample_idx_idx == self.samples
            doc_idx = int(self.doc_idx[doc_idx_idx])
            doc_offset = int(self.indexed_dataset.sizes[doc_idx]) - 1
            break
          doc_idx_idx += 1
          doc_offset = 0

      sample_idx[sample_idx_idx, 0], sample_idx[sample_idx_idx, 1] = doc_idx_idx, doc_offset
      sample_idx_idx += 1

    return sample_idx

  def _build_shuffle_idx(self):
    print(f"building shuffle_idx for {self.samples=}")
    st = time.perf_counter()
    shuffle_idx = np.arange(self.samples, dtype=np.int32)
    at = time.perf_counter()
    if self.shuffle: self.rng.shuffle(shuffle_idx)
    print(f"shuffle_idx built in {at - st:.3f}s, shuffled in {time.perf_counter() - at:.3f}s")
    return shuffle_idx

class BlendedGPTDataset:
  def __init__(self, paths:list[Path], weights:list[float], samples:int, seqlen:int, seed:int, shuffle:bool):
    self.shuffle = shuffle
    self.rng = np.random.RandomState(seed)

    # normalize weights
    total_weight = sum(weights)
    self.weights = [w / total_weight for w in weights]

    self.samples = samples
    surplus = 0.005
    samples_per_blend = [math.ceil(math.ceil(self.samples * w) * (1 + surplus)) for w in self.weights]

    self.datasets = [GPTDataset(path, samples_per_blend[i], seqlen, seed + i, shuffle) for i,path in enumerate(paths)]

    # check for cache
    cache_hash = hashlib.sha256(f"{samples}:{seqlen}:{seed}:{shuffle}".encode()).hexdigest()
    cache_path = paths[0].with_name(f"{paths[0].name}.{cache_hash}.blend_cache")
    print(f"try loading BlendedGPTDataset from {cache_path}...")
    if cache_path.exists():
      print("cache found, loading...")
      with open(cache_path, "rb") as f:
        self.dataset_idx, self.dataset_sample_idx = pickle.load(f)
    else:
      print("cache not found, building index...")
      self.dataset_idx, self.dataset_sample_idx = self._build_blend_idx()
      # save cache
      with open(cache_path, "wb") as f:
        pickle.dump((self.dataset_idx, self.dataset_sample_idx), f)

  def get(self, idx:int):
    tokens = self.datasets[self.dataset_idx[idx]][self.dataset_sample_idx[idx]]
    return tokens

  def _build_blend_idx(self):
    dataset_idx = np.zeros(self.samples, dtype=np.int16)
    dataset_sample_idx = np.zeros(self.samples, dtype=np.int64)

    unspent_datasets = set(range(len(self.datasets)))
    dataset_sample_counts = [0] * len(self.datasets)

    for i in tqdm(range(self.samples)):
      error_argmax, error_max = 0, 0.0
      for di in unspent_datasets:
        error = self.weights[di] * max(i, 1) - dataset_sample_counts[di]
        if error > error_max:
          error_max = error
          error_argmax = di

      dataset_idx[i] = error_argmax
      dataset_sample_idx[i] = dataset_sample_counts[error_argmax]

      dataset_sample_counts[error_argmax] += 1

    return dataset_idx, dataset_sample_idx

def batch_load_llama3(bs:int, samples:int, seqlen:int, base_dir:Path, seed:int=0, val:bool=True):
  if val:
    dataset = BlendedGPTDataset([
      base_dir / "validation" / "c4-validationn-91205-samples.en_text_document",
    ], [
      1.0
    ], samples, seqlen, seed, False)
  else:
    dataset = BlendedGPTDataset([
      base_dir / "c4-train.en_6_text_document",
      base_dir / "c4-train.en_7_text_document",
    ], [
      1.0, 1.0
    ], samples, seqlen, seed, True)

  for b in range(math.ceil(samples / bs)):
    batch = []
    for i in range(bs):
      tokens = dataset.get(b * bs + i)
      batch.append(tokens)
    yield Tensor.stack(batch, dim=0)

if __name__ == "__main__":
  def load_unet3d(val):
    assert not val, "validation set is not supported due to different sizes on inputs"

    from extra.datasets.kits19 import get_train_files, get_val_files, preprocess_dataset, TRAIN_PREPROCESSED_DIR, VAL_PREPROCESSED_DIR
    preprocessed_dir = VAL_PREPROCESSED_DIR if val else TRAIN_PREPROCESSED_DIR
    files = get_val_files() if val else get_train_files()

    if not preprocessed_dir.exists(): preprocess_dataset(files, preprocessed_dir, val)
    with tqdm(total=len(files)) as pbar:
      for x, _, _ in batch_load_unet3d(preprocessed_dir, val=val):
        pbar.update(x.shape[0])

  def load_resnet(val):
    from extra.datasets.imagenet import get_train_files, get_val_files
    files = get_val_files() if val else get_train_files()
    with tqdm(total=len(files)) as pbar:
      for x,y,c in batch_load_resnet(val=val):
        pbar.update(x.shape[0])

  def load_retinanet(val):
    from extra.datasets.openimages import BASEDIR, download_dataset
    from pycocotools.coco import COCO
    dataset = COCO(download_dataset(base_dir:=getenv("BASEDIR", BASEDIR), "validation" if val else "train"))
    with tqdm(total=len(dataset.imgs.keys())) as pbar:
      for x in batch_load_retinanet(dataset, val, base_dir):
        pbar.update(x[0].shape[0])

  def load_llama3(val):
    bs = 24
    samples = 5760 if val else 1_200_000 * 1152
    seqlen = 8192

    max_, min_ = 0, math.inf
    for tokens in tqdm(batch_load_llama3(bs, samples, seqlen, Path(getenv("BASEDIR", "/raid/datasets/c4/")), seed=5760, val=bool(val)), total=samples//bs):
      max_ = max(max_, tokens.shape[1])
      min_ = min(min_, tokens.shape[1])
    print(f"max seq length: {max_}")
    print(f"min seq length: {min_}")

  load_fn_name = f"load_{getenv('MODEL', 'resnet')}"
  if load_fn_name in globals():
    globals()[load_fn_name](getenv("VAL", 1))
