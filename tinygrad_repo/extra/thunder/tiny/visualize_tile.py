from tinygrad.helpers import colored

WARP_THREADS = 64
BASE_TILE_ROWS = 16
BASE_TILE_COLS = 16
BASE_TILE_NEPT = (BASE_TILE_ROWS * BASE_TILE_COLS) // WARP_THREADS
DTYPE_SIZE = 2
INST = "ds_read_b64"

def row_col(threadIdx_x):
  local_warpid = threadIdx_x // WARP_THREADS
  warp_laneid = threadIdx_x % WARP_THREADS

  ret = []

  for inner in range(BASE_TILE_NEPT):
    if BASE_TILE_ROWS == 16 and BASE_TILE_COLS == 16:
      row = warp_laneid % 16
      col = 4 * (warp_laneid // 16)
    elif BASE_TILE_ROWS == 16 and BASE_TILE_COLS == 32:
      row = warp_laneid % 16
      col = 8 * (warp_laneid // 16)

    row_offset = 0
    col_offset = inner

    # swizzle then find row and col
    offset = (row + row_offset) * BASE_TILE_COLS + (col + col_offset)
    offset *= DTYPE_SIZE

    if BASE_TILE_ROWS == 16 and BASE_TILE_COLS == 16:
      swizzle = ((offset % 512) >> 7) << 3
      offset = offset ^ swizzle
    elif BASE_TILE_ROWS == 16 and BASE_TILE_COLS == 32:
      swizzle = ((offset % 1024) >> 9) << 5
      offset = offset ^ swizzle

    offset //= DTYPE_SIZE

    row = offset // BASE_TILE_COLS
    col = offset % BASE_TILE_COLS

    ret.append((row, col))

  return ret

# ===

def shm_phase(inst, threadIdx_x):
  match inst:
    case "ds_read_b128":
      match threadIdx_x:
        case 0 | 1 | 2 | 3 | 12 | 13 | 14 | 15 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27: return 0
        case 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 16 | 17 | 18 | 19 | 28 | 29 | 30 | 31: return 1
        case 32 | 33 | 34 | 35 | 44 | 45 | 46 | 47 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59: return 2
        case 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 48 | 49 | 50 | 51 | 60 | 61 | 62 | 63: return 3
    case "ds_read_b64":
      if threadIdx_x < 32: return 0
      else: return 1
    case "ds_write_b64":
      if threadIdx_x < 16: return 0
      elif threadIdx_x < 32: return 1
      elif threadIdx_x < 48: return 2
      else: return 3

def shm_bank(inst, row, col):
  bank =  row * (BASE_TILE_COLS // 2) + (col // 2)

  match inst:
    case "ds_read_b128": bank = bank % 64
    case "ds_read_b64": bank = bank % 64
    case "ds_write_b64": bank = bank % 32

  return bank

def map_range(value, from_min, from_max, to_min, to_max):
  ratio = (value - from_min) / (from_max - from_min)
  return to_min + ratio * (to_max - to_min)

def shm_bank_gradient(inst, bank):
  # rgb color for each bank
  # for 16 bit elements, two elements per bank row wise

  # gradient from blue to red
  amount = map_range(bank, 0, (64 if inst != "ds_write_b64" else 32) - 1, 0, 120)
  amount = int(amount)
  return (amount, amount // 2, 120 - amount)

def color_code(phase):
  match phase:
    case 0: return "red"
    case 1: return "green"
    case 2: return "blue"
    case 3: return "yellow"

def rgb_bg(text, color):
  return f"\033[48;2;{color[0]};{color[1]};{color[2]}m{text}\033[0m"

def visualize_threads(inst=INST):
  for threadIdx_x in range(WARP_THREADS):
    row, col = zip(*row_col(threadIdx_x))
    print(f"Thread {threadIdx_x:2}: ", end="")
    for r, c in zip(row, col):
      phase = shm_phase(inst, threadIdx_x)
      color = color_code(phase)
      print(f"{color}({r:3},{c:3})\033[0m ", end="")
    print()

  unique_pairs = set()
  for threadIdx_x in range(WARP_THREADS):
    rc_list = row_col(threadIdx_x)
    for rc in rc_list:
      unique_pairs.add(rc)
  assert len(unique_pairs) == 64 * BASE_TILE_NEPT, f"Expected {64 * BASE_TILE_NEPT} unique pairs, got {len(unique_pairs)}"

def visualize_tile(inst=INST):
  tile = [[-1 for _ in range(BASE_TILE_COLS)] for _ in range(BASE_TILE_ROWS)]
  for threadIdx_x in range(WARP_THREADS):
    rc_list = row_col(threadIdx_x)
    for r, c in rc_list:
      try:
        tile[r][c] = threadIdx_x
      except:
        pass

  bank_conflicts = {}

  print("\nTile layout (each number indicates the thread holding that position):")
  for r in range(BASE_TILE_ROWS):
    for c in range(BASE_TILE_COLS):
      phase = shm_phase(inst, tile[r][c])
      bank = shm_bank(inst, r, c)
      color = color_code(phase)
      bank_color = shm_bank_gradient(inst, bank)

      if (bank, phase) not in bank_conflicts:
        bank_conflicts[(bank, phase)] = []
      bank_conflicts[(bank, phase)].append((r, c, tile[r][c]))

      if phase == -1:
        bank_color = (0, 0, 0)

      text = colored(f"{tile[r][c]:2}", color)
      text = rgb_bg(text, bank_color)
      print(f"{text:2}", end=" ")
    print()

  for (bank, phase), positions in bank_conflicts.items():
    if len(positions) > 1:
      unique_threads = set(pos[2] for pos in positions)
      if len(unique_threads) > 1:
        print(f"{len(unique_threads)} way bank conflict: bank {bank}")

if __name__ == "__main__":
  visualize_tile()
  # visualize_threads()
