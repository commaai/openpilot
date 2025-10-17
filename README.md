# Dolphin

- This is an README version for dolphin team

## Usage

### Step 1: Clone, checkout to the vinfast branch

```sh
git clone https://github.com/hungpham3112/openpilot.git
cd openpilot/
```

```sh
git checkout feat/vinfast
```

### Step 2: Setup dependencies

- Install op alias

```sh
./tools/op.sh install

```
- Setup dependencies
```sh
op setup
```

- If using python you should create virtual environment and install pip (Optional)
```sh
op venv
python -m ensurepip && python -m pip install pip
```

### Step 3: How to flash panda

- In `panda/board`, there is `flash.py` script used to flash panda firmware. Here is the flow:

`flash.py (build + sign firmware) -> USB sends command to bootstub -> bootstub uses custom flash driver (unlock/erase/write) to program panda.bin.signed -> NVIC_SystemReset() -> bootstub verifies -> jump to new firmware.`

- Connect panda to USB Type A port and run script to verify tx_hook.

```sh
cd panda/
python examples/vinfast_tester.py
```

If the output shows `I'm from vinfast_tx_hook`, it means you setup correctly.

### Step 4: Change vinfast hooks

- All the vinfast hooks is located in `opendbc/safety/modes/vinfast.h`
