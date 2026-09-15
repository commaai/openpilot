# openpilot CTF

* all the flags are contained in this route: `0c7f0c7f0c7f0c7f/2026-10-31--13-37-00`
* there's 2 flags in each segment, with roughly increasing difficulty
* everything you'll need to find the flags is in the openpilot repo
  * grep is also your friend
  * first, [set up](https://github.com/commaai/openpilot/tree/master/tools) your PC
  * read the docs and check out the tools in `openpilot/tools/`
  * tip: once you get replay and the UI up, start by familiarizing yourself with seeking in replay

## Getting started

```bash
# start the route replay
openpilot/tools/replay/replay '0c7f0c7f0c7f0c7f/2026-10-31--13-37-00' --dcam --ecam

# start the UI in another terminal
cd openpilot/selfdrive/ui && ./ui.py
```
