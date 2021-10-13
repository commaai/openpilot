Welcome to the first part of the comma CTF!
----------------

* all the flags are contained in this route: `0c7f0c7f0c7f0c7f|2021-10-13--13-00-00`
* there's 2 flags in each segment, with roughly increasing difficulty
* everything you'll need to find the flags is in the openpilot repo
  * grep is also your friend
  * first, [setup](https://github.com/commaai/openpilot/tree/master/tools#setup-your-pc) your PC
  * read the docs & checkout out the tools in tools/ and selfdrive/debug/
  * tip: once you get the replay and UI up, start by familiarizing yourself with seeking in replay

getting started
```bash
# start the route reply
cd selfdrive/ui/replay
./replay '0c7f0c7f0c7f0c7f|2021-10-13--13-00-00' --dcam --ecam

# start the UI in another terminal
selfdrive/ui/ui
```
