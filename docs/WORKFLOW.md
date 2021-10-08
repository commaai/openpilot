# the workflow: how we develop openpilot

Most development happens on desks instead of cars and with publicly available tools. In fact, neither a car or comma device are required for a lot of openpilot development.

## computer

We use high end workstations running Ubuntu 20.04, but most work doesn't require any fancy.

Install Ubuntu 20.04 on a machine and follow the [setup guide](../tools/).


## what's on our desk?

jungle + C3



## random tips

TODO: flesh these points out more and organize them better

* usually we'll create small scripts to handle common debugging tasks. those generally live in selfdrive/debug.
  * e.g. selfdrive/debug/dump.py to dump messages from arbitrary services
* manager is rarely run on the computer. Usually, only the individual processes are run, sometimes with replay to provide data.
* Most of us just use a simple text editor. Linting and sanity checks will be done with git hooks, using pre-commit.
