TODO
======

An incomplete list of known issues and desired featues.

- TX and RX amounts on UI are wrong for a few frames at startup because we
  subtract (total sent - 0). We should initialize sent bytes before displaying.

- Rewrite common/dbc.py to be faster and cleaner, potentially in C++.

- Add module and class level documentation where appropriate.

- Fix lock file cleanup so there isn't always 1 pending upload when the vehicle
  shuts off.
