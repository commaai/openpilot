# debug scripts

## [can_printer.py](can_printer.py)

```
usage: can_printer.py [-h] [--bus BUS] [--max_msg MAX_MSG] [--addr ADDR]

simple CAN data viewer

optional arguments:
  -h, --help         show this help message and exit
  --bus BUS          CAN bus to print out (default: 0)
  --max_msg MAX_MSG  max addr (default: None)
  --addr ADDR
```

## [dump.py](dump.py)

```
usage: dump.py [-h] [--pipe] [--raw] [--json] [--dump-json] [--no-print] [--addr ADDR] [--values VALUES] [socket [socket ...]]

Dump communication sockets. See cereal/services.py for a complete list of available sockets.

positional arguments:
  socket           socket names to dump. defaults to all services defined in cereal

optional arguments:
  -h, --help       show this help message and exit
  --pipe
  --raw
  --json
  --dump-json
  --no-print
  --addr ADDR
  --values VALUES  values to monitor (instead of entire event)
```

## [vw_mqb_config.py](vw_mqb_config.py)

```
usage: vw_mqb_config.py [-h] [--debug] {enable,show,disable}

Shows Volkswagen EPS software and coding info, and enables or disables Heading Control
Assist (Lane Assist). Useful for enabling HCA on cars without factory Lane Assist that want
to use openpilot integrated at the CAN gateway (J533).

positional arguments:
  {enable,show,disable}
                        show or modify current EPS HCA config

optional arguments:
  -h, --help            show this help message and exit
  --debug               enable ISO-TP/UDS stack debugging output

This tool is meant to run directly on a vehicle-installed comma three, with
the openpilot/tmux processes stopped. It should also work on a separate PC with a USB-
attached comma panda. Vehicle ignition must be on. Recommend engine not be running when
making changes. Must turn ignition off and on again for any changes to take effect.
```
