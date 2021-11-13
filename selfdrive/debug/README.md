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

Dump communcation sockets. See cereal/services.py for a complete list of available sockets.

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
