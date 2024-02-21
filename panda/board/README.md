## Programming

```
./flash.py        # flash application
./recover.py      # flash bootstub
```

## Debugging

To print out the serial console from the STM32, run `tests/debug_console.py`

Troubleshooting
----

If your panda will not flash and green LED is on, use `recover.py`.
If panda is blinking fast with green LED, use `flash.py`.

Otherwise if LED is off and panda can't be seen with `lsusb` command, use [panda paw](https://comma.ai/shop/products/panda-paw) to go into DFU mode.

If your device has an internal panda and none of the above works, try running `../tests/reflash_internal_panda.py`.
