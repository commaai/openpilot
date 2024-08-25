# comma serial

The comma serial gets you access to a low level serial console on your comma three, while providing a stable 12V to power the device.

The serial is available on the [comma shop](https://comma.ai/shop/products/comma-serial).

## setup

* Connect all three cables to the serial
* Connect the USB A to your computer
* Connect the USB-C to the OBD-C port on your comma three

## usage

On the comma three, the serial console is exposed through a UART-to-USB chip, and

```
tools/serial/connect.sh can be used to connect.
```

On the comma 3X, the serial console is accessible through the `panda` use the script

```
panda/tests/som_debug.sh
```
