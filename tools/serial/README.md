# comma serial

The comma serial gets you access to a low level serial console on your comma three, while providing a stable 12V to power the device.

The serial is available on the [comma shop](https://comma.ai/shop/products/comma-serial).

## setup

* Connect all three cables to the serial
* Connect the USB A to your computer
* Connect the USB-C to the OBD-C port on your comma three

## usage

```
sudo screen /dev/ttyUSB0 115200
```
or use `connect.sh` to run the previous command in a loop.


The username and password are both `comma`.
