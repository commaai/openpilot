Programming
----

**Panda**

```
./recover.sh           # flash bootstub
```

```
./flash.sh           # flash application
```

Troubleshooting
----

If your panda will not flash and green LED is on, use `recover.sh`.
If panda is blinking fast with green LED, use `flash.sh`.
Otherwise if LED is off and panda can't be seen with `lsusb` command, use [panda paw](https://comma.ai/shop/products/panda-paw) to go into DFU mode.


[dfu-util](http://github.com/dsigma/dfu-util.git) for flashing
