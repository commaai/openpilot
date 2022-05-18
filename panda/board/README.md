Dependencies
--------

**Mac**

```
xcode-select --install
./get_sdk_mac.sh
```

**Debian / Ubuntu**

```
./get_sdk.sh
```


Programming
----

**Panda**

```
scons -u -j$(nproc)  # Compile
./flash_h7.sh        # for red panda
./flash.sh           # for other pandas
```

Troubleshooting
----

If your panda will not flash and is quickly blinking a single Green LED, use:
```
./recover_h7.sh  # for red panda
./recover.sh     # for other pandas
```

A [panda paw](https://comma.ai/shop/products/panda-paw) can be used to put panda into DFU mode.


[dfu-util](http://github.com/dsigma/dfu-util.git) for flashing
