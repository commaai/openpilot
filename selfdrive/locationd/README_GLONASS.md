# KAITAI BUG
kaitai byte aligns before a `switch-on`, which is wrong in the glonass case
as the switch occurs at bit5, alignment at this point would skip 3 bits.

This line is commented out `generated/glonass.cpp:20`

