DEBUG = 1

def security_access_algorithm(seed):
    # k4 = 4 bits
    k4 = seed >> 5 & 8 | seed >> 0xB & 4 | seed >> 0x18 & 1 | seed >> 1 & 2
    if DEBUG: print("k4=",hex(k4))
    if DEBUG: print("seed&0x20000=",hex(seed&0x20000))

    # k32 = 32 bits
    if seed & 0x20000 == 0:
        k32 = (seed & ~(0xff << k4 & 0xFFFFFFFF)) << 0x20 - k4 & 0xFFFFFFFF | seed >> k4 & 0xFFFFFFFF
    else:
        k32 = (~(0xff << k4 & 0xFFFFFFFF) << 0x20 - k4 & seed & 0xFFFFFFFF) >> 0x20 - k4 & 0xFFFFFFFF | seed << k4 & 0xFFFFFFFF
    if DEBUG: print("k32=",hex(k32))

    # k2 = 2 bits
    k2 = seed >> 4 & 2 | seed >> 0x1F
    if DEBUG: print("k2=",hex(k2))
    if k2 == 0:
        return k32 | seed
    if k2 == 1:
        return k32 & seed
    if k2 == 2:
        return k32 ^ seed
    return k32

if __name__== "__main__":
    seed = 0x01234567
    key = security_access_algorithm(seed)
    print("seed=" , hex(seed))
    print(" key=" , hex(key))
