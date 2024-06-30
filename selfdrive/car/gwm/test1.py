def crc8(data: bytes, poly: int = 0x1D, init_val: int = 0x00, xor_out: int = 0x2D):
    crc = init_val
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFF  # Ensure crc remains 8-bit
    return crc ^ xor_out

# Example list of strings
string_list = ['0', '4', '0', '60', '0', '0', '11']

# Convert the list of strings to a byte string
byte_array = bytearray(int(x) for x in string_list)
message = bytes(byte_array)

# Calculate the CRC value
crc_value = crc8(message)
print(f"CRC-8 value: {crc_value}")

# For demonstration, let's validate against the expected CRC values for multiple examples
test_cases = [
    (['0', '4', '0', '60', '0', '0', '11'], 90),
    (['0', '4', '0', '40', '0', '0', '12'], 98),
    (['0', '4', '0', '40', '0', '0', '13'], 127),
    (['0', '4', '0', '40', '0', '0', '14'], 88),
    (['0', '4', '0', '40', '0', '0', '0'], 254),
    (['0', '6', '0', '40', '0', '0', '1'], 111),
    (['0', '6', '0', '40', '0', '0', '2'], 72),
    (['0', '6', '0', '20', '0', '0', '3'], 232),
    (['0', '6', '0', '20', '0', '0', '4'], 187),
    (['0', '6', '0', '20', '0', '0', '5'], 166),
    (['0', '6', '0', '20', '0', '0', '6'], 129),
    (['0', '6', '0', '20', '0', '0', '7'], 156),
    (['0', '6', '0', '20', '0', '0', '8'], 39),
    (['0', '6', '0', '20', '0', '0', '9'], 58),
]

for case, expected_crc in test_cases:
    byte_array = bytearray(int(x) for x in case)
    message = bytes(byte_array)
    crc_value = crc8(message)
    print(f"CRC-8 value for {case}: {crc_value}, Expected: {expected_crc}, {'Match' if crc_value == expected_crc else 'Mismatch'}")
