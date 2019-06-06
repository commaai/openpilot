import base64
import binascii

try:
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_rfc6979_signature, encode_rfc6979_signature
    )
except ImportError:
    pass


def base64url_decode(input):
    rem = len(input) % 4

    if rem > 0:
        input += b'=' * (4 - rem)

    return base64.urlsafe_b64decode(input)


def base64url_encode(input):
    return base64.urlsafe_b64encode(input).replace(b'=', b'')


def merge_dict(original, updates):
    if not updates:
        return original

    try:
        merged_options = original.copy()
        merged_options.update(updates)
    except (AttributeError, ValueError) as e:
        raise TypeError('original and updates must be a dictionary: %s' % e)

    return merged_options


def number_to_bytes(num, num_bytes):
    padded_hex = '%0*x' % (2 * num_bytes, num)
    big_endian = binascii.a2b_hex(padded_hex.encode('ascii'))
    return big_endian


def bytes_to_number(string):
    return int(binascii.b2a_hex(string), 16)


def der_to_raw_signature(der_sig, curve):
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8

    r, s = decode_rfc6979_signature(der_sig)

    return number_to_bytes(r, num_bytes) + number_to_bytes(s, num_bytes)


def raw_to_der_signature(raw_sig, curve):
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8

    if len(raw_sig) != 2 * num_bytes:
        raise ValueError('Invalid signature')

    r = bytes_to_number(raw_sig[:num_bytes])
    s = bytes_to_number(raw_sig[num_bytes:])

    return encode_rfc6979_signature(r, s)
