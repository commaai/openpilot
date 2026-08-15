#!/usr/bin/env python3
"""
Self-contained Qualcomm Hexagon **testsig** generator.

Replicates: python2 elfsigner.py -t 0x67489311 -o .
Dependencies: standard library + cryptography (pip install cryptography).
Multiple serial numbers: use -t multiple times.
"""

import argparse, base64, hashlib, os, struct
from datetime import datetime, timedelta, timezone
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.name import _ASN1Type
from cryptography.x509.oid import ExtensionOID, NameOID, ObjectIdentifier

# Embedded assets (raw base64) --- no external files needed
# Compact test_elf_nop.so contents used for signing. Only bytes covered by
# program headers are needed; section headers and alignment gaps are not.
ORIG_PHDRS = [
    {'t': 1, 'o': 0x0000, 'v': 0x0000, 'p': 0x0000, 'fs': 0x02fc, 'ms': 0x02fc, 'fl': 0x4, 'al': 0x1000},
    {'t': 1, 'o': 0x1000, 'v': 0x1000, 'p': 0x1000, 'fs': 0x0104, 'ms': 0x0104, 'fl': 0x5, 'al': 0x1000},
    {'t': 1, 'o': 0x2000, 'v': 0x2000, 'p': 0x2000, 'fs': 0x0004, 'ms': 0x0004, 'fl': 0x4, 'al': 0x1000},
    {'t': 1, 'o': 0x3000, 'v': 0x4000, 'p': 0x4000, 'fs': 0x00d0, 'ms': 0x0100, 'fl': 0x6, 'al': 0x1000},
    {'t': 2, 'o': 0x3010, 'v': 0x4010, 'p': 0x4010, 'fs': 0x00a8, 'ms': 0x00a8, 'fl': 0x6, 'al': 0x4},
]
ORIG_SEGS = {
    0x0000: base64.b64decode("""
f0VMRgEBAQAAAAAAAAAAAAMApAABAAAAsBAAADQAAACIMQAAAwAAADQAIAAFACgAFQASAAEAAAAAAAAAAAAAAAAAAAD8AgAA/AIA
AAQAAAAAEAAAAQAAAAAQAAAAEAAAABAAAAQBAAAEAQAABQAAAAAQAAABAAAAACAAAAAgAAAAIAAABAAAAAQAAAAEAAAAABAAAAEA
AAAAMAAAAEAAAABAAADQAAAAAAEAAAYAAAAAEAAAAgAAABAwAAAQQAAAEEAAAKgAAACoAAAABgAAAAQAAAADAAAAEwAAABIAAAAR
AAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAAAAAAAAAJAAAACwAAAAwAAAANAAAA
DwAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAwAFAAAAAACwEAAAAAAAAAMABwAAAAAAwBAAAAAAAAADAAgAAAAA
AAAgAAAAAAAAAwAJAAAAAAAAQAAAAAAAAAMACgAAAAAACEAAAAAAAAADAAsAAAAAAMxAAAAAAAAAAwAPAAAAAAAAQQAAAAAAAAMA
EAAPAAAAABAAAGwAAAASAAUAFQAAAAAAAAAAAAAAEgAAAGkAAAAAQQAAAAAAABAA8f9KAAAAzEAAAAQAAAARAA8AMQAAAARAAAAA
AAAAEAAKAAEAAAAIQAAAAAAAABAACwBcAAAAwBAAAEQAAAASAAgAYgAAANBAAAAAAAAAEADx/3UAAAAAQQAAAAAAABAA8f9GAAAA
sBAAAAQAAAASAAcAAF9fRFRPUl9MSVNUX18AX2luaXQAX19yZWdpc3Rlcl9mcmFtZV9pbmZvX2Jhc2VzAF9fQ1RPUl9FTkRfXwBs
aWJjLnNvAG5vcABub3BfdmFyAGxpYmdjYy5zbwBfZmluaQBfZWRhdGEAX19ic3Nfc3RhcnQAX2VuZAB0ZXN0X2VsZl9ub3Auc28A
AADIQAAAIgoAAAAAAAA=
"""),
    0x1000: base64.b64decode("""
AcCdoADbnaEB2J2hGMAJalTP6nH//+pyGNgq8///4HJI3+BxAMAY8wHAgJEIwAEQAkAAeAEoAyg0wABa///7ckz/+3Eb2xjzm//7
vwDAm5EGwAAQAMCgUPj//1k4wJ2RG0CdkR7AHpAAwJ9SAAAAAMFAAAAcxElqDkKc4k9AnJE8wJyRDkIOjADAnFIAAAAAAAAAAAAA
AAAAAAAAAAAAAMBAAAAO1ElqHMCOkQDAnFIAwJ9SAAAAAAAAAAAAAAAAAcCdoADbnaEPwAlqENDqcf//6nIPzyrz///7clD/+3Eb
2w/zm8AbsADAm5EGwAAQAMCgUPj//1kbQJ2RHsAekADAn1I=
"""),
    0x2000: base64.b64decode("AAAAAA=="),
    0x3000: base64.b64decode("""
AAAAAAAAAAAAAAAAAAAAAAEAAAA+AAAAAQAAAFIAAAAOAAAAegAAAAwAAAAAEAAADQAAAMAQAAAEAAAA1AAAAAUAAABkAgAABgAA
ADQBAAAKAAAAigAAAAsAAAAQAAAAAwAAALhAAAACAAAADAAAABQAAAAHAAAAFwAAAPACAAABAABwAwAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBAAAAAAAAAAAAAAAAAAABwEAAACgAAAA==
"""),
}
ORIG_EHDR = ORIG_SEGS[0][:0x34]

ATTESTCA_KEY = base64.b64decode("""LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFb3dJQkFBS0NBUUVBeGpLNkRZbVZxbGIvZHFtb1BtY1FsNkV6Vmx3S3dIUnFCQmdsZzBkQnd6djJMR1NKCi9xWnlpbkFWZlloaEFkKzZwbm1kYk1vMVMzaXdjNXUrWUtRYlFOSGpVMUZrZDN4c29weHA3YTZHNnZUbSsyNVEKVnRHRkdLRHk3dm9mRWRKNTE2NVpQU0JtZ0thcWFQcFIxRlNHQnRDaEV3bzhkVEFZdjBaeXFYeGlmeDU4ejcvbgozT1BBbDJQdkpjZWpzd1J2ekRvcVRDR0xlMjVMbzFiYlQ1RGM4bFhZeWFmcjJVcGpXc1ZBaGlHQVNaK2dIRzYrCk1xWDFUdUlvM3FJTGNOcnRNUVh6K2NMamZaSnYzZExqNmRFSlZ0c3MvT3lMdUQ2bVozMXZUdzMxakhXUGRLeDIKNHh3dXVrbkp5OUVrUElQem5ORTZGcFZ4dmdHQWN4aEhKY0pLdndJQkF3S0NBUUVBaENIUlhsdTVIRG4vcEhFYQoxRVMxdW10M2p1Z0hLdmhHcldWdVY0VFdnaWY1Y3UyeFZHNzNCdlZqcVFXV0FUL1J4RkVUbmR3amg2WEs5NzBwCmxjSzgxZUZDTjR1WVQ2aEliR2hHbm5SWjhmaVovUFExanpaWXV4WDNTZndVdG94Uk9uUTdmaFdacXhuRzhLYmgKT0RoWldlQnJZZ2JTK01xN0tpNzNHNmhCcWhNbU1xcFJoK2dUNUVZbnBHZmJUbVRlUUJveXBrTEtkZm1PUytsWQpDM0lFaXJ1LzRhTXhpc0FDNzMvY3VWenQ1T2k5S3BaVVNhWTJkK1pINWtyWWtEMWhMVDFFRTJxbDhuTS9YZ2NlCk81bDkxeVMxZXg5Z2pzZ0lwTnc5elF0U05qMW80SUFraWRnWDhCak5MdmJSYThWdzBibDgxZXp2UEFoSE9tM0wKazkxTGV3S0JnUUR3MkYrUUZxK2xWcEtzL3pFNTlyUjNCdHBBTFVIVEFqWW1rbTVNUHRLeWtYVUQyQ2lNV3lYVgpxNGNaRnBaV2tUY1YzKzN2Q2FQcnNWdFBlcFZsaTZvM3VyMmR2VVFyQ2IxQUh0d2E4R21Bb2VseUJFSzZSeUdPCjh0dW16aTZqZDEyM0xQQTgvQ1JzL01ncmtjM2QxWTVSSDhzS3F1OGZCdlRncjNId2JTYkRPd0tCZ1FEU3EyRGQKZWxmOWFtZ0dyL3FneHJpclpUaWVKWHVJeUVIUG53cUsvNU1qa01jMUh3cVZMZ1NKUi81Y1dYL0U0UXR1Zko5UQp1b2kzNmExdGlweGhETlNqc214d0gxNi9pQ2hRc2V3Ym9BRkw4anFxdi9yQW12RUtmVmRyelU1V3c3dTY5dndECkdET2QxVnhxYzVFTVJVMTVRbS8wS3NMcjNRQTA3QUdsV3MrV1RRS0JnUUNna0QrMVpIVVk1R0hJcWlEUlR5TDYKQkpHQUhpdmlBWGx2REVtSUtlSE1ZUGl0T3NXeTUyNk9jbG9RdWJtUEMzb09sVWxLQm0xSHk1STAvR09aQjhiUAowZE8rZmkxeUJuNHF2ejFuU3ZFQWEvRDJyWUhSaE1FSjl6MFozc25DVDVQUGMwclRVc0x6VXpBZEM5NlQ0N1EyCkZUSUhISjlxQktOQWRQYWdTTVNDSndLQmdRQ01ja0NUcHVWVG5FVlp5cWNWMmRCeVEzc1VHUDBGMnRhS2FnY0gKVlF6Q1lJVE9GTEc0eVZoYmhWUTlrUC9ZbGdlZS9iK0xKd1hQOFI1SkJ4THJYZU1YekVoS3Y1Ui9zQnJnZHAxbgp3QURkVENjY2YveUFaMHRjVTQrZE00bVBMU2ZSK2YxWFpYZStqajJjVFF0ZGc0ajdnWi80SElIeWsxVjRuVlp1ClBJcGtNd0tCZ0FrZ0ljYlh0VHlDbGxJWnFWVTJseUUrcng3Wm5RNGs5ZTBaSENhaXFBT3JDNVBseUtwQ0hRZEgKWGYxcVhtRTBPeFhBeVEycU8wbWJWMzgza2ovU3E3b0p0RHRveS9Bc3ZiRG1vNjZjTzJRSXRSRmZPazJ6Q3UycQpDMFpraE9nazNGWUo4aXloV3pPV3VDWExKck9QVVVqekFkSDJTYWloVy9KY0hVUmhNTnBTCi0tLS0tRU5EIFJTQSBQUklWQVRFIEtFWS0tLS0tCg==""")
ATTESTCA_CERT = base64.b64decode("""MIIEIDCCAwigAwIBAgIBBTANBgkqhkiG9w0BAQsFADCBsjELMAkGA1UEBhMCVVMxEzARBgNVBAgTCkNhbGlmb3JuaWExEjAQBgNVBAcTCVNhbiBEaWVnbzEwMC4GA1UECxMnR2VuZXJhbCBVc2UgVGVzdCBLZXkgKGZvciB0ZXN0aW5nIG9ubHkpMRowGAYDVQQLExFDRE1BIFRlY2hub2xvZ2llczERMA8GA1UEChMIUVVBTENPTU0xGTAXBgNVBAMTEFFQU0EgU0hBMjU2IFJvb3QwHhcNMTMwNDIyMjIwNjE4WhcNMzMwNDE3MjIwNjE4WjB8MQswCQYDVQQGEwJVUzELMAkGA1UECBMCQ0ExEjAQBgNVBAcTCVNhbiBEaWVnbzEaMBgGA1UECxMRQ0RNQSBUZWNobm9sb2dpZXMxETAPBgNVBAoTCFFVQUxDT01NMR0wGwYDVQQDExRRUFNBIE9QRU5EU1AgVEVTVCBDQTCCASAwDQYJKoZIhvcNAQEBBQADggENADCCAQgCggEBAMYyug2JlapW/3apqD5nEJehM1ZcCsB0agQYJYNHQcM79ixkif6mcopwFX2IYQHfuqZ5nWzKNUt4sHObvmCkG0DR41NRZHd8bKKcae2uhur05vtuUFbRhRig8u76HxHSedeuWT0gZoCmqmj6UdRUhgbQoRMKPHUwGL9Gcql8Yn8efM+/59zjwJdj7yXHo7MEb8w6Kkwhi3tuS6NW20+Q3PJV2Mmn69lKY1rFQIYhgEmfoBxuvjKl9U7iKN6iC3Da7TEF8/nC432Sb93S4+nRCVbbLPzsi7g+pmd9b08N9Yx1j3SsduMcLrpJycvRJDyD85zROhaVcb4BgHMYRyXCSr8CAQOjeDB2MB8GA1UdIwQYMBaAFElk8+VAE1VZc2dnWT99Qreru/tXMB0GA1UdDgQWBBQnxAfEeRhBNAnuLkunmI4I+aSyOzAPBgNVHRMECDAGAQH/AgEAMAsGA1UdDwQEAwIBBjAWBgorBgEEAYspCQYDBAgAAeJAAAn78TANBgkqhkiG9w0BAQsFAAOCAQEAYHPAAlh+ezXdqUDIptraYfoiVxw2YsX++Ytg2eJ69YFVlCo33bLJFwQMj+zTMauRgvLew2cZTK47ghVV7130M13E53aN49p/DTOe3u5OFGA+z+ZLrqhraUPT+UhaAuVO9Yu9eOLudsPvgJTeD1a7RaC6PmPsUFPxLUlmlJn3lSXjlYe98+hittLnJ9gTnjdTVH/PgEJhMvUcjjyBWdRsog54VpyqesqLJedC4OF7fHJZ4S7rxDAINI15aDBQrOW/LD6HsBdr4WikS5Lnmecaw+2Um/ge/3Jl/kFBgh8EyORmSzaN4q1OoPYykxTGxenP3Z6D9WJurPd0d0fnuf+bNw==""")
ROOTCA_CERT = base64.b64decode("""MIIEGzCCAwOgAwIBAgIBATANBgkqhkiG9w0BAQsFADCBsjELMAkGA1UEBhMCVVMxEzARBgNVBAgTCkNhbGlmb3JuaWExEjAQBgNVBAcTCVNhbiBEaWVnbzEwMC4GA1UECxMnR2VuZXJhbCBVc2UgVGVzdCBLZXkgKGZvciB0ZXN0aW5nIG9ubHkpMRowGAYDVQQLExFDRE1BIFRlY2hub2xvZ2llczERMA8GA1UEChMIUVVBTENPTU0xGTAXBgNVBAMTEFFQU0EgU0hBMjU2IFJvb3QwHhcNMTMwMzI4MjMxOTA4WhcNMzMwMzIzMjMxOTA4WjCBsjELMAkGA1UEBhMCVVMxEzARBgNVBAgTCkNhbGlmb3JuaWExEjAQBgNVBAcTCVNhbiBEaWVnbzEwMC4GA1UECxMnR2VuZXJhbCBVc2UgVGVzdCBLZXkgKGZvciB0ZXN0aW5nIG9ubHkpMRowGAYDVQQLExFDRE1BIFRlY2hub2xvZ2llczERMA8GA1UEChMIUVVBTENPTU0xGTAXBgNVBAMTEFFQU0EgU0hBMjU2IFJvb3QwggEgMA0GCSqGSIb3DQEBAQUAA4IBDQAwggEIAoIBAQC3mmlyc5XmZ4nQcUf8gXoHX82fCU12SW6VJdlz5IyKOJzl+IeYs2ArpkDHXaF2NwYvS4cJVBHtvx5TLbsBMAF9teFORqSs6wl+r+3nQwCogNOn/8JZrcPdxkjA8cVAkydxSK0jPxAdAGf8vGXD7tKDWWZyHquPoWqNVG/P4OyHAWMKCYg/w7/5MYTOcV1TXW2BraH7dztGkS4ey2hsOPlJzxP74cN1WyXjLPkn5CZWkx+95CKN5i+T9S+FeKD/1zbxuNlwv4x3x1Ohw9KBJYQzrB/wP9wrsVEnh2K9jy7rapKFFWOgQj8omg1EbIMqdOHuSZYcexFvAqN233xxluDBAgEDozwwOjAdBgNVHQ4EFgQUSWTz5UATVVlzZ2dZP31Ct6u7+1cwDAYDVR0TBAUwAwEB/zALBgNVHQ8EBAMCAQYwDQYJKoZIhvcNAQELBQADggEBAJ31kC2nKcTK1XrArhpkmnAX8zCPZkl+Azm7qF/Mr09h8FJiXJ7hBBoLHD+/+DifgUkLF4EjLOUnTTUPLPPKS5KgvuDkgJKvAMvv/GXxuGabdl4EebYCbJvnEgwkyG4pYVd5pGHQ0z2Md4nI6aMYco2X44bIjdqxFJwWOgPhioef1KbO/6CzykG0cPvpJB0XkWR8QGztFt9HofF+uVgpY2t1dL4/SuM/wJSeS8rdqstk0BYq/EDFFs99r1wP2R6hjJgCPkMvo7WiSE2yRrEkMNhgyEKrrD4pr7YWtsFPkYeTMXOvYoY16aOfvcrw0FfH+yATdn/OdAQ2saJISmilWh4=""")

PT_NULL = 0
PF_OS_PHDR = 0x07000000
PF_OS_HASH = 0x02200000
MBN_V3 = 3
SIG_SIZE = 256
CC_SIZE = 0x1800
PAD_FF = b'\xff'
IPAD = 0x3636363636363636
OPAD = 0x5C5C5C5C5C5C5C5C


def _pad(data, size, pad=PAD_FF):
  if len(data) > size:
    raise ValueError("data too large: %d > %d" % (len(data), size))
  return data + pad * (size - len(data))


def _orig_segment_data(ph):
  off, size = ph['o'], ph['fs']
  for base_off, data in ORIG_SEGS.items():
    rel = off - base_off
    if 0 <= rel and rel + size <= len(data):
      return data[rel:rel + size]
  raise KeyError("missing original segment at 0x%x" % off)


def _build_ehdr(base_ehdr, num_phdrs):
  e = bytearray(base_ehdr)
  struct.pack_into('<I', e, 0x20, 0)
  struct.pack_into('<H', e, 0x2c, num_phdrs)
  struct.pack_into('<H', e, 0x30, 0)
  struct.pack_into('<H', e, 0x32, 0)
  return bytes(e)


def _build_elf(ehdr, phs, segs):
  phoff = struct.unpack_from('<I', ehdr, 0x1c)[0]
  d = bytes(ehdr)
  if len(d) < phoff:
    d += b'\x00' * (phoff - len(d))
  for ph in phs:
    d += struct.pack('<IIIIIIII', ph['t'], ph['o'], ph['v'],
             ph['p'], ph['fs'], ph['ms'], ph['fl'], ph['al'])
  for off, sdata in sorted(segs.items()):
    if len(d) < off:
      d += b'\x00' * (off - len(d))
    d = d[:off] + sdata + d[off + len(sdata):]
  return d


def _qti_hmac(data, msm=0, sw=0):
  def _u(v):
    return bytes.fromhex(format(v, 'x').zfill(16))
  Si, So = _u(sw ^ IPAD), _u(msm ^ OPAD)
  a = hashlib.sha256(data).digest()
  b = hashlib.sha256(Si + a).digest()
  c = hashlib.sha256(So + b).digest()
  return c


def _raw_pkcs1_sign(private_key, data):
  """Raw RSA-PKCS1-v1_5 signing WITHOUT DigestInfo wrapper.
  This matches OpenSSL:  pkeyutl -sign -pkeyopt rsa_padding_mode:pkcs1"""
  numbers = private_key.private_numbers()
  d = numbers.d
  n = numbers.public_numbers.n
  key_len = (n.bit_length() + 7) // 8
  pad_len = key_len - 3 - len(data)
  if pad_len < 8:
    raise ValueError("data too long for key size")
  em = b'\x00\x01' + b'\xff' * pad_len + b'\x00' + data
  m_int = int.from_bytes(em, 'big')
  sig_int = pow(m_int, d, n)
  return sig_int.to_bytes(key_len, 'big')


def _new_cert(ca_key, ca_cert, attrs):
  k = rsa.generate_private_key(public_exponent=3, key_size=2048)
  def _na(oid, value, typ):
    return x509.NameAttribute(oid, value, _type=typ)

  # Match the SecTools/OpenSSL attestation cert profile accepted by DSP loaders.
  n = [
    _na(NameOID.COUNTRY_NAME, "US", _ASN1Type.PrintableString),
    _na(NameOID.COMMON_NAME, "SecTools Test User", _ASN1Type.PrintableString),
    _na(NameOID.LOCALITY_NAME, "San Diego", _ASN1Type.PrintableString),
    _na(NameOID.ORGANIZATION_NAME, "SecTools", _ASN1Type.PrintableString),
    _na(NameOID.STATE_OR_PROVINCE_NAME, "California", _ASN1Type.PrintableString),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "01 %.16X SW_ID" % attrs['sw'], _ASN1Type.T61String),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "02 %.16X HW_ID" % attrs['hw'], _ASN1Type.T61String),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "04 %.4X OEM_ID" % attrs['oid'], _ASN1Type.T61String),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "05 %.8X SW_SIZE" % attrs['sz'], _ASN1Type.T61String),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "06 %.4X MODEL_ID" % attrs['mid'], _ASN1Type.T61String),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "07 0001 %s" % attrs['ha'], _ASN1Type.PrintableString),
    _na(NameOID.ORGANIZATIONAL_UNIT_NAME, "03 %.16X DEBUG" % attrs['dbg'], _ASN1Type.PrintableString),
  ]

  nvb = datetime.now(timezone.utc).replace(microsecond=0)
  nva = nvb + timedelta(days=20 * 365)
  b = (x509.CertificateBuilder()
     .subject_name(x509.Name(n))
     .issuer_name(ca_cert.subject)
     .public_key(k.public_key())
     .serial_number(1)
     .not_valid_before(nvb)
     .not_valid_after(nva))
  b = b.add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False)
  b = b.add_extension(x509.UnrecognizedExtension(ExtensionOID.BASIC_CONSTRAINTS, b'\x30\x03\x02\x01\x00'), False)
  b = b.add_extension(x509.KeyUsage(
    digital_signature=True,
    content_commitment=False,
    key_encipherment=False,
    data_encipherment=False,
    key_agreement=False,
    key_cert_sign=False,
    crl_sign=False,
    encipher_only=False,
    decipher_only=False,
  ), False)
  b = b.add_extension(x509.UnrecognizedExtension(ObjectIdentifier("1.3.6.1.4.1.1449.9.6.3"), b'\x00\x01\xe2\x40\x00\x01\xe2\x40'), False)
  return k, b.sign(ca_key, hashes.SHA256())


def _build_chain(attest, ca, root):
  c = (attest.public_bytes(serialization.Encoding.DER) +
     ca.public_bytes(serialization.Encoding.DER) +
     root.public_bytes(serialization.Encoding.DER))
  return _pad(c, CC_SIZE, b'\xff')


def _sign(serial_num, out_dir):
  orig = ORIG_PHDRS
  base_ehdr = ORIG_EHDR

  nph = len(orig) + 2
  phsz = nph * 32
  ehsz = len(base_ehdr)
  phoff = struct.unpack_from('<I', base_ehdr, 0x1c)[0]
  hv = 0x5000
  ho = 0x1000
  hfs = 0x1a08

  prog = {'t': PT_NULL, 'o': 0, 'v': 0, 'p': 0,
      'fs': ehsz + phsz, 'ms': 0, 'fl': PF_OS_PHDR, 'al': 0}
  hph = {'t': PT_NULL, 'o': ho, 'v': hv, 'p': hv,
       'fs': hfs, 'ms': 0x2000, 'fl': PF_OS_HASH, 'al': 0x1000}
  shifted = [dict(ph, o=ph['o'] + 0x3000) for ph in orig]
  allph = [prog, hph] + shifted

  # Temporary ELF for hash 0
  tmp_segs = {}
  for i, ph in enumerate(shifted):
    tmp_segs[ph['o']] = _orig_segment_data(orig[i])
  ehdr = _build_ehdr(base_ehdr, nph)
  tmp = _build_elf(ehdr, allph, tmp_segs)

  phb = tmp[phoff:phoff + phsz]
  hash0 = hashlib.sha256(ehdr + phb).digest()
  hash1 = struct.pack('<I', serial_num) + b'\x00' * 28
  hs = [hash0, hash1]
  for ph in orig:
    hs.append(hashlib.sha256(_orig_segment_data(ph)).digest())
  ht = b''.join(hs)

  cs, ss, ccs = len(ht), SIG_SIZE, CC_SIZE
  dst = hv + 40
  sp = dst + cs
  cp = sp + ss
  isz = cs + ss + ccs

  mbn = struct.pack('<IIIIIIIIII',
            0, MBN_V3, 0, dst, isz, cs, sp, ss, cp, ccs)
  dts = mbn + ht
  hmac = _qti_hmac(dts)

  ca_key = serialization.load_pem_private_key(ATTESTCA_KEY, password=None)
  ca_cert = x509.load_der_x509_certificate(ATTESTCA_CERT)
  root_cert = x509.load_der_x509_certificate(ROOTCA_CERT)

  attrs = {'sw': 0, 'hw': 0, 'oid': 0, 'mid': 0,
       'sz': len(dts), 'ha': 'SHA256', 'dbg': 2}
  nk, ac = _new_cert(ca_key, ca_cert, attrs)
  sig = _raw_pkcs1_sign(nk, hmac)
  sig = _pad(sig, SIG_SIZE, b'\x00')
  cc = _build_chain(ac, ca_cert, root_cert)

  hseg = mbn + ht + sig + cc
  if len(hseg) != hfs:
    raise RuntimeError("hash seg size mismatch %d vs %d" % (len(hseg), hfs))

  segs = {ho: hseg}
  for i, ph in enumerate(shifted):
    segs[ph['o']] = _orig_segment_data(orig[i])

  final = _build_elf(ehdr, allph, segs)

  os.makedirs(out_dir, exist_ok=True)
  out = os.path.join(out_dir, "testsig-0x%08x.so" % serial_num)
  with open(out, 'wb') as f:
    f.write(final)
  print("Signing complete! Output saved at %s" % out)
  return out


def main():
  p = argparse.ArgumentParser(description="Qualcomm testsig generator")
  p.add_argument("-t", "--testsig", action="append", default=None,
           help="serial number (e.g. 0x67489311); repeatable. Defaults to value from /sys/devices/soc0/serial_number")
  p.add_argument("-o", "--output_dir", default=".")
  args = p.parse_args()

  serials = args.testsig if args.testsig else []

  if not serials:
    # Read default serial number from device
    try:
      with open('/sys/devices/soc0/serial_number', 'r') as f:
        serial_str = f.read().strip()
      serials = [serial_str]
      print("Using serial number from /sys/devices/soc0/serial_number: %s" % serial_str)
    except FileNotFoundError:
      raise SystemExit("Error: No serial number provided (-t) and /sys/devices/soc0/serial_number not found.")
    except PermissionError:
      raise SystemExit("Error: Cannot read /sys/devices/soc0/serial_number (permission denied).")

  for s in serials:
    v = int(s.strip(), 0)
    if not (0 <= v <= 0xFFFFFFFF):
      raise ValueError("bad serial %r" % s)
    _sign(v, args.output_dir)


if __name__ == "__main__":
  main()
