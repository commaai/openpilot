#!/usr/bin/env python

from __future__ import absolute_import, print_function

import json
import optparse
import sys
import time

from . import DecodeError, __package__, __version__, decode, encode


def main():

    usage = '''Encodes or decodes JSON Web Tokens based on input.

  %prog [options] input

Decoding examples:

  %prog --key=secret json.web.token
  %prog --no-verify json.web.token

Encoding requires the key option and takes space separated key/value pairs
separated by equals (=) as input. Examples:

  %prog --key=secret iss=me exp=1302049071
  %prog --key=secret foo=bar exp=+10

The exp key is special and can take an offset to current Unix time.\
'''
    p = optparse.OptionParser(
        usage=usage,
        prog=__package__,
        version='%s %s' % (__package__, __version__),
    )

    p.add_option(
        '-n', '--no-verify',
        action='store_false',
        dest='verify',
        default=True,
        help='ignore signature and claims verification on decode'
    )

    p.add_option(
        '--key',
        dest='key',
        metavar='KEY',
        default=None,
        help='set the secret key to sign with'
    )

    p.add_option(
        '--alg',
        dest='algorithm',
        metavar='ALG',
        default='HS256',
        help='set crypto algorithm to sign with. default=HS256'
    )

    options, arguments = p.parse_args()

    if len(arguments) > 0 or not sys.stdin.isatty():
        if len(arguments) == 1 and (not options.verify or options.key):
            # Try to decode
            try:
                if not sys.stdin.isatty():
                    token = sys.stdin.read()
                else:
                    token = arguments[0]

                token = token.encode('utf-8')
                data = decode(token, key=options.key, verify=options.verify)

                print(json.dumps(data))
                sys.exit(0)
            except DecodeError as e:
                print(e)
                sys.exit(1)

        # Try to encode
        if options.key is None:
            print('Key is required when encoding. See --help for usage.')
            sys.exit(1)

        # Build payload object to encode
        payload = {}

        for arg in arguments:
            try:
                k, v = arg.split('=', 1)

                # exp +offset special case?
                if k == 'exp' and v[0] == '+' and len(v) > 1:
                    v = str(int(time.time()+int(v[1:])))

                # Cast to integer?
                if v.isdigit():
                    v = int(v)
                else:
                    # Cast to float?
                    try:
                        v = float(v)
                    except ValueError:
                        pass

                # Cast to true, false, or null?
                constants = {'true': True, 'false': False, 'null': None}

                if v in constants:
                    v = constants[v]

                payload[k] = v
            except ValueError:
                print('Invalid encoding input at {}'.format(arg))
                sys.exit(1)

        try:
            token = encode(
                payload,
                key=options.key,
                algorithm=options.algorithm
            )

            print(token)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
