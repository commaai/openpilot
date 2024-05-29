#!/usr/bin/env python3
import sys
import argparse
import json
import codecs

from hexdump import hexdump
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.tools.lib.live_logreader import raw_live_logreader

codecs.register_error("strict", codecs.backslashreplace_errors)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Dump communication sockets. See cereal/services.py for a complete list of available sockets.')
	parser.add_argument('--pipe', action='store_true')
	parser.add_argument('--raw', action='store_true')
	parser.add_argument('--json', action='store_true')
	parser.add_argument('--dump-json', action='store_true')
	parser.add_argument('--no-print', action='store_true')
	parser.add_argument('--addr', default='127.0.0.1')
	parser.add_argument('--values', help='values to monitor (instead of entire event)')
	parser.add_argument('-c', '--count', type=int, help='number of iterations to run before exiting')
	parser.add_argument('-o', '--output', help='output file')
	parser.add_argument("socket", type=str, nargs='*', default=list(SERVICE_LIST.keys()), help="socket names to dump. defaults to all services defined in cereal")
	args = parser.parse_args()
	
	lr = raw_live_logreader(args.socket, args.addr)
	
	values = None
	if args.values:
		values = [s.strip().split(".") for s in args.values.split(",")]
	
	count = args.count if args.count else sys.maxsize
	iterations = 0
	
	output_file = open(args.output, 'w') if args.output else sys.stdout
	
	output_file.write('--------------------------------------------------\n')
	output_file.write(f'    Dump communication sockets: {", ".join(args.socket)}\n')
	output_file.write('--------------------------------------------------\n\n')
	
	try:
		for msg in lr:
			with log.Event.from_bytes(msg) as evt:
				if iterations >= count:
					break
				
				if not args.no_print:
					if args.pipe:
						output_file.write(str(msg))
						output_file.flush()
					elif args.raw:
						hexdump(msg)
					elif args.json:
						output_file.write(json.dumps(evt.to_dict()))
						output_file.write('\n')
					elif args.dump_json:
						output_file.write(json.dumps(evt.to_dict()))
						output_file.write('\n')
					elif values:
						output_file.write(f"logMonotime = {evt.logMonoTime}\n")
						for value in values:
							if hasattr(evt, value[0]):
								item = evt
								for key in value:
									item = getattr(item, key)
								output_file.write(f"{'.'.join(value)} = {item}\n")
						output_file.write("\n")
					else:
						try:
							output_file.write(str(evt))
						except UnicodeDecodeError:
							w = evt.which()
							s = f"( logMonoTime {evt.logMonoTime}\n  {w} = "
							s += str(evt.__getattr__(w))
							s += f"\n  valid = {evt.valid} )\n"
							output_file.write(s)
				
				iterations += 1
			
			output_file.write('\n\n--------------------------------------------------\n')
			output_file.write(f'    Dump Count: {iterations}\n')
			output_file.write('--------------------------------------------------\n\n')
	
	except KeyboardInterrupt:
		pass
	
	if args.output:
		output_file.close()
