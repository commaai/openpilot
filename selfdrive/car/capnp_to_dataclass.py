import re

TYPE_MAP = {
  'Text': 'str',
  'Bool': 'bool',
  'Int16': 'int',
  'UInt32': 'int',
  'UInt64': 'int',
  'Float32': 'float',
  'Data': 'bytes',
  'List': 'list',
}

TXT = """

"""

if __name__ == '__main__':

  TXT = input('Paste one struct only')

  in_struct = False

  builder = []

  for line in TXT.splitlines():
    line = line.strip()
    if re.search(':.*;', line):
      if not in_struct:
        # print(line)
        name, typ, cmt = re.search('([a-zA-Z]+)\s*@\s*\d+\s*:\s*([a-zA-Z0-9\(\)]+)(?:.*#(.*))?', line.strip()).groups()  # noqa # type: ignore
        # print((name, typ, cmt))
        if name.endswith('DEPRECATED'):
          continue

        if 'List' in typ:
          second_typ = typ.split("(")[1][:-1]
          typ = f'list[{TYPE_MAP.get(second_typ, second_typ)}]'

        new_typ = TYPE_MAP.get(typ, typ)
        # print(f'  {name}: {new_typ} = auto_field()')
        # print()
        new_cmt = f'  # {cmt.strip()}' if cmt else ''
        builder.append(f'  {name}: {new_typ} = auto_field(){new_cmt}')
    elif re.search('{', line):
      in_struct = True
    elif re.search('}', line):
      in_struct = False
    elif line == '':
      builder.append('')

  print('\n'.join(builder))
