import os
import fnmatch
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.third_party.CythonPEG import cython_peg as cp

extensions = ['*.pyx', '*.pxd']
dirs_to_ignore = ['openpilot', 'third_party']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='*', default=[], help='Path(s) to the .pyx or .pxd file(s).')
    return parser.parse_args()

def read_parse_and_write(file_path):
    with open(file_path, 'r') as file:
        cython_tokens = file.read()

    stub_file, unparsed_tokens = cp.cython_string_2_stub(cython_tokens)
    file_path = os.path.splitext(file_path)[0]
    last_slash_index = file_path.rfind('/')
    new_path = file_path[:last_slash_index]
    stubs_path = new_path + '/stubs'
    os.makedirs(stubs_path, exist_ok=True)
    pyi_file_path =  stubs_path + '/' + file_path[last_slash_index + 1:] + '.pyi'

    with open(pyi_file_path, mode='w+') as f:
        f.write(stub_file)

def has_openpilot(path):
    for d in os.listdir(path):
        if d == 'openpilot':
            return True
    return False

def parse_all_files():
    current_path = os.getcwd()
    last_index = current_path.rfind('openpilot')
    last_openpilot_path = current_path[:last_index+9]

    if(has_openpilot(last_openpilot_path)):
        openpilot_path = last_openpilot_path
    else:
        openpilot_path = current_path[:last_index-1]

    cp.set_indent(" ")
    for root, dirs, files in os.walk(openpilot_path):
        dirs[:] = [d for d in dirs if not (d.startswith('.') or d in dirs_to_ignore)]
        for filename in files:
            if any(fnmatch.fnmatch(filename, extension) for extension in extensions):
                file_path = os.path.join(root, filename)
                read_parse_and_write(file_path)

def parse_one_or_more_files(files):
    for file in files:
        file_path = os.path.join(os.getcwd(), file)
        read_parse_and_write(file_path)

if __name__ == '__main__':
    args = parse_arguments()
    if args.files:
        parse_one_or_more_files(args.files)
    else:
        parse_all_files()
