#!/usr/bin/env python3
import os
import ast
import re
import subprocess

fouts = {x.decode('utf-8') for x in subprocess.check_output(['git', 'ls-files']).strip().split()}

pyf = [os.path.join(root, f) for root, _, files in os.scandir(".") for f in files if re.match(r".*\.py$",f)]

imps = set()

class Analyzer(ast.NodeVisitor):
    def visit_Import(self, node):
        for alias in node.names:
            imps.add(alias.name)
        self.generic_visit(node)
    def visit_ImportFrom(self, node):
        imps.add(node.module)
        self.generic_visit(node)

lines = {'openpilot': 0, 'car': 0, 'tools': 0, 'tests': 0}

for f in sorted(pyf):
    if f not in fouts:
        continue
    if not os.path.isfile(f):
        continue
    with open(f) as file:
        src = file.read()
    lns = len(src.split("\n"))
    ast.walk(ast.parse(src), Analyzer())
    filename, file_extension = os.path.splitext(f)
    if 'test' in os.path.basename(filename):
        lines['tests'] += lns
    elif f.startswith('tools/') or f.startswith('scripts/') or f.startswith('selfdrive/debug'):
        lines['tools'] += lns
    elif f.startswith('selfdrive/car'):
        lines['car'] += lns
    else:
        lines['openpilot'] += lns

print("%d lines of openpilot python" % lines['openpilot'])
print("%d lines of car ports" % lines['car'])
print("%d lines of tools/scripts/debug" % lines['tools'])
print("%d lines of tests" % lines['tests'])
#print(sorted(list(imps)))
