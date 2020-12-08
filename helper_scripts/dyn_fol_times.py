import time
import ast
import matplotlib.pyplot as plt
import numpy as np

with open('C:/Git/dyn_fol_times2', 'r') as f:
  data = f.read().split('\n')
times = []

for line in data:
  try:
    if line == '':
      continue
    line = '{' + line.replace('one_it', '"one_it"').replace('mpc_id', '"mpc_id"') + '}'
    # print(line)
    times.append(ast.literal_eval(line))
  except:
    pass
