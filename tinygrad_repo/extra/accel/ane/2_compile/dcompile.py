#!/usr/bin/env python3
import os
import sys
import networkx as nx
import pylab as plt
from networkx.drawing.nx_pydot import read_dot

ret = os.system("./a.out "+sys.argv[1]+" debug")
assert(ret == 0)

df = "debug/model.hwx.zinir_graph_after_reg_spill.dot"

#from graphviz import render
#render('dot', 'png', df)

#plt = Image(pdot.create_png()
#display(plt)
