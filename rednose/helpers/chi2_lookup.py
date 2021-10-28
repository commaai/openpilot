import os

import numpy as np


def gen_chi2_ppf_lookup(max_dim=200):
  from scipy.stats import chi2
  table = np.zeros((max_dim, 98))
  for dim in range(1, max_dim):
    table[dim] = chi2.ppf(np.arange(.01, .99, .01), dim)

  np.save('chi2_lookup_table', table)


def chi2_ppf(p, dim):
  table = np.load(os.path.dirname(os.path.realpath(__file__)) + '/chi2_lookup_table.npy')
  result = np.interp(p, np.arange(.01, .99, .01), table[dim])
  return result


if __name__ == "__main__":
  gen_chi2_ppf_lookup()
