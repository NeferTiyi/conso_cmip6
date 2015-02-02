#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this must come first
from __future__ import print_function, unicode_literals, division

# standard library imports
# from argparse import ArgumentParser
import os
import os.path
import glob
import datetime as dt
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


########################################
def string_to_percent(x):
  """
  """
  return float(x.strip("%"))/100.


########################################
def string_to_size_unit(x):
  """
  """
  (size, unit) = (float(x[:-1]), x[-1])
  return SizeUnit(size, unit)


########################################
def string_to_float(x):
  """
  """
  return float(x.strip("h"))


########################################
def string_to_date(ssaammjj, fmt="%Y-%m-%d"):
  """
  """
  return dt.datetime.strptime(ssaammjj, fmt)


# ########################################
# def date_to_string(dtdate, fmt="%Y-%m-%d"):
#   """
#   """
#   return dt.datetime.strftime(dtdate, fmt)


########################################
def get_last_file(dir_data, pattern):
  """
  """
  current_dir = os.getcwd()
  os.chdir(dir_data)
  filename = pattern + "*"
  return_value = sorted(glob.glob(os.path.join(dir_data, filename)))[-1]
  os.chdir(current_dir)
  return return_value


########################################
class Project(object):

  #---------------------------------------
  def __init__(self):
    self.project   = ""
    self.date_init = ""
    self.deadline  = ""
    self.alloc     = 0

  #---------------------------------------
  def fill_data(self, filein):
    import json
    dico = json.load(open(filein, "r"))
    self.project = dico["project"]
    self.deadline = string_to_date(dico["deadline"]) + \
                    dt.timedelta(days=-1)
    self.alloc = dico["alloc"]

  #---------------------------------------
  def get_date_init(self, filein):
    data = np.genfromtxt(
      filein,
      skip_header=1,
      converters={0: string_to_date,
                  1: string_to_percent},
      missing_values="nan",
    )
    dates, utheos = zip(*data)

    (x1, x2) = (np.nanargmin(utheos), np.nanargmax(utheos))

    m = np.array([[x1, 1.], [x2, 1.]])
    n = np.array([utheos[x1], utheos[x2]])

    (a, b) = np.linalg.solve(m, n)

    delta = int(round((-b/a)-x1 + 1))

    d1 = dates[x1]
    self.date_init = d1 + dt.timedelta(days=delta)


########################################
class SizeUnit(object):
  #---------------------------------------
  def __init__(self, size, unit):
    self.size = size
    self.unit = unit

  #---------------------------------------
  def __repr__(self):
    return "{:6.2f}{}o".format(self.size, self.unit)

  #---------------------------------------
  def convert_size(self, unit_out):
    """
    """
    prefixes = ["K", "M", "G", "T", "P", "H"]

    if not self.size or \
       self.unit == unit_out:
      size_out = self.size
    else:
      idx_deb = prefixes.index(self.unit)
      idx_fin = prefixes.index(unit_out)
      size_out = self.size
      for i in xrange(abs(idx_fin-idx_deb)):
        if idx_fin > idx_deb:
          size_out = size_out / 1024
        else:
          size_out = size_out * 1024

    return SizeUnit(size_out, unit_out)


########################################
if __name__ == '__main__':
  pass
