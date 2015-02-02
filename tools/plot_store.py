#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this must come first
from __future__ import print_function, unicode_literals, division

# standard library imports
from argparse import ArgumentParser
import os
import os.path
# import glob
# import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gencmip6 import *


# ########################################
# def string_to_percent(x):
#   """
#   """
#   return float(x.strip("%"))/100.


# ########################################
# def string_to_size_unit(x):
#   """
#   """
#   (size, unit) = (float(x[:-1]), x[-1])
#   return SizeUnit(size, unit)


# ########################################
# def string_to_float(x):
#   """
#   """
#   return float(x.strip("h"))


# ########################################
# def string_to_date(ssaammjj, fmt="%Y-%m-%d"):
#   """
#   """
#   return dt.datetime.strptime(ssaammjj, fmt)


# ########################################
# def date_to_string(dtdate, fmt="%Y-%m-%d"):
#   """
#   """
#   return dt.datetime.strftime(dtdate, fmt)


# ########################################
# def get_last_file(dir_data, pattern):
#   """
#   """
#   current_dir = os.getcwd()
#   os.chdir(dir_data)
#   filename = pattern + "*"
#   return_value = sorted(glob.glob(os.path.join(dir_data, filename)))[-1]
#   os.chdir(current_dir)
#   return return_value


# ########################################
# class Project(object):

#   #---------------------------------------
#   def __init__(self):
#     self.project   = ""
#     self.date_init = ""
#     self.deadline  = ""
#     self.alloc     = 0

#   #---------------------------------------
#   def fill_data(self, filein):
#     import json
#     dico = json.load(open(filein, "r"))
#     self.project = dico["project"]
#     self.deadline = string_to_date(dico["deadline"]) + \
#                     dt.timedelta(days=-1)
#     self.alloc = dico["alloc"]

#   #---------------------------------------
#   def get_date_init(self, filein):
#     data = np.genfromtxt(
#       filein,
#       skip_header=1,
#       converters={0: string_to_date,
#                   1: string_to_percent},
#       missing_values="nan",
#     )
#     dates, utheos = zip(*data)

#     (x1, x2) = (np.nanargmin(utheos), np.nanargmax(utheos))

#     m = np.array([[x1, 1.], [x2, 1.]])
#     n = np.array([utheos[x1], utheos[x2]])

#     (a, b) = np.linalg.solve(m, n)

#     delta = int(round((-b/a)-x1 + 1))

#     d1 = dates[x1]
#     self.date_init = d1 + dt.timedelta(days=delta)


# ########################################
# class SizeUnit(object):
#   #---------------------------------------
#   def __init__(self, size, unit):
#     self.size = size
#     self.unit = unit

#   #---------------------------------------
#   def __repr__(self):
#     return "{:6.2f}{}o".format(self.size, self.unit)

#   #---------------------------------------
#   def convert_size(self, unit_out):
#     """
#     """
#     prefixes = ["K", "M", "G", "T", "P", "H"]

#     if not self.size or \
#        self.unit == unit_out:
#       size_out = self.size
#     else:
#       idx_deb = prefixes.index(self.unit)
#       idx_fin = prefixes.index(unit_out)
#       size_out = self.size
#       for i in xrange(abs(idx_fin-idx_deb)):
#         if idx_fin > idx_deb:
#           size_out = size_out / 1024
#         else:
#           size_out = size_out * 1024

#     return SizeUnit(size_out, unit_out)


########################################
class DirVolume(object):
  #---------------------------------------
  def __init__(self, date, login, dirname, size):
    self.date  = date
    self.login = login
    self.dirname = dirname
    self.dirsize = size

  #---------------------------------------
  def __repr__(self):
    return "{}={}".format(self.dirname, self.dirsize)


########################################
class StoreDict(dict):
  #---------------------------------------
  def __init__(self):
    self = {}

  #---------------------------------------
  def fill_data(self, filein):
    data = np.genfromtxt(
      filein,
      skip_header=1,
      converters={0: string_to_date,
                  1: str,
                  2: string_to_size_unit,
                  3: str},
      missing_values="nan",
    )

    for date, login, dirsize, dirname in data:
      self.add_item(date, login, dirsize, dirname)

  #---------------------------------------
  def add_item(self, date, login, dirsize, dirname):
    """
    """
    if login not in self:
      self[login] = Login(date, login)
    self[login].add_directory(date, login, dirsize, dirname)

  #---------------------------------------
  def get_items(self):
    """
    """
    items = (subitem for item in self.itervalues()
                     for subitem in item.listdir)
    items = sorted(items, key=lambda item: item.login)

    return items

  #---------------------------------------
  def get_items_by_name(self, pattern):
    """
    """
    # items = (item for item in self.itervalues() if item.dir)
    items = (subitem for item in self.itervalues()
                     for subitem in item.listdir
                      if pattern in subitem.dirname)
    items = sorted(items, key=lambda item: item.login)

    return items


########################################
class Login(object):
  #---------------------------------------
  def __init__(self, date, login):
    self.date  = date
    self.login = login
    self.total = SizeUnit(0., "K")
    self.listdir = []

  #---------------------------------------
  def __repr__(self):
    return "{}/{:%F}: {}".format(self.login, self.date, self.listdir)

  #---------------------------------------
  def add_to_total(self, dirsize):
    """
    """
    somme = self.total.convert_size("K").size + \
            dirsize.convert_size("K").size
    self.total = SizeUnit(somme, "K")

  #---------------------------------------
  def add_directory(self, date, login, dirsize, dirname):
    """
    """
    self.listdir.append(DirVolume(date, login, dirname, dirsize))
    self.add_to_total(dirsize)


########################################
def plot_init():
  paper_size  = np.array([29.7, 21.0])
  fig, ax = plt.subplots(figsize=(paper_size/2.54))

  return fig, ax


########################################
def plot_data(ax, coords, ylabels, values):
  """
  """
  ax.barh(coords, values, align="center", color="linen",
          linewidth=0.2, label="volume sur STORE ($To$)")


########################################
def plot_config(ax, coords, ylabels, dirnames, title, tot_volume):
  """
  """
  # ... Config axes ...
  # -------------------
  # 1) Range
  ymin, ymax = coords[0]-1, coords[-1]+1
  ax.set_ylim(ymin, ymax)

  # 2) Ticks labels
  ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  ax.set_yticks(coords, minor=False)
  ax.set_yticklabels(ylabels, size="x-small", fontweight="bold")
  ax.invert_yaxis()

  xmin, xmax = ax.get_xlim()
  xpos = xmin + (xmax-xmin)/50.
  for (ypos, text) in zip(coords, dirnames):
    ax.text(s=text, x=xpos, y=ypos, va="center", ha="left",
                size="xx-small", color="gray", style="italic")

  # 3) Define axes title
  ax.set_xlabel("$To$", fontweight="bold")

  # ... Main title and legend ...
  # -----------------------------
  ax.set_title(title, fontweight="bold", size="large")
  ax.legend(loc="best", fontsize="x-small", frameon=False)

  tot_label = "volume total = {}".format(tot_volume)
  plt.figtext(x=0.95, y=0.93, s=tot_label, backgroundcolor="linen",
              ha="right", va="bottom", fontsize="small")


########################################
def plot_save(img_name):
  """
  """
  dpi = 200.

  with PdfPages(img_name) as pdf:
    pdf.savefig(dpi=dpi)

    # pdf file's metadata
    d = pdf.infodict()
    d["Title"]   = "Occupation GENCMIP6 sur STORE par login"
    d["Author"]  = "plot_bilan.py"
    # d["Subject"] = "Time spent over specific commands during create_ts \
    #                 jobs at IDRIS and four configurations at TGCC"
    # d["Keywords"] = "bench create_ts TGCC IDRIS ncrcat"
    # d["CreationDate"] = dt.datetime(2009, 11, 13)
    # d["ModDate"] = dt.datetime.today()


########################################
def get_arguments():
  parser = ArgumentParser()
  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Verbose mode")
  parser.add_argument("-f", "--full", action="store_true",
                      help="plot all the directories in IGCM_OUT" +
                           "(default: plot IPSLCM6 directories)")
  parser.add_argument("-p", "--pattern", action="store",
                      default="IPSLCM6",
                      help="plot the whole period")

  return parser.parse_args()


########################################
if __name__ == '__main__':

  # .. Initialization ..
  # ====================
  # ... Command line arguments ...
  # ------------------------------
  args = get_arguments()
  if args.verbose:
    print(args)

  # ... Files and directories ...
  # -----------------------------
  dir_data = os.path.join("..", "output")
  file_pattern = "OUT_CONSO_"
  file_param = get_last_file(dir_data, file_pattern+"PARAM")
  file_utheo = get_last_file(dir_data, file_pattern+"UTHEO")
  file_bilan = get_last_file(dir_data, file_pattern+"BILAN")
  file_login = get_last_file(dir_data, file_pattern+"LOGIN")
  file_store = get_last_file(dir_data, file_pattern+"STORE")

  # .. Get project info ..
  # ======================
  gencmip6 = Project()
  gencmip6.fill_data(file_param)
  gencmip6.get_date_init(file_utheo)

  # .. Fill in data dict ..
  # =======================
  stores = StoreDict()
  stores.fill_data(file_store)

  # .. Extract data depending on C.L. arguments ..
  # ==============================================
  if args.full:
    selected_items = stores.get_items()
  else:
    selected_items = stores.get_items_by_name(args.pattern)

  if args.verbose:
    for item in selected_items:
      print(
        "{:8s} {:%F} {} {:>18s} {} ".format(
          item.login,
          item.date,
          item.dirsize,
          item.dirsize.convert_size("K"),
          item.dirname,
        )
      )

  # .. Compute data to be plotted ..
  # ================================
  ylabels = [item.login for item in selected_items]
  values  = np.array([item.dirsize.convert_size("T").size
                          for item in selected_items],
                     dtype=float)
  dirnames = [item.dirname for item in selected_items]
  date = selected_items[0].date

  nb_items = len(ylabels)
  coords  = np.linspace(1, nb_items, num=nb_items)

  # .. Plot stuff ..
  # ================
  # ... Initialize figure ...
  # -------------------------
  (fig, ax) = plot_init()

  # ... Plot data ...
  # -----------------
  plot_data(ax, coords, ylabels, values)

  # ... Tweak figure ...
  # --------------------
  title = "Occupation {} de STORE par login\n{:%d/%m/%Y}".format(
    gencmip6.project.upper(),
    date
  )
  plot_config(ax, coords, ylabels, dirnames, title,
              SizeUnit(np.sum(values), "T"))

  # ... Save figure ...
  # -------------------
  dirout = "img"
  img_name = "store.pdf"
  plot_save(os.path.join(dirout, img_name))

  plt.show()
  exit()
