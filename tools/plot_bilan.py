#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this must come first
from __future__ import print_function, unicode_literals, division

# standard library imports
from argparse import ArgumentParser
import os
import os.path
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


########################################
def string_to_percent(x):
  """
  """
  return float(x.strip("%"))/100.


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
class BilanDict(dict):
  #---------------------------------------
  def __init__(self):
    self = {}

  #---------------------------------------
  def init_range(self, date_beg, date_end, inc=1):
    """
    """
    delta = date_end - date_beg

    (deb, fin) = (0, delta.days+1)

    dates = (date_beg + dt.timedelta(days=i)
             for i in xrange(deb, fin, inc))

    for date in dates:
      self.add_item(date)

  #---------------------------------------
  def fill_data(self, filein):
    data = np.genfromtxt(
      filein,
      skip_header=1,
      converters={0: string_to_date,
                  1: string_to_float,
                  2: string_to_percent,
                  3: string_to_percent},
      missing_values="nan",
    )

    for date, conso, real_use, theo_use in data:
      if date in self:
        self.add_item(date, conso, real_use, theo_use)
        self[date].fill()

  #---------------------------------------
  def add_item(self, date, conso=np.nan,
               real_use=np.nan, theo_use=np.nan):
    """
    """
    self[date] = Conso(date, conso, real_use, theo_use)

  #---------------------------------------
  def theo_equation(self):
    """
    """
    (dates, theo_uses) = \
      zip(*((item.date, item.theo_use)
            for item in self.get_items_in_full_range()))

    (idx_min, idx_max) = \
        (np.nanargmin(theo_uses), np.nanargmax(theo_uses))

    x1 = dates[idx_min].timetuple().tm_yday
    x2 = dates[idx_max].timetuple().tm_yday

    y1 = theo_uses[idx_min]
    y2 = theo_uses[idx_max]

    m = np.array([
      [x1, 1.],
      [x2, 1.]
    ], dtype="float")
    n = np.array([
      y1,
      y2
    ], dtype="float")

    try:
      (a, b) = np.linalg.solve(m, n)
    except np.linalg.linalg.LinAlgError:
      (a, b) = (None, None)

    if a and b:
      for date in dates:
        self[date].theo_equ = date.timetuple().tm_yday*a + b

  #---------------------------------------
  def get_items_in_range(self, date_beg, date_end, inc=1):
    """
    """
    items = (item for item in self.itervalues()
                   if item.date >= date_beg and
                      item.date <= date_end)
    items = sorted(items, key=lambda item: item.date)

    return items[::inc]

  #---------------------------------------
  def get_items_in_full_range(self, inc=1):
    """
    """
    items = (item for item in self.itervalues())
    items = sorted(items, key=lambda item: item.date)

    return items[::inc]

  #---------------------------------------
  def get_items(self, inc=1):
    """
    """
    items = (item for item in self.itervalues()
                   if item.isfilled())
    items = sorted(items, key=lambda item: item.date)

    return items[::inc]


class Conso(object):
  #---------------------------------------
  def __init__(self, date, conso=np.nan,
               real_use=np.nan, theo_use=np.nan):
    self.date     = date
    self.conso    = conso
    self.real_use = real_use
    self.theo_use = theo_use
    self.theo_equ = np.nan
    self.filled   = False

  #---------------------------------------
  def __repr__(self):
    return "{:.2f} ({:.2%})".format(self.conso, self.real_use)

  #---------------------------------------
  def isfilled(self):
    return self.filled

  #---------------------------------------
  def fill(self):
    self.filled = True


########################################
def plot_init():
  paper_size  = np.array([29.7, 21.0])
  fig, ax_conso = plt.subplots(figsize=(paper_size/2.54))
  ax_theo = ax_conso.twinx()

  return fig, ax_conso, ax_theo


########################################
def plot_data(ax_conso, ax_theo, xcoord, xlabels,
              consos, theo_uses, real_uses, theo_equs):
  """
  """
  ax_conso.bar(xcoord, consos, align="center", color="linen",
               linewidth=0.2, label="conso (heures)")

  ax_theo.plot(xcoord, theo_equs, "--",
               color="firebrick", linewidth=0.5,
               solid_capstyle="round", solid_joinstyle="round")
  ax_theo.plot(xcoord, theo_uses, "+-", color="firebrick",
               linewidth=1, markersize=8,
               # solid_capstyle="round", solid_joinstyle="round",
               label="conso théorique (%)")
  ax_theo.plot(xcoord, real_uses, "+-", color="forestgreen",
               linewidth=1, markersize=8,
               # solid_capstyle="round", solid_joinstyle="round",
               label="conso réelle (%)")


########################################
def plot_config(ax_conso, ax_theo, xcoord, xlabels, ymax, title):
  """
  """
  # ... Config axes ...
  # -------------------
  # 1) Range
  xmin, xmax = xcoord[0]-1, xcoord[-1]+1
  ax_conso.set_xlim(xmin, xmax)
  ax_conso.set_ylim(0., ymax)
  ax_theo.set_ylim(0., 100)

  # 2) Ticks labels
  inc_label = 7 if nb_items > 37 else 1
  ax_conso.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
  ax_conso.set_xticks(xcoord, minor=True)
  ax_conso.set_xticks(xcoord[::inc_label], minor=False)
  ax_conso.set_xticklabels(
    xlabels[::inc_label], rotation="45", size="x-small"
  )

  # 3) Define axes title
  ax_conso.set_ylabel("heures", fontweight="bold")
  ax_theo.set_ylabel("%", fontweight="bold")

  # ... Main title and legend ...
  # -----------------------------
  ax_conso.set_title(title, fontweight="bold", size="large")
  ax_theo.legend(loc="upper right", fontsize="x-small", frameon=False)
  ax_conso.legend(loc="upper left", fontsize="x-small", frameon=False)


########################################
def plot_save(img_name):
  """
  """
  dpi = 200.

  with PdfPages(img_name) as pdf:
    pdf.savefig(dpi=dpi)

    # pdf file's metadata
    d = pdf.infodict()
    d["Title"]   = "Conso GENCMIP6"
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
                      help="plot the whole period")
  parser.add_argument("-i", "--increment", action="store",
                      type=int, default=1, dest="inc",
                      help="sampling increment")
  parser.add_argument("-r", "--range", action="store", nargs=2,
                      type=string_to_date,
                      help="date range: ssaa-mm-jj ssaa-mm-jj")
  parser.add_argument("-m", "--max", action="store_true",
                      help="plot with y_max = allocation")

  return parser.parse_args()


########################################
if __name__ == '__main__':

  # .. Initialization ..
  # ====================
  # ... Command line arguments ...
  # ------------------------------
  args = get_arguments()

  # ... Files and directories ...
  # -----------------------------
  dir_data = os.path.join("..", "output")
  file_pattern = "OUT_CONSO_"
  file_param = get_last_file(dir_data, file_pattern+"PARAM")
  file_utheo = get_last_file(dir_data, file_pattern+"UTHEO")
  file_bilan = get_last_file(dir_data, file_pattern+"BILAN")
  file_login = get_last_file(dir_data, file_pattern+"LOGIN")
  file_store = get_last_file(dir_data, file_pattern+"STORE")

  if args.verbose:
    print(file_param)
    print(file_utheo)
    print(file_bilan)
    print(file_login)
    print(file_store)

  # .. Get project info ..
  # ======================
  gencmip6 = Project()
  gencmip6.fill_data(file_param)
  gencmip6.get_date_init(file_utheo)

  # .. Fill in conso data ..
  # ========================
  # ... Initialization ...
  # ----------------------
  bilan = BilanDict()
  bilan.init_range(gencmip6.date_init, gencmip6.deadline)
  # ... Extract data from file ...
  # ------------------------------
  bilan.fill_data(file_bilan)
  # ... Compute theoratical use from known data  ...
  # ------------------------------------------------
  bilan.theo_equation()

  # .. Extract data depending on C.L. arguments ..
  # ==============================================
  # args.range = [
  #   string_to_date("2015-01-01"),
  #   string_to_date("2015-01-31")
  # ]
  # args.full  = True
  if args.full:
    selected_items = bilan.get_items_in_full_range(args.inc)
  elif args.range:
    selected_items = bilan.get_items_in_range(
      args.range[0], args.range[1], args.inc
    )
  else:
    selected_items = bilan.get_items(args.inc)

  # .. Compute data to be plotted ..
  # ================================
  nb_items = len(selected_items)

  xcoord    = np.linspace(1, nb_items, num=nb_items)
  xlabels   = ["{:%d-%m}".format(item.date)
               for item in selected_items]
  consos    = np.array([item.conso for item in selected_items],
                        dtype=float)
  theo_uses = np.array([100.*item.theo_use for item in selected_items],
                       dtype=float)
  real_uses = np.array([100.*item.real_use for item in selected_items],
                       dtype=float)
  theo_equs = np.array([100.*item.theo_equ for item in selected_items],
                       dtype=float)

  # .. Plot stuff ..
  # ================
  # ... Initialize figure ...
  # -------------------------
  (fig, ax_conso, ax_theo) = plot_init()

  # ... Plot data ...
  # -----------------
  plot_data(ax_conso, ax_theo, xcoord, xlabels,
            consos, theo_uses, real_uses, theo_equs)

  # ... Tweak figure ...
  # --------------------
  if args.max:
    ymax = gencmip6.alloc
  else:
    ymax = np.nanmax(consos) + np.nanmax(consos)*.1

  title = "Consommation {}\n({:%d/%m/%Y} - {:%d/%m/%Y})".format(
    gencmip6.project.upper(),
    gencmip6.date_init,
    gencmip6.deadline
  )

  plot_config(ax_conso, ax_theo, xcoord, xlabels, ymax, title)

  # ... Save figure ...
  # -------------------
  dirout = "img"
  img_name = "bilan.pdf"
  plot_save(os.path.join(dirout, img_name))

  plt.show()
