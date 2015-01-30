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
class LoginDict(dict):
  #---------------------------------------
  def __init__(self):
    self = {}

  #---------------------------------------
  def fill_data(self, filein):
    data = np.genfromtxt(
      filein,
      skip_header=1,
      converters={0: string_to_date,
                  1: str},
      missing_values="nan",
    )

    for date, login, conso in data:
      self.add_item(date, login, conso)

  #---------------------------------------
  def add_item(self, date, login, conso):
    """
    """
    self[login] = Login(date, login, conso)

  # #---------------------------------------
  # def get_items_in_full_range(self, inc=1):
  #   """
  #   """
  #   items = (item for item in self.itervalues())
  #   items = sorted(items, key=lambda item: item.date)

  #   return items[::inc]

  #---------------------------------------
  def get_items(self):
    """
    """
    items = (item for item in self.itervalues())
    items = sorted(items, key=lambda item: item.login)

    return items

  #---------------------------------------
  def get_items_not_null(self):
    """
    """
    items = (item for item in self.itervalues()
                   if item.conso > 0.)
    items = sorted(items, key=lambda item: item.login)

    return items


class Login(object):
  #---------------------------------------
  def __init__(self, date, login, conso):
    self.date  = date
    self.login = login
    self.conso = conso

  #---------------------------------------
  def __repr__(self):
    return "{} ({:.2}h)".format(self.login, self.conso)

  # #---------------------------------------
  # def isfilled(self):
  #   return self.filled

  # #---------------------------------------
  # def fill(self):
  #   self.filled = True


########################################
def plot_init():
  paper_size  = np.array([29.7, 21.0])
  fig, ax = plt.subplots(figsize=(paper_size/2.54))

  return fig, ax


########################################
def plot_data(ax, ycoord, ylabels, consos):
  """
  """
  print(ycoord)
  print(consos)

  ax.barh(ycoord, consos, align="center", color="linen",
          linewidth=0.2, label="conso (heures)")


########################################
def plot_config(ax, ycoord, ylabels, title):
  """
  """
  # ... Config axes ...
  # -------------------
  # 1) Range
  ymin, ymax = ycoord[0]-1, ycoord[-1]+1
  ax.set_ylim(ymin, ymax)

  # 2) Ticks labels
  ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  ax.set_yticks(ycoord, minor=False)
  ax.set_yticklabels(ylabels, size="x-small", fontweight="bold")

  # 3) Define axes title
  ax.set_xlabel("heures", fontweight="bold")

  # ... Main title and legend ...
  # -----------------------------
  ax.set_title(title, fontweight="bold", size="large")
  ax.legend(loc="best", fontsize="x-small", frameon=False)


########################################
def plot_save(img_name):
  """
  """
  dpi = 200.

  with PdfPages(img_name) as pdf:
    pdf.savefig(dpi=dpi)

    # pdf file's metadata
    d = pdf.infodict()
    d["Title"]   = "Conso GENCMIP6 par login"
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
                      help="plot all the logins" +
                           " (default: plot only non-zero)")

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

  # .. Get project info ..
  # ======================
  gencmip6 = Project()
  gencmip6.fill_data(file_param)
  gencmip6.get_date_init(file_utheo)

  # .. Fill in data dict ..
  # =======================
  # ... Initialization ...
  # ----------------------
  logins = LoginDict()
  logins.fill_data(file_login)

  # .. Extract data depending on C.L. arguments ..
  # ==============================================
  if args.full:
    selected_items = logins.get_items()
  else:
    selected_items = logins.get_items_not_null()

  if args.verbose:
    for login in selected_items:
      print(login)

  # .. Compute data to be plotted ..
  # ================================
  nb_items = len(selected_items)

  ycoord  = np.linspace(1, nb_items, num=nb_items)
  ylabels = [item.login for item in selected_items]
  consos  = np.array([item.conso for item in selected_items],
                      dtype=float)
  date = selected_items[0].date

  # .. Plot stuff ..
  # ================
  # ... Initialize figure ...
  # -------------------------
  (fig, ax) = plot_init()

  # ... Plot data ...
  # -----------------
  plot_data(ax, ycoord, ylabels, consos)

  # # ... Tweak figure ...
  # # --------------------
  title = "Consommation {} par login\n{:%d/%m/%Y}".format(
    gencmip6.project.upper(),
    date
  )
  plot_config(ax, ycoord, ylabels, title)

  # ... Save figure ...
  # -------------------
  dirout = "img"
  img_name = "login.pdf"
  plot_save(os.path.join(dirout, img_name))

  plt.show()
  exit()
