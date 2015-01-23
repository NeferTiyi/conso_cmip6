#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this must come first
from __future__ import print_function, unicode_literals, division

# standard library imports
from argparse import ArgumentParser
import os
import os.path
import glob
# import ConfigParser as cp
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


########################################
def string_to_date(ssaammjj):
  """
  """
  ssaa, mm, jj = (int(i) for i in ssaammjj.split("-"))
  return dt.date(ssaa, mm, jj)


########################################
def date_to_string(dtdate):
  """
  """
  # return "-".join([dtdate.year, dtdate.month, dtdate.day])
  return "{:04d}-{:02d}-{:02d}".\
         format(dtdate.year, dtdate.month, dtdate.day)


########################################
def get_last_file(pattern):
  """
  """
  current_dir = os.getcwd()
  os.chdir(dir_data)
  filename = file_pattern + pattern + "*"
  return_value = glob.glob(os.path.join(dir_data, filename))[-1]
  os.chdir(current_dir)
  return return_value


########################################
def parse_param(filename):
  """
  """
  param = Project()
  with open(filename, "r") as filein:
    for ligne in filein:
      if ligne.split(":"):
        clef, val = ligne.strip().split(":")
        if clef == "project":
          param.project = val.strip()
        elif clef == "date_beg":
          param.date_beg = val.strip()
        elif clef == "date_end":
          param.date_end = val.strip()
        elif clef == "alloc":
          param.alloc = float(val)
  return param


########################################
def parse_store(filename):
  """
  """
  # conso_dict = ConsoDict()
  with open(filename, "r") as filein:
    for ligne in filein:
      print(ligne.strip().split())
  #     if ligne.split():
  #       conso_dict.add_daily_conso(
  #         ligne.split()[0],
  #         ligne.split()[2],
  #         ligne.split()[5],
  #         ligne.split()[8],
  #         ligne.split()[10]
  #       )
  # return conso_dict


########################################
class Project(dict):

#---------------------------------------
  def __init__(self):
    self.project = ""
    self.date_beg = ""
    self.date_end = ""
    self.alloc = 0


########################################
class StoreLogin(object):

#---------------------------------------
  def __init__(self, date, login, size, unit, dirpath):

    self.date    = date
    self.login   = login
    self.size    = size
    self.unit    = unit
    self.dirpath = dirpath


#---------------------------------------
  def __repr__(self):
    return "<{date}: {total}h>".\
           format(date=self.date, total=self.total)


########################################
if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Verbose mode")
  parser.add_argument("-f", "--full", action="store_true",
                      help="plot the whole period")
  parser.add_argument("-i", "--increment", action="store",
                      type=int, default=1, dest="inc",
                      help="sampling increment")
  parser.add_argument("-r", "--range", action="store", nargs=2,
                      help="date range: ssaa-mm-jj ssaa-mm-jj")
  parser.add_argument("-m", "--max", action="store_true",
                      help="plot with y_max = allocation")

  args = parser.parse_args()

  # ... Initialization ...
  # ----------------------
  # Files and directories
  dir_data = os.path.join("..", "output")
  file_pattern = "OUT_CONSO_"

  file_param = get_last_file("PARAM")
  file_store = get_last_file("STORE")

  # Parse files
  param = parse_param(file_param)
  store = parse_store(file_store)



  exit()

  # ... Plot data ...
  # -----------------
  # fig = plt.figure(figsize=(paper_size*scale))

  fig, ax_conso = plt.subplots()
  ax_theo = ax_conso.twinx()

  inc_label = 1
  if args.full:
    clefs = conso_dict.get_period_range("2015-01-01", "2015-06-30", args.inc)
    inc_label = 7
  elif args.range:
    clefs = conso_dict.get_period_range(args.range[0], args.range[1], args.inc)
  else:
    clefs = conso_dict.get_period_from_keys()

  nb_days = len(clefs)
  if nb_days > 31:
    inc_label = 7

  abscisses = np.linspace(1, nb_days, num=nb_days)

  conso = []
  for clef in clefs:
    if clef in conso_dict:
      conso.append(conso_dict[clef].total)
    else:
      conso.append(0.)
  conso = np.array(conso, dtype=float)

  theo = []
  for clef in clefs:
    if clef in conso_dict:
      theo.append(conso_dict[clef].use_theo)
    else:
      theo.append(np.nan)
  theo = np.array(theo, dtype=float)

  real = []
  for clef in clefs:
    if clef in conso_dict:
      real.append(conso_dict[clef].use_real)
    else:
      real.append(np.nan)
  real = np.array(real, dtype=float)

  date_label = ["-".join(clef.split("-")[1:]) for clef in clefs]

  (a, b) = get_theo_equation(abscisses, theo)
  theo_xval = np.array([0, nb_days+1])
  theo_yval = a*theo_xval + b
  # theo_yval = [a*x + b for x in theo_xval]

  ax_conso.bar(abscisses, conso, align="center", color="linen",
               linewidth=0.2, label="conso (heures)")

  # if args.full:
  ax_theo.plot(theo_xval, theo_yval, "--",
               color="firebrick", linewidth=0.5,
               solid_capstyle="round", solid_joinstyle="round")
  ax_theo.plot(abscisses, theo, "+-", color="firebrick",
               linewidth=1, markersize=8,
               solid_capstyle="round", solid_joinstyle="round",
               label="conso théorique (%)")
  ax_theo.plot(abscisses, real, "+-", color="forestgreen",
               linewidth=1, markersize=8,
               solid_capstyle="round", solid_joinstyle="round",
               label="conso réelle (%)")

  # ... Define axes ...
  # -------------------
  # 1) Range
  xmin, xmax = min(abscisses) - 1, nb_days + 1
  ax_conso.set_xlim(xmin, xmax)
  if args.max:
    # ymax = conso_dict.get_last_alloc()
    ymax = param.alloc
  else:
    ymax = max(conso) + max(conso)*.1
  ax_conso.set_ylim(0., ymax)
  ax_theo.set_ylim(0., 100)

  # 2) Ticks labels
  ax_conso.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
  ax_conso.set_xticks(abscisses, minor=True)
  ax_conso.set_xticks(abscisses[::inc_label], minor=False)
  ax_conso.set_xticklabels(
    date_label[::inc_label], rotation="45", size="x-small"
  )

  # 3) Define axes title
  ax_conso.set_ylabel("heures", fontweight="bold")
  ax_theo.set_ylabel("%", fontweight="bold")

  # ... Main title and legend ...
  # -----------------------------
  title = "Consommation {project}\n({beg}/{end})".format(
    project=param.project.upper(),
    beg=param.date_beg,
    end=param.date_end
  )
  ax_conso.set_title(title, fontweight="bold", size="large")
  ax_theo.legend(loc="best", fontsize="x-small", frameon=False)
  ax_conso.legend(loc="upper left", fontsize="x-small", frameon=False)

  # Be sure that everything is visible
  # plt.tight_layout()

  fig.savefig("conso.png", dpi=200)
  # fig.show()

  exit()
