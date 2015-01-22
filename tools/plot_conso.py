#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this must come first
from __future__ import print_function, unicode_literals, division

# standard library imports
from argparse import ArgumentParser
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
def get_theo_equation(xval, yval):
  """
  """
  # Examples
  # --------
  # Solve the system of equations
  # ``3*x0 + 1*x1 = 9`` and
  # ``1*x0 + 2*x1 = 8``:

  # >>> a = np.array([[3,1], [1,2]])
  # >>> b = np.array([9,8])
  # >>> x = np.linalg.solve(a, b)
  # >>> x
  # array([ 2.,  3.])

  # Check that the solution is correct:

  # >>> np.allclose(np.dot(a, x), b)
  # True

  a = np.array([[xval[np.nanargmin(yval)], 1.],
                [xval[np.nanargmax(yval)], 1.]])
  b = np.array([np.nanmin(yval), np.nanmax(yval)])

  return np.linalg.solve(a, b)


########################################
class DailyConso(object):

#---------------------------------------
  def __init__(self, date, total, alloc, use_theo, use_real):

    miss_val = "-99.99%"

    # ssaa, mm, jj = date.split("-")
    # self.date = "".join((ssaa, mm, jj))
    # self.mmjj = "-".join((mm, jj))
    self.total = float(total)
    self.alloc = float(alloc)

    if use_theo == miss_val:
      # self.use_theo = None
      self.use_theo = np.nan
    else:
      self.use_theo = float(use_theo[0:-1])

    if use_real == miss_val:
      # self.use_real = None
      self.use_real = np.nan
    else:
      self.use_real = float(use_real[0:-1])

#---------------------------------------
  def __repr__(self):
    return "<{date}: {total}h>".\
           format(date=self.date, total=self.total)


########################################
class ConsoDict(dict):

#---------------------------------------
  def __init__(self):
    self = {}

# #---------------------------------------
#   def __repr__(self):
#     return_value = ""
#     for key in sorted(self):
#       return_value = return_value + key + ": " + "\n"

#     return return_value

#---------------------------------------
  def add_daily_conso(self, date, total, alloc, use_theo, use_real):
    """
    """
    self[date] = DailyConso(date, total, alloc, use_theo, use_real)

#---------------------------------------
  def get_period_range(self, date_beg, date_end, inc):
    """
    """
    d1 = string_to_date(date_beg)
    d2 = string_to_date(date_end)
    delta = d2 - d1

    (deb, fin) = (0, delta.days+1)

    return [date_to_string(d1 + dt.timedelta(days=i))
            for i in xrange(deb, fin, inc)]

#---------------------------------------
  def get_period_from_keys(self):
    """
    """
    return sorted(self)
    # return [self[key].mmjj for key in sorted(self)]

#---------------------------------------
  def get_last_alloc(self):
    """
    """
    # last_key = sorted(conso_dict)[-1]
    return conso_dict[sorted(conso_dict)[-1]].alloc


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

  filename = "OUT_CONSO_BILAN_full"

  conso_dict = ConsoDict()

  with open(filename, "r") as filein:
    for ligne in filein:
      if ligne.split():
        conso_dict.add_daily_conso(
          ligne.split()[0],
          ligne.split()[2],
          ligne.split()[5],
          ligne.split()[8],
          ligne.split()[10]
        )

  # ... Initialization ...
  # ----------------------
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
    ymax = conso_dict.get_last_alloc()
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
  ax_conso.set_ylabel("Cumul (heures)", fontweight="bold")
  ax_theo.set_ylabel("Conso (%)", fontweight="bold")

  # ... Main title and legend ...
  # -----------------------------
  title = "Consommation gencmip6 (janvier-juin 2015)"
  ax_conso.set_title(title, fontweight="bold", size="large")
  ax_theo.legend(loc="best", fontsize="x-small", frameon=False)
  ax_conso.legend(loc="upper left", fontsize="x-small", frameon=False)

  # Be sure that everything is visible
  # plt.tight_layout()

  fig.savefig("conso.png", dpi=200)
  # fig.show()

  exit()
