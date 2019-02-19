#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

def get_hist(df, x_axis, y_axis, titre, colour, font_size=None, horizontal=False):
    if horizontal:
        hist = df.plot.barh(x=x_axis, y=y_axis, color=colour, title =titre, fontsize = font_size, edgecolor = "none").get_figure()
    else:
        hist = df.plot.bar(x=x_axis, y=y_axis, color=colour, title =titre, fontsize = font_size, edgecolor = "none").get_figure()
    path_fig = "img/"+titre+'.png'
    hist.savefig(path_fig,  bbox_inches="tight")