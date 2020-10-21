import datetime
import numpy
import matplotlib
from matplotlib import pyplot, cm
import pandas
import textwrap
import typing

import arviz
import pymc3

from . import assumptions
from . import model
from . import preprocessing



def plot_testcount_forecast(
    result: pandas.Series,
    m: preprocessing.fbprophet.Prophet,
    forecast: pandas.DataFrame,
    considered_holidays: preprocessing.NamedDates, *,
    ax: matplotlib.axes.Axes=None
) -> matplotlib.axes.Axes:
    """ Helper function for plotting the detailed testcount forecasting result.

    Parameters
    ----------
    result : pandas.Series
        the date-indexed series of smoothed/predicted testcounts
    m : fbprophet.Prophet
        the prophet model
    forecast : pandas.DataFrame
        contains the prophet model prediction
    holidays : dict of { datetime : str }
        dictionary of the holidays that were used in the model
    ax : optional, matplotlib.axes.Axes
        an existing subplot to use

    Returns
    -------
    ax : matplotlib.axes.Axes
        the (created) subplot that was plotted into
    """
    if not ax:
        _, ax = pyplot.subplots(figsize=(13.4, 6))
    m.plot(forecast[forecast.ds >= m.history.set_index('ds').index[0]], ax=ax)
    ax.set_ylim(bottom=0)
    ax.set_xlim(pandas.to_datetime('2020-03-01'))
    plot_vlines(ax, considered_holidays, alignment='bottom')
    ax.legend(frameon=False, loc='upper left', handles=[
        ax.scatter([], [], color='black', label='training data'),
        ax.plot([], [], color='blue', label='prediction')[0],
        ax.plot(result.index, result.values, color='orange', label='result')[0],
    ])
    ax.set_ylabel('total tests')
    ax.set_xlabel('')
    return ax


