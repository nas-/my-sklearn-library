import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 125
pd.options.display.max_columns = None
import matplotlib.pyplot as plt


def plotc(c1, c2, array):
    fig = plt.figure(figsize=(16, 8))
    sel = np.array(list(array))
    plt.scatter(c1, c2, c=sel, s=100)
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)


def plot_periodogram(ts, detrend='linear', ax=None):
    """
    How many Fourier pairs should we actually include in our feature set?
    We can answer this question with the periodogram.
    The periodogram tells you the strength of the frequencies in a time series.
    Specifically, the value on the y-axis of the graph is (a ** 2 + b ** 2) / 2, where a and b are the coefficients of the sine and cosine at that frequency (as in the Fourier Components)
    @url=https://www.kaggle.com/hiro5299834/tps-jan-2022-blend-stacking-models#What-is-Seasonality?
    :param ts: Target timeseries, with time index.
    :param detrend:
    :param ax:
    :return:
    """

    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
