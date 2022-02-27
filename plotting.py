import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 125
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

def plot_correlation_matrix(frame: pd.DataFrame) -> None:
    """
    Plots a correlation matrix of Frame
    :param frame:
    """
    corr_mat = frame.corr()
    plt.figure(figsize=(15, 10))
    mask = np.triu(corr_mat)
    sns.heatmap(corr_mat, cmap='RdBu_r', center=0.0, square=True, mask=mask)
    plt.title('PDF Correlations', fontsize=16)
    plt.show()


def plot_correlations(frame, cols):
    """
    Plot pairplots between columns = cols
    :param frame:
    :param cols:
    """
    sns.set()
    sns.pairplot(frame[cols], size=2.5)
    plt.show()


def plot_check_normality(frame, col):
    """
    Check normality. Positive skewedness, NP.log. If many zeros, dummy variable (0/1) + np log where is not 0
    :param frame: pd.Dataframe to check
    :param col: Column to analyze
    """
    # sns.distplot(frame[col], fit=norm)
    sns.displot(frame[col])
    fig = plt.figure()
    res = stats.probplot(frame[col], plot=plt)


def plot_kdeplot_for_features(df:pd.DataFrame, ncols: int = 40):
    """
    Plots Kdeplots for ncols features in dataframe.
    :param df:
    :param ncols:
    """
    # Visualizing first few rows
    numerical = df.columns[df.dtypes != "object"].to_numpy()
    fig = plt.figure(figsize=(20, 50))
    rows, cols = 10, 4
    for idx, num in enumerate(numerical[:ncols]):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.grid(alpha=0.7, axis="both")
        sns.kdeplot(x=num, fill=True, color='#50B2C0', linewidth=0.6, data=df, label="Train")
        ax.set_xlabel(num)
        ax.legend()
    fig.tight_layout()
    fig.show()


def plot_boxplots_for_features(data:pd.DataFrame):
    """
    plots boxplots for n features
    :param data:
    """
    fig = plt.figure(figsize=(20, 50))
    rows, cols = 10, 2
    for idx in range(4):
        y = idx
        ax = fig.add_subplot(rows, cols, idx+1)
        ax.grid(alpha = 0.7, axis ="both")
        sns.boxplot(x="target",y=y,data=data)
        ax.set_xlabel(y, fontsize=14)
        ax.legend()
        plt.xticks(rotation=60, fontsize=14)
        plt.yticks(fontsize=14)
    fig.tight_layout()
    fig.show()


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
