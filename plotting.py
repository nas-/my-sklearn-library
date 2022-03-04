import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 125
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

"""
Dedicated to plotting functionalities

"""


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plots a correlation matrix of Frame
    :param df:
    """
    corr_mat = df.corr()
    plt.figure(figsize=(15, 10))
    mask = np.triu(corr_mat)
    sns.heatmap(corr_mat, cmap='RdBu_r', center=0.0, square=True, mask=mask)
    plt.title('PDF Correlations', fontsize=16)
    plt.show()


def plot_correlations(df: pd.DataFrame, cols: list) -> None:
    """
    Plot pairplots between columns = cols
    :param df:
    :param cols:
    """
    sns.set()
    sns.pairplot(df[cols], height=2.5)
    plt.show()


def plot_check_normality(df: pd.DataFrame, col: str):
    """
    @url=https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    Check normality. Positive skewedness, NP.log. If many zeros, dummy variable (0/1) + np log where is not 0
    :param df: pd.Dataframe to check
    :param col: Column to analyze
    """
    # sns.distplot(df[col], fit=norm)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    sns.histplot(df[col], ax=ax)
    ax = fig.add_subplot(1, 2, 2)
    res = stats.probplot(df[col], plot=ax)
    plt.show()


def plot_kdeplot_for_features(df: pd.DataFrame, maxcols: int = 40):
    """
    Plots Kdeplots for ncols features in df.
    :param df:
    :param maxcols:
    """
    # Visualizing first few rows
    numerical = df.columns[df.dtypes != "object"].to_numpy()
    cols_to_plot = numerical[:maxcols]
    rows, cols = int(np.ceil(len(cols_to_plot) / 4)), 4
    fig = plt.figure(figsize=(20, rows * 10))

    # fig = plt.figure(figsize=(20, 50))
    # rows, cols = 10, 4
    for idx, num in enumerate(numerical[:maxcols]):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.grid(alpha=0.7, axis="both")
        sns.kdeplot(x=num, fill=True, color='#50B2C0', linewidth=0.6, data=df, label=num)
        ax.set_xlabel(num)
        ax.legend()
    fig.tight_layout()
    fig.show()


def plot_boxplots_for_features(df: pd.DataFrame, target: str):
    """
    @url=https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python?scriptVersionId=19403046&cellId=24
    plots boxplots for n features
    :param target:
    :param df:
    """
    columns = [col for col in df.columns if col != target]
    rows, cols = int(np.ceil(len(columns) / 2)), 2
    fig = plt.figure(figsize=(20, rows * 10))
    for idx, col in enumerate(columns):
        single_data = pd.concat([df[target], df[col]], axis=1)
        ax = fig.add_subplot(rows, cols, idx + 1)
        sns.boxplot(x=col, y=target, data=single_data)
    # fig.axis(ymin=0, ymax=800000)
    # plt.xticks(rotation=90)
    fig.tight_layout()
    fig.show()


def plot_periodogram(ts: pd.Series, detrend='linear', ax=None):
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
