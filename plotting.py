import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 125
pd.options.display.max_columns = None
import matplotlib.pyplot as plt


def plotc(c1, c2, array):
    fig = plt.figure(figsize=(16, 8))
    sel = np.array(list(array))
    # array=train.Cover_Type.values

    plt.scatter(c1, c2, c=sel, s=100)
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)
