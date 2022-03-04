from unittest import TestCase
import pandas as pd
from plotting import *

df = pd.DataFrame([(.2, .3, .01, .6), (0.2, .5, 1.6, .1), (.6, .0, .4, .4), (.2, .1, .7, .7)],
                  columns=['dogs', 'cats', 'kek', 'rew'])


class Test(TestCase):
    def test_plot_correlation_matrix(self):
        try:
            plot_correlation_matrix(df)
        except:
            self.fail()

    def test_plot_correlations(self):

        try:
            plot_correlations(df, ['dogs', 'cats'])
        except:
            self.fail()

    def test_plot_check_normality(self):
        try:
            plot_check_normality(df, 'cats')
        except:
            self.fail()

    def test_plot_kdeplot_for_features(self):
        try:
            plot_kdeplot_for_features(df, 5)
        except:
            self.fail()

    def test_plot_boxplot_for_features(self):
        try:
            plot_boxplots_for_features(df, 'cats')
        except:
            self.fail()
