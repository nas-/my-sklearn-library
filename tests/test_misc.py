from unittest import TestCase
from misc import reduce_mem_usage
import pandas as pd


class Test(TestCase):
    def test_reduce_mem_usage(self):
        df = pd.DataFrame([(.2, .3, .01, .6), (0.2, .5, 1.6, .1), (.6, .0, .4, .4), (.2, .1, .7, .7)],
                          columns=['dogs', 'cats', 'kek', 'rew'])
        df = reduce_mem_usage(df)
        self.assertTrue(df.dtypes.cats == 'float32')
