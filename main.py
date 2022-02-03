from costum_transformers import BoostedHybrid
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np

a = BoostedHybrid(LinearRegression(), LogisticRegression())
print(a.__name__)


