import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


class Metric:
    def __init__(self, results, statistic):
        self.results = results
        self.statistic = statistic

    def se(self):
        return np.array(self.results).std(ddof=1)
    
    def plot(self):
        plt.hist(self.results, bins=30)
        
