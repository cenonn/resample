import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


class UnivariateMetric:
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        return self.results.std(ddof=1)

    def point_estimate(self, center):
        return center(self.results)
    
    def plot(self, bins=30):
        plt.hist(self.results, bins=bins)

    def ci(self, confidence=0.95, kind='efron'):
        quantile = 100*(1 - confidence)/2
        res = self.results
        L, U = np.percentile(res, quantile), np.percentile(res, 100 - quantile)
        if kind == 'efron':
            return L, U
        elif kind == 'loc_percentile':
            return 2*self.observed - U, 2*self.observed - L
        elif kind == 'scale_percentile':
            return (self.observed ** 2 / U, self.observed ** 2 / L)
        else:
            raise Exception("unsupported ci type")
            
class MultivariateMetric:
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        return np.apply_along_axis(np.std, 0, self.results, ddof=0)
    
    def point_estimate(self, center):
        return np.apply_along_axis(center, 0, self.results)

    def plot(self, col, bins=30):
        plt.hist(self.results[:,col], bins=bins)

    def ci(self, col, confidence=0.95, kind='efron'):
        quantile = 100*(1 - confidence)/2
        res = self.results[:,col]
        L, U = np.percentile(res, quantile), np.percentile(res, 100 - quantile)
        if kind == 'efron':
            return L, U
        elif kind == 'loc_percentile':
            return 2*self.observed - U, 2*self.observed - L
        elif kind == 'scale_percentile':
            return (self.observed ** 2 / U, self.observed ** 2 / L)
        else:
            raise Exception("unsupported ci type")
            
