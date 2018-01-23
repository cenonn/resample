import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


class UnivariateMetric:
    """Defines an object for univariate bootstraps

        Attributes:
            results (np.array): array that contains the estimate
                for every bootstrap replication
            statistic (function): statistic to be calculated
            observed (int, float): the sample statistic from the original data
    """
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        """A function that takes no arguments and returns the standard deviation
            of the statistic.
        """
        return self.results.std(ddof=1)

    def bias(self):
        """A function that takes no arguments and returns the bias
            of the statistic.
        """
        return np.mean(self.results) - observed

    def point_estimate(self, center, correction=False):
        """Calculate a point estimate from the bootstrap distribution
            using center as a point estimate.

        Args:
            center (function): measure of center to use
            correction (boolean): if True, apply bias correction

        Returns:
            A numeric estimate of the center
        """
        if not correction:
            return center(self.results)
        else:
            return center(self.results) - bias()
    
    def plot(self, bins=30):
        """Create a histogram of the bootstrap distribution

        Args:
            bins (int): the number of bins for the histogram
        """
        bins = int(bins)
        plt.hist(self.results, bins=bins)

    def ci(self, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic

        Args:
            confidence (float): the confidence level of the estimate, 
                between 0.0 and 1.0
            kind (string): type of interval to calculate, either efrom or
                percentile intervals
        
        Returns:
            A tuple of the interval
        """
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
    """Defines an object for multivariate bootstraps

        Attributes:
            results (np.array): array that contains the estimate
                for every bootstrap replication for every variable
            statistic (function): statistic to be calculated
            observed (np.array): the sample statistics from the original data
    """
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        """A function that takes no arguments and returns an array
            that contains the standard deviation of the statistic 
            for all variables.
        """
        return np.apply_along_axis(np.std, 0, self.results, ddof=0)

    def bias(self):
        """A function that takes no arguments and returns an array
            that contains the bias of the statistic for all variables.
        """
        return np.mean(self.results) - observed
    
    def point_estimate(self, center):
        """Calculate a point estimate of each variable

        Args:
            center (function): mesure of center to use
            correction (boolean): if True, apply bias correction

        Returns:
            An array with a point estimate for each variable
        """
        if not correction:
            return np.apply_along_axis(center, 0, self.results)
        else:
            return np.apply_along_axis(center, 0, self.results) - bias()

    def plot(self, col, bins=30):
        """Create a histogram of one of the bootstrap distribution 

        Args:
            col (int): the index of the variable to plot
            bins (int): the number of bins for the histogram
        """
        col = int(col)
        bins = int(bins)
        plt.hist(self.results[:,col], bins=bins)

    def ci(self, col, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic for one variable

        Args:
            col (int): the index of the variable
            confidence (float): the confidence level of the estimate, 
                between 0.0 and 1.0
            kind (string): type of interval to calculate, either efrom or
                percentile intervals
        
        Returns:
            A tuple of the interval
        """
        col = int(col)
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
            
