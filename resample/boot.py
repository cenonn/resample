import numpy as np
import pandas as pd

from .result import UnivariateResult
from .result import MultivariateResult


def boot(data, statistic, r=10000):
    """Creates UnivariateMetric or MultivariateMetric objects.
    
    Args:
        data (np.array, pd.Series, np.DataFrame): input data to be bootstrapped
        statistic (function): statistic to be calculated
        r (int): number of bootstrap replications
    
    Returns:
        Metric objects used to get bootstrap estimates
    """
    r=int(r) #make sure r is an int
    results = []
    sample_size = len(data)
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            for _ in range(r):
                boot_sample = np.random.choice(data, sample_size, replace=True)
                results.append(statistic(boot_sample))
            return UnivariateMetric(results, statistic, statistic(data))
        else:
            raise Exception("numpy array must contain only 1 variable")
    elif isinstance(data, pd.Series): 
        for _ in range(r):
            boot_sample = data.sample(n=sample_size, replace=True)
            results.append(statistic(boot_sample))
        return UnivariateMetric(results, statistic, statistic(data))
    elif isinstance(data, pd.DataFrame): 
        if len(data.columns) == 1:
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True).as_matrix()
                results.append(statistic(boot_sample))
            return UnivariateMetric(results, statistic, statistic(data))
        else: 
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True)
                results.append(boot_sample.apply(statistic))
            return MultivariateMetric(results, statistic, statistic(data))
    else:
        raise Exception("data type not supported")


