import numpy as np
import pandas as pd

from .metric import UnivariateMetric
from .metric import MultivariateMetric


def boot(data, statistic, r=10000):
    r=int(r)
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

