import numpy as np
import pandas as pd

from .result import Results
from .statistic import Statistic


def boot(data, statistic, group_cols=None, output_cols=None, r=10000, **kwargs):
    """Creates UnivariateResult or MultivariateResult objects.

        :param data: input data to be bootstrapped
        :param statistic: statistic to be calculated
        :param group_cols: group columns for data, default is none
        :param output_cols: output columns for data, default is none
        :param r: number of bootstrap replications
        :type data: np.array, pd.Series, np.DataFrame
        :type statistic: Statistic, function
        :type group_cols: list
        :type output_cols: list
        :type r: int
        :return: results from the bootstrap simulation
        :rtype: Results

    """
    if isinstance(statistic, Statistic):
        func = lambda *args: statistic.func(*args, **kwargs)
        if not statistic.is_valid:
            raise Warning("results from bootstrap may not be valid")
    else:
        func = lambda *args: statistic(*args, **kwargs)

    r = int(r)  # make sure r is an int
    results = []
    sample_size = len(data)

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            for _ in range(r):
                boot_sample = np.random.choice(data, sample_size, replace=True)
                results.append(func(boot_sample))
            return Results(results, func, func(data), data)
        else:
            raise Exception("numpy array must contain only 1 variable")
    elif isinstance(data, pd.Series):
        for _ in range(r):
            boot_sample = data.sample(n=sample_size, replace=True)
            results.append(func(boot_sample))
        return Results(results, func, func(data), data)
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True).as_matrix()
                results.append(func(boot_sample))
            return Results(results, func, func(data), data)
        else:
            if group_cols is None:
                for _ in range(r):
                    # frac=1 to return sample of same size
                    boot_sample = data.sample(frac=1, replace=True)
                    if output_cols is None:
                        results.append(func(boot_sample))
                    else:
                        X = boot_sample[list(boot_sample.columns.difference(output_cols))]
                        if len(output_cols) == 1:
                            y = boot_sample[output_cols[0]]
                        else:
                            y = boot_sample[output_cols]
                        current_res = func(X, y)
                        results.append(current_res)
                if output_cols is None:
                    return Results(results, func, func(data), data)
                else:
                    X = data[data.columns.difference(output_cols)]
                    if len(output_cols) == 1:
                        y = boot_sample[output_cols[0]]
                    else:
                        y = boot_sample[output_cols]
                    return Results(results, func, func(X, y), data, output_cols=output_cols)
            else:
                indices = data.reset_index().groupby(group_cols)["index"].apply(list).to_dict()
                grouped_data = {}
                for key, val in indices.items():
                    grouped_data[key] = data.loc[val][data.loc[val].columns.difference(group_cols)]
                observed = np.asarray(func(*list(grouped_data.values())))
                for _ in range(r):
                    current_iter = []
                    for _, val in grouped_data.items():
                        boot_sample = val.sample(frac=1, replace=True)
                        current_iter.append(boot_sample)
                    if len(observed) == 1:
                        current_res = func(*current_iter)[0]
                    else:
                        current_res = func(*current_iter)
                    results.append(current_res)
                return Results(results, func, observed, data, group_cols=group_cols)
    else:
        raise Exception("data type not supported")


