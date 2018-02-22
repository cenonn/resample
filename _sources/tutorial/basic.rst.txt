Basic Tutorial
===================================

boot()
-----------------------------------
The boot() function encompasses the actual bootstrap algorithm and is the
fundamental piece of the package. In order to perform a univariate 
bootstrap, a dataset and the estimator to be calculated are passed
into the function::
    
    import numpy as np
    import pandas as pd
    import resample as rs

    data = pd.read_csv("car_weights.data", header=None)
    bootstrap = rs.boot(data, np.mean)

The boot function will then perform 10,000 boostrap simulations by 
default. This returns a UnivariateResults object.

A few things to note:

#. The dataset needs to be either an np.array, pd.Series, or
   univariate pd.Dataframe.
#. The function passed can be any function, user-defined or 
   otherwise, as long as it returns a single value.


Results Objects
-----------------------------------
Results objects hold the results from all the boostrap simulations, the
estimator that was calculated, and the sample statistic. The primary use of
this object will usually be to calculate the standard error of our estimate::

    bootstrap.se()

It can also be used to plot a histograme of the bootstrap distribution::

    bootstrap.plot()



