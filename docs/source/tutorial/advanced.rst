.. _advanced: 

***********************************
Advanced Tutorial
***********************************
**resample** provides other features when performing bootstraps:
handling multivariate data and validity checks.

Multivariate Data 
===================================
**resample** allows many different kinds of multivariate data. The one
thing that they all have in common is that the dataset needs to be passed into
``boot`` as a ``pd.DataFrame``.

Matrix Statistics
-----------------------------------
The simplest case would be to calculate a statistic that looks at how each 
variable is dependent on one another such as the covariance or correlation :: 

    multi_bootstrap = rs.boot(score[["mec", "vec"]], np.cov)

Performing the actual bootstrap is almost the same as the univariate case.
The only difference is that the input data has multiple variables, thus 
the calculations are handling matrices rather than atomic values. For example,
calculating a point estimate would now return a 2x2 matrix rather than a single
value ::

    multi_bootstrap.point_estimate(np.mean)

The only times this changes is when plotting the bootstrap distribution or when
calculating confidence intervals. In either of these two situations, the
``col`` and ``row`` arguments will need to be specified to denote which
specific value to analyze ::

    multi_bootstrap.plot(col=0, row=1)

.. plot::

    import pandas as pd
    import numpy as np
    import resample as rs

    data = pd.read_csv("score.csv")
    bootstrap = rs.boot(data[["mec", "vec"]], np.cov)
    bootstrap.plot(col=0, row=1)

::
    
    multi_bootstrap.ci(col=0 row=0)

Grouping Variables
-----------------------------------
**resample** also allows the user to calculate statistics that compare
different groups. For example, a user may want to look at the difference in
means between two groups ::

    def diff_mean(group1, group2):
        return np.mean(group1) - np.mean(group2)
        
The ``boot`` function handles this by specifying the ``group_cols`` argument; a
``list`` with the column names that specify the different groups should be
passed. In the example below, we want to get a bootstrap estimate of the
difference beween the average algebra and statistics scores. This will also
require changing the structure of the score dataset to contain a column that
specifies an observation's group ::

    group_cols = ["alg", "sta"]
    data = score[group_cols].melt(value_vars=group_cols, var_name="test")
    boot_groups = rs.boot(data, diff_mean, group_cols=["test"])

Specifying ``group_col`` will return the same type of object as before; the
functionality remains the same. 

Output Variables
-----------------------------------
A user can also use **resample** to bootstrap situations that specify specific
dependent and independent variables such as estimating regression
coefficients ::

    from sklearn.linear_model import LinearRegression

    def get_coefs(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_

    boot_reg = rs.boot(hormone, get_coefs, output_cols=["amount"])

Like all of the previous examples, this will return the same time of object as
before.

Validity Checks using Statistics Objects
========================================
Certain estimators will not be valid to bootstrap. This includes statistics
like the maximum and minimum. resample solves this problem by building up
estimators using Statistics objects. 

These objects contain common statistics and hold information on whether
they are valid to bootstrap or not. More complicated estimators can be 
created by adding, subtracting, etc. with other estimators and numeric values.
After being created, they need to be passed into the *boot* function
inplace of a function.

If someone wanted to look at the average of the mean and median, they would
need to ::

    estimator = (rs.Mean() + rs.Median()) / 2
    bootstrap = rs.boot(data["mec"], estimator)

This particular case uses a valid estimator, so resample will not give a
warning. 

Using the max on the otherhand would cause resample to give a warning ::

    estimator = rs.Max()
    bootstrap = rs.boot(data["mec"], estimator)
    # would raise a python Warning: "results from bootstrap may not be valid

This feature can be completely bypassed if a user wants to proceed with the
bootstrap anyway.



