---
title: 'resample: Simple bootstraping for Python'
tags:
  - Python
  - statistics
  - bootstrap
  - simulation
authors:
  - name: Robert Cenon
    orcid: 0000-0002-2721-1292
    affiliation: 1
affiliations:
 - name: California Polytechnic State University, San Luis Obispo
   index: 1
date: 15 June 2018
bibliography: paper.bib
---

# Summary

Single value estimates do not provide enough information to describe a pattern; error will affect the accuracy of an estimate. Traditionally, a formula for the standard error of an estimator needs to be derived to find a numerical approximation of that accuracy. In the case of the sample mean, one of the most common estimators, a closed-form equation for its standard error exists. The problem with this approach is that even with other common estimators such as the sample median, no such equation exists (Efron 1993).

The bootstrap is a statistical technique for estimating the uncertainty of a statistic developed by Bradley Efron in 1979. Its main benefit is that it is a simple algorithm that works for almost any statistic. 

``resample`` is a Python package that creates a convenient framework for performing nonparametric bootstraps. This package allows the user to easily specify a variety of different bootstraps, aiming to fill in the gap of a general use bootstrap package in Python. Various other packages exist for bootstraps, but are built with specific cases in mind. ``resample`` achieves this as well as including the following features:

- Calculation of various confidence intervals (i.e. Efron, Percentile, BCa)
- Method calls to calculate standard error, point estimates, and bias
- Quick plotting of bootstrap distributions
- User defined estimators with validity checks
- Specification of more complex bootstraps

# Example

resample allows the user to define and run a bootstrap in one line. In this case, we will bootstrap the variance of the “mec” column from a DataFrame:

```python
bootstrap = rs.boot(data[“mec”], np.var)
```

This will create a results object that contains the observed value from the original data, the results of simulation, methods for various estimates, plotting functionality, etc. For example, a plot of the simulation can be created by:

```python
bootstrap.plot()
```
![Example of basic plots](mec_var.png)

resample can calculate many different confidence intervals by specifying the kind argument:

```python
bootstrap.ci() #efron interval by default
bootstrap.ci(kind=”loc_percentile”)
bootstrap.ci(kind=”BCa”)
```

Full documentation with a tutorial can be accessed at <https://cenonn.github.io/resample/>.

# References