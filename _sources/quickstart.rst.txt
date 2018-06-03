.. _quickstart:

Getting Started - What is resample
-----------------------------------
Resampling in statistics refers to any method that draws repeated samples from
the original dataset. This package implements one of the most well known
resampling algorithms, the bootstrap. 

Bootstrapping is a statistical technique for estimating the uncertainty of a
statistic developed by Bradley Efron in 1979. This technique is extremely
useful when no simple closed-form formula for the standard error or the
distribution exists. **resample** provides a simply way of performing these
simulations.

Installation with git
===================================
**resample** is designed to be used alongside `Anaconda
<http://continuum.io/downloads>`_ but has to be installed seperately. Download
the git repository using the following unix command::
    
    git clone https://github.com/cenonn/resample.git

Go to the **resample** directory and enter::
    
    python setup.py install

After the install is complete, place the following line in any python
environment to use **resample**::

    import resample as rs



.. toctree::
    :maxdepth: 2
    :caption: Contents:
